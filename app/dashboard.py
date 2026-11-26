import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import savgol_filter

from data_analysis import MAX_ORIENTATION_DRIFT_DEG, analyse_start, load_imu, lowpass
from push_detection import DEFAULT_DIRECTION, find_pushes, refine_offset, valid_frames
from video_analysis import load_tracking
from video_player import VideoPlayer

IMU_FILES = ("imu_raw.csv", "imu.csv")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
NOISE_WINDOW_S = 0.3   # video noise = raw position minus a Savitzky-Golay smoothing over this window
JUMP_PX = 50           # residual above this = tracking jump (wrong blob / lost marker); fast real
                       # motion stays below ~30 px, measured jumps were 100-200 px
IMU_STILL_S = (0.3, 1.8)   # still window after the IMU start (same as analyse_start's default)
IMU_LOWPASS_HZ = 15.0      # same low-pass as analyse_start
MIN_SYNC_R = 0.9   # below this video and IMU do not show the same movement: sync not trusted (good recordings: 0.98-1.0)


def find_analyses(root):
    # Saved analysis folders (those with a tracking.csv or an IMU file), newest first
    root = Path(root)
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir()
             if p.is_dir() and ((p / "tracking.csv").exists() or any((p / f).exists() for f in IMU_FILES))]
    return sorted(found, key=lambda p: p.name, reverse=True)


def load_analysis(folder):
    # Read everything that exists in the folder; missing parts are None
    folder = Path(folder)
    data = {"folder": folder, "tracking": None, "imu": None, "imu_result": None,
            "imu_info": None, "imu_error": None, "markers": [], "sync": None, "pushes": []}
    if (folder / "tracking.csv").exists():
        data["tracking"] = load_tracking(folder / "tracking.csv")   # video signals end where the sled leaves the frame
    meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
    offset = meta.get("imu_offset_s") or 0.0   # IMU time 0 on the video time axis
    for name in IMU_FILES:
        if (folder / name).exists():
            try:
                data["imu"] = load_imu(folder / name)
                # The still window is relative to the IMU's own start, so analyse first ...
                data["imu_result"], data["imu_info"] = analyse_start(data["imu"])
            except Exception as e:   # unknown format, no start found, ...: show what we have
                data["imu_error"] = str(e)
            # ... then move everything onto the video time axis
            if data["imu"] is not None:
                data["imu"]["t"] += offset
            if data["imu_result"] is not None:
                data["imu_result"]["t"] += offset
                data["imu_info"]["t_start"] += offset
            break
    # Fine sync: slide the IMU position over the video position (see push_detection.refine_offset)
    if data["tracking"] is not None and data["imu_result"] is not None:
        try:
            shift, r0, r, direction = refine_offset(data["tracking"], data["imu_result"])
            data["sync"] = {"shift_s": shift, "r_before": r0, "r": r, "direction": direction,
                            "ok": r >= MIN_SYNC_R}
            if data["sync"]["ok"]:
                data["imu"]["t"] += shift
                data["imu_result"]["t"] += shift
                data["imu_info"]["t_start"] += shift
        except Exception as e:   # too short / no overlap: keep the computer-clock offset
            data["imu_error"] = f"sync: {e}"
    if data["tracking"] is not None:
        if data["sync"] and data["sync"]["ok"]:
            data["pushes"] = find_pushes(data["tracking"], data["imu_result"], data["sync"]["direction"])
        else:   # no IMU or video and IMU do not match: video only, find_pushes' default direction
            data["pushes"] = find_pushes(data["tracking"])
    data["video"] = next((p for p in sorted(folder.glob("video.*")) if p.suffix.lower() in VIDEO_EXTS), None)
    if (folder / "markers.json").exists():
        data["markers"] = json.loads((folder / "markers.json").read_text())
    if data["tracking"] is None and data["imu"] is None:
        raise ValueError(f"No tracking.csv or IMU file in {folder}")
    return data


def summarise(data):
    # Label -> value text for the KPI cards
    out = {}
    df = data["tracking"]
    if df is not None and len(df):
        t = df["t"].to_numpy()
        duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        tracked = df["x_filt"].notna()
        out["Frames"] = f"{len(df)}"
        out["Duration"] = f"{duration:.2f} s"
        out["Frame rate"] = f"{(len(df) - 1) / duration:.1f} fps" if duration > 0 else "-"
        out["Tracked"] = f"{100 * tracked.mean():.0f} %"
    result, info = data["imu_result"], data["imu_info"]
    if result is not None:
        out["IMU start"] = f"{info['t_start']:.2f} s"
        out["Max speed (IMU)"] = f"{result['v'].max():.2f} m/s"
        out["Max accel. (IMU)"] = f"{result['a_filt'].abs().max():.1f} m/s²"
    if data["sync"] is not None:
        if data["sync"]["ok"]:
            out["IMU sync"] = f"{1000 * data['sync']['shift_s']:+.0f} ms (r {data['sync']['r']:.2f})"
            out["Forward in video"] = "+x (right)" if data["sync"]["direction"] > 0 else "-x (left)"
        else:
            out["IMU sync"] = f"FAILED (r {data['sync']['r']:.2f}), not applied"
    info = data["imu_info"]
    if info is not None and info.get("orientation_drift_deg") is not None             and info["orientation_drift_deg"] > MAX_ORIENTATION_DRIFT_DEG:
        out["IMU orientation"] = f"not settled ({info['orientation_drift_deg']:.1f}°), not used"
    pushes = data["pushes"]
    out["Pushes (detected)"] = f"{len(pushes)}"
    if pushes and pushes[0]["imu_ok"] is not None:
        out["Pushes seen by IMU"] = f"{sum(p['imu_ok'] for p in pushes)}/{len(pushes)}"
    out["Markers"] = f"{len(data['markers'])}"
    return out


def marker_names(df):
    # Marker names from the tracking columns, e.g. ["GREEN", "PINK"]
    return [c[:-len("_x_filt")] for c in df.columns if c.endswith("_x_filt") and c != "mid_x_filt"]


def video_noise(df):
    # {"GREEN_x": residual, ...}: raw marker position minus a smoothed version (px, NaN where
    # the marker was not found). What is left is the frame-to-frame jitter of the detection;
    # very fast real motion leaks in a little, so the still parts give the cleanest value.
    t = df["t"].to_numpy(float)
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 30.0
    window = max(5, int(round(NOISE_WINDOW_S * fps)) | 1)   # odd number of frames
    out = {}
    for name in marker_names(df):
        for axis in ("x", "y"):
            raw = df[f"{name}_{axis}"].to_numpy(float)
            found = ~np.isnan(raw)
            residual = np.full_like(raw, np.nan)
            if found.sum() > window:
                # Bridge short gaps for the filter only; residuals are kept where the marker was seen
                filled = np.interp(np.arange(len(raw)), np.flatnonzero(found), raw[found])
                residual[found] = (raw - savgol_filter(filled, window, 2))[found]
            out[f"{name}_{axis}"] = residual
    return out


def robust_std(values):
    # Standard deviation from the median absolute deviation: the typical jitter, not
    # dominated by a few frames where the tracking jumped (plain std is)
    v = values[np.isfinite(values)]
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def imu_filtered(result):
    # Gravity- and bias-free acceleration (sensor axes) after the 15 Hz low-pass, n x 3
    fs = 1.0 / np.median(np.diff(result["t"]))
    return lowpass(result[["lx", "ly", "lz"]].to_numpy(float), fs, fc=min(IMU_LOWPASS_HZ, 0.4 * fs))


def development_values(data):
    # Label -> value text for the Development tab (diagnostics, not shown in the Analysis tab)
    out = {}
    df = data["tracking"]
    tracked = df["x_filt"].notna() if df is not None and len(df) else None
    if tracked is not None and tracked.any():
        x = df["x_filt"][tracked]
        y = df["y_filt"][tracked]
        out["x range"] = f"{x.max() - x.min():.0f} px"
        out["y range"] = f"{y.max() - y.min():.0f} px"
        out["Path length"] = f"{np.hypot(x.diff(), y.diff()).sum():.0f} px"
    else:
        out["x range"] = out["y range"] = out["Path length"] = "no tracking data"
    info = data["imu_info"]
    out["IMU drift v_end"] = f"{info['v_end']:.2f} m/s" if info is not None else "no IMU analysis"

    if df is not None and len(df):
        noise = video_noise(df)
        for name in marker_names(df):
            rx, ry = noise[f"{name}_x"], noise[f"{name}_y"]
            if np.isfinite(rx).sum() < 2:
                out[f"Video noise {name}"] = "not found"
                continue
            sx, sy = robust_std(rx), robust_std(ry)
            jumps = int((np.hypot(rx, ry) > JUMP_PX).sum())
            out[f"Video noise {name}"] = f"x {sx:.2f} / y {sy:.2f} px"
            out[f"Tracking jumps {name}"] = f"{jumps} frames > {JUMP_PX} px"
        d = df["distance"].dropna()
        out["Marker distance"] = f"{d.mean():.1f} ± {d.std():.1f} px" if len(d) else "no data"
    result = data["imu_result"]
    if result is not None:
        t = result["t"].to_numpy()
        still = (t >= t[0] + IMU_STILL_S[0]) & (t <= t[0] + IMU_STILL_S[1])
        if still.any():
            std = result.loc[still, ["lx", "ly", "lz"]].std().to_numpy()
            out["IMU noise at rest"] = f"{np.linalg.norm(std):.3f} m/s² (before low-pass)"
    return out


def render_cards(frame, values, columns=6):
    # Replace the content of `frame` with one card per (label, value)
    for child in frame.winfo_children():
        child.destroy()
    for i, (label, value) in enumerate(values.items()):
        card = ttk.Frame(frame, relief="groove", padding=6)
        card.grid(row=i // columns, column=i % columns, sticky="ew", padx=3, pady=3)
        frame.columnconfigure(i % columns, weight=1)
        ttk.Label(card, text=label).pack(anchor="w")
        ttk.Label(card, text=value, font=("TkDefaultFont", 14, "bold")).pack(anchor="w")


class DevelopmentTab(ttk.Frame):
    # Diagnostic values for the recording that is open in the Analysis tab

    def __init__(self, master):
        super().__init__(master)
        self.title = ttk.Label(self, text="Open a recording in the Analysis tab to see its values here.")
        self.title.pack(anchor="w", padx=10, pady=10)
        self.cards = ttk.Frame(self)
        self.cards.pack(fill="x", padx=10)

        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.94, wspace=0.18, hspace=0.42)
        (self.ax_imu_raw, self.ax_imu_filt), (self.ax_noise, self.ax_dist) = self.fig.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def show(self, data):
        self.title.config(text=f"Recording: {data['folder'].name}")
        render_cards(self.cards, development_values(data), columns=4)
        self.show_plots(data)

    def show_plots(self, data):
        for ax in self.fig.axes:
            ax.clear()
            ax.grid(alpha=0.3)
            ax.set_xlabel("t (s)")

        def no_data(ax, title, text):
            ax.set_title(title)
            ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")

        colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
        imu, result = data["imu"], data["imu_result"]
        title = "IMU before filtering: raw acceleration (sensor axes, with gravity)"
        if imu is not None:
            self.ax_imu_raw.set_title(title, fontsize=10)
            for axis, c in colors.items():
                self.ax_imu_raw.plot(imu["t"], imu["a" + axis], color=c, lw=0.8, label="a" + axis)
            self.ax_imu_raw.set_ylabel("m/s²")
            self.ax_imu_raw.legend(fontsize=8, loc="upper right")
        else:
            no_data(self.ax_imu_raw, title, "no IMU file")

        title = f"IMU after filtering: gravity + bias removed, low-pass {IMU_LOWPASS_HZ:.0f} Hz"
        if result is not None:
            self.ax_imu_filt.set_title(title, fontsize=10)
            filt = imu_filtered(result)
            for i, (axis, c) in enumerate(colors.items()):
                self.ax_imu_filt.plot(result["t"], result["l" + axis], color=c, lw=0.6, alpha=0.3)
                self.ax_imu_filt.plot(result["t"], filt[:, i], color=c, lw=1.0, label="a" + axis)
            self.ax_imu_filt.set_ylabel("m/s²  (faint = before low-pass)")
            self.ax_imu_filt.legend(fontsize=8, loc="upper right")
        else:
            no_data(self.ax_imu_filt, title, data["imu_error"] or "no IMU file")

        df = data["tracking"]
        title = f"Video noise: raw position minus smoothed ({NOISE_WINDOW_S:.1f} s)"
        if df is not None and df["x_filt"].notna().any():
            self.ax_noise.set_title(title, fontsize=10)
            for name, residual in video_noise(df).items():
                self.ax_noise.plot(df["t"], residual, lw=0.7,
                                   ls="-" if name.endswith("_x") else ":", label=name)
            self.ax_noise.set_ylabel("px")
            self.ax_noise.legend(fontsize=8, loc="upper right", ncol=2)

            self.ax_dist.set_title("Distance between markers", fontsize=10)
            self.ax_dist.plot(df["t"], df["distance"], ".", ms=2, alpha=0.4, color="tab:green", label="centres, raw")
            self.ax_dist.plot(df["t"], df["distance_filt"], color="tab:green", lw=1, label="centres, Kalman")
            self.ax_dist.set_ylabel("px")
            self.ax_dist.legend(fontsize=8, loc="upper right")
        else:
            no_data(self.ax_noise, title, "no tracking data")
            no_data(self.ax_dist, "Distance between markers", "no tracking data")
        self.canvas.draw_idle()


class Dashboard(ttk.Frame):
    # Dashboard for one saved analysis: pick a folder, see KPI cards and plots

    def __init__(self, master, analyses_dir):
        super().__init__(master)
        self.analyses_dir = Path(analyses_dir)
        self.data = None
        self.on_load = None   # called with the loaded data after every load

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, text="Analysis:").pack(side="left")
        self.choice = tk.StringVar()
        self.combo = ttk.Combobox(controls, textvariable=self.choice, width=48, state="readonly")
        self.combo.pack(side="left", padx=5)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.load(self.analyses_dir / self.choice.get()))
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left", padx=5)
        ttk.Button(controls, text="Open folder...", command=self.browse).pack(side="left", padx=5)
        self.status = ttk.Label(controls, text="")
        self.status.pack(side="left", padx=15)

        self.cards = ttk.Frame(self)
        self.cards.pack(fill="x", padx=10)

        # One row per detected push: time, strength (largest forward speed), % of the strongest push
        columns = ("n", "t", "speed", "rel", "imu")
        self.push_table = ttk.Treeview(self, columns=columns, show="headings", height=5)
        for col, title, width in (("n", "#", 40), ("t", "t (s)", 80), ("speed", "max speed", 120),
                                  ("rel", "% of strongest", 110), ("imu", "seen by IMU", 90)):
            self.push_table.heading(col, text=title)
            self.push_table.column(col, width=width, anchor="e")
        self.push_table.pack(fill="x", padx=10, pady=(8, 0))
        self.push_table.bind("<<TreeviewSelect>>", self.on_push_selected)

        # Video player (left) and plots (right); the sash between them can be dragged
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=10)
        video_col = ttk.Frame(body)
        body.add(video_col, weight=2)
        # Above the video: forward direction of the sled in the image
        self.direction_label = tk.Label(video_col, text="", font=("TkDefaultFont", 20, "bold"))
        self.direction_label.pack(fill="x")
        self.player = VideoPlayer(video_col, on_frame=self.on_video_frame)
        self.player.pack(fill="both", expand=True)
        # Below the video: one button per detected push, jumps the video there
        self.push_buttons = ttk.Frame(video_col)
        self.push_buttons.pack(fill="x", pady=(4, 0))
        plots = ttk.Frame(body)
        body.add(plots, weight=3)

        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.fig.subplots_adjust(left=0.06, right=0.95, bottom=0.08, top=0.95, wspace=1.2, hspace=0.4)
        # Top row: video (markers), bottom row: IMU. Kept in separate charts for now.
        gs = self.fig.add_gridspec(2, 6)
        self.ax_pos = self.fig.add_subplot(gs[0, 0:3])
        self.ax_dist = self.fig.add_subplot(gs[0, 3:6])
        self.ax_imu_v = self.fig.add_subplot(gs[1, 0:2])
        self.ax_acc = self.fig.add_subplot(gs[1, 2:4])
        self.ax_gyro = self.fig.add_subplot(gs[1, 4:6])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # Current video frame: vertical line on every time plot, redrawn with blitting so it can
        # follow the playing video (full redraws are far too slow). Pattern from the matplotlib
        # docs, "Faster rendering by using blitting":
        # https://matplotlib.org/stable/users/explain/animations/blitting.html
        self.cursors = []
        self.cursor_dot = None
        self.cursor_t = None
        self.background = None
        self.canvas.mpl_connect("draw_event", self.on_draw)
        # Click on a plot: jump the video to that time
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        self.refresh()

    def refresh(self):
        # Re-scan analyses/ and load the newest one if nothing is shown yet
        folders = find_analyses(self.analyses_dir)
        self.combo["values"] = [p.name for p in folders]
        if folders and self.data is None:
            self.choice.set(folders[0].name)
            self.load(folders[0])
        elif not folders:
            self.status.config(text="No saved analyses yet")

    def browse(self):
        folder = filedialog.askdirectory(initialdir=self.analyses_dir, title="Select an analysis folder")
        if folder:
            self.choice.set(Path(folder).name)
            self.load(Path(folder))

    def load(self, folder):
        try:
            self.data = load_analysis(folder)
        except Exception as e:
            messagebox.showerror("Luge Start Tracker", str(e))
            return
        self.status.config(text=str(folder))
        self.show_cards(summarise(self.data))
        self.show_push_table()
        self.show_plots()
        self.load_video()
        self.show_direction()
        self.show_push_buttons()
        if self.on_load is not None:   # e.g. the Development tab follows the open recording
            self.on_load(self.data)

    def show_push_table(self):
        self.push_table.delete(*self.push_table.get_children())
        for n, p in enumerate(self.data["pushes"], 1):
            imu = {None: "-", True: "yes", False: "no"}[p["imu_ok"]]
            has_speed = np.isfinite(p["speed"])   # push strength comes from the IMU only
            self.push_table.insert("", "end", iid=str(n - 1), values=(
                n, f"{p['t']:.2f}", f"{p['speed']:.2f} m/s" if has_speed else "-",
                f"{p['rel']:.0f} %" if np.isfinite(p["rel"]) else "-", imu))

    def show_cards(self, values):
        render_cards(self.cards, values)

    def show_plots(self):
        d = self.data
        df = d["tracking"]
        for ax in self.fig.axes:
            ax.clear()
            ax.grid(alpha=0.3)

        def no_data(ax, title, text="no data"):
            ax.set_title(title)
            ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")

        if df is not None and df["x_filt"].notna().any():
            t = df["t"]
            self.ax_pos.set_title("Marker position")
            self.ax_pos.plot(t, df["x_filt"], color="tab:blue", label="x (Kalman)")
            self.ax_pos.plot(t, df["y_filt"], color="tab:orange", label="y (Kalman)")
            self.ax_pos.set_xlabel("t (s)")
            self.ax_pos.set_ylabel("position (px)")
            self.ax_pos.legend(fontsize=8, loc="upper left")
            # y range from the trustworthy frames, so a lost marker (Kalman running off) does not squash the plot
            ok = valid_frames(df)
            if ok.any():
                lo = min(df.loc[ok, "x_filt"].min(), df.loc[ok, "y_filt"].min())
                hi = max(df.loc[ok, "x_filt"].max(), df.loc[ok, "y_filt"].max())
                self.ax_pos.set_ylim(lo - 0.1 * (hi - lo), hi + 0.15 * (hi - lo))

            self.ax_dist.set_title("Distance between markers")
            self.ax_dist.plot(t, df["distance_filt"], label="Kalman")
            self.ax_dist.plot(t, df["distance"], ".", alpha=0.3, markersize=2, label="raw")
            self.ax_dist.set_xlabel("t (s)")
            self.ax_dist.set_ylabel("px")
            self.ax_dist.legend(fontsize=8)
        else:
            for ax, title in ((self.ax_pos, "Marker position"),
                              (self.ax_dist, "Distance between markers")):
                no_data(ax, title, "no tracking data")

        result = d["imu_result"]
        if result is not None:
            self.ax_imu_v.set_title("Speed (IMU, integrated)")
            self.ax_imu_v.plot(result["t"], result["v"], color="tab:red", lw=1.2)
            self.ax_imu_v.axhline(0, color="gray", lw=0.8)   # signed: < 0 = backwards
            self.ax_imu_v.set_xlabel("t (s)")
            self.ax_imu_v.set_ylabel("m/s")
            self.ax_acc.set_title("Forward acceleration (IMU)")
            self.ax_acc.plot(result["t"], result["a_fwd"], alpha=0.4, label="raw")
            self.ax_acc.plot(result["t"], result["a_filt"], label="low-pass")
            self.ax_acc.axvline(d["imu_info"]["t_start"], color="red", ls="--", label="start")
            self.ax_acc.set_xlabel("t (s)")
            self.ax_acc.set_ylabel("m/s²")
            self.ax_acc.legend(fontsize=8)
        else:
            no_data(self.ax_imu_v, "Speed (IMU, integrated)", d["imu_error"] or "no IMU file")
            no_data(self.ax_acc, "Forward acceleration (IMU)", d["imu_error"] or "no IMU file")

        imu = d["imu"]
        if imu is not None and imu[["gx", "gy", "gz"]].notna().any().any():
            self.ax_gyro.set_title("Angular velocity (IMU)")
            for c in ("gx", "gy", "gz"):
                self.ax_gyro.plot(imu["t"], imu[c], label=c)
            self.ax_gyro.set_xlabel("t (s)")
            self.ax_gyro.set_ylabel("rad/s")
            self.ax_gyro.legend(fontsize=8)
        else:
            no_data(self.ax_gyro, "Angular velocity (IMU)", "no gyroscope data")

        # Detected pushes: red triangles on the position, red lines on the IMU acceleration
        for p in d["pushes"]:
            if df is not None:
                y = np.interp(p["t"], df["t"], df["x_filt"])
                self.ax_pos.plot(p["t"], y, "rv")
                if np.isfinite(p["rel"]):
                    self.ax_pos.annotate(f"{p['rel']:.0f}%", (p["t"], y), textcoords="offset points",
                                         xytext=(0, 6), ha="center", fontsize=7, color="red")
            if p["imu_ok"] is not None:
                self.ax_acc.axvline(p["t"], color="red", ls="-" if p["imu_ok"] else ":", lw=1)

        # Cursor for the current video frame (animated: drawn only by draw_cursor, not by draw)
        self.background = None
        time_axes = (self.ax_pos, self.ax_dist, self.ax_imu_v, self.ax_acc, self.ax_gyro)
        self.cursors = [ax.axvline(np.nan, color="magenta", lw=1.5, animated=True) for ax in time_axes]
        self.cursor_dot = self.ax_pos.plot([], [], "o", color="magenta", markersize=7, animated=True)[0]

        # User markers as vertical lines on the time plots
        for m in d["markers"]:
            for ax in (self.ax_pos, self.ax_dist, self.ax_imu_v, self.ax_acc, self.ax_gyro):
                ax.axvline(m["t"], color="orange", ls=":")
            self.ax_pos.annotate(m.get("label", ""), (m["t"], 1), xycoords=("data", "axes fraction"),
                                 rotation=90, va="top", fontsize=8, color="darkorange")
        self.canvas.draw_idle()

    # --- video player <-> plots ---

    def load_video(self):
        d = self.data
        if d["video"] is None:
            self.player.close("no video in this analysis")
            return
        df = d["tracking"]
        overlay = None
        if df is not None and {"frame", "x_filt", "y_filt"} <= set(df.columns):
            pos = df.set_index("frame")[["x_filt", "y_filt"]]

            def overlay(i):
                # Tracked sled position of this frame, drawn on the video to check the tracking
                if i not in pos.index or pos.loc[i].isna().any():
                    return None
                return pos.loc[i, "x_filt"], pos.loc[i, "y_filt"]
        self.player.load(d["video"], tracking=df, overlay=overlay)

    def show_direction(self):
        # Arrow pointing the way the sled moves forward in the video
        sync = self.data["sync"]
        if self.data["tracking"] is None:
            self.direction_label.config(text="")
            return
        if sync is not None and sync["ok"]:
            direction, note, color = sync["direction"], "from IMU sync", "black"
        else:
            direction, note, color = DEFAULT_DIRECTION, "assumed, no IMU sync", "gray"
        arrow = "forward  ⟶" if direction > 0 else "⟵  forward"
        self.direction_label.config(text=f"{arrow}   ({note})", fg=color)

    def show_push_buttons(self):
        for child in self.push_buttons.winfo_children():
            child.destroy()
        pushes = self.data["pushes"]
        if not pushes:
            ttk.Label(self.push_buttons, text="No pushes detected").pack(side="left")
            return
        ttk.Label(self.push_buttons, text="Jump to:").pack(side="left", padx=(0, 4))
        for n, p in enumerate(pushes):
            ttk.Button(self.push_buttons, text=f"Push {n + 1} · {p['t']:.2f} s",
                       command=lambda n=n: self.jump_to_push(n)).pack(side="left", padx=2)

    def jump_to_push(self, n):
        self.player.pause()
        self.player.seek_time(self.data["pushes"][n]["t"])
        self.push_table.selection_set(str(n))   # highlight the same push in the table

    def on_video_frame(self, i, t):
        self.cursor_t = t
        self.draw_cursor()

    def on_plot_click(self, event):
        if event.inaxes is not None and event.xdata is not None:
            self.player.pause()
            self.player.seek_time(event.xdata)

    def on_push_selected(self, event):
        sel = self.push_table.selection()
        if sel and self.data is not None:
            self.player.pause()
            self.player.seek_time(self.data["pushes"][int(sel[0])]["t"])

    def on_draw(self, event):
        # Full redraw (new data, resize): keep the plots without cursor as the blit background
        self.background = self.canvas.copy_from_bbox(self.fig.bbox)
        self.draw_cursor(blit=False)

    def draw_cursor(self, blit=True):
        if not self.cursors or self.cursor_t is None:
            return
        t = self.cursor_t
        for line in self.cursors:
            line.set_xdata([t, t])
        df = self.data["tracking"] if self.data else None
        if df is not None and df["x_filt"].notna().any():
            ok = df["x_filt"].notna()
            self.cursor_dot.set_data([t], [np.interp(t, df["t"][ok], df["x_filt"][ok])])
        else:
            self.cursor_dot.set_data([], [])
        if blit:
            if self.background is None:   # plots not drawn yet; on_draw will add the cursor
                return
            self.canvas.restore_region(self.background)
        for artist in self.cursors + [self.cursor_dot]:
            self.fig.draw_artist(artist)
        if blit:
            self.canvas.blit(self.fig.bbox)
