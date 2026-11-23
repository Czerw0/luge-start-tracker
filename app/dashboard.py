import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_analysis import analyse_start, load_imu
from push_detection import find_pushes, refine_offset

IMU_FILES = ("imu_raw.csv", "imu.csv")


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
        data["tracking"] = pd.read_csv(folder / "tracking.csv")
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
            shift, r0, r = refine_offset(data["tracking"], data["imu_result"])
            data["imu"]["t"] += shift
            data["imu_result"]["t"] += shift
            data["imu_info"]["t_start"] += shift
            data["sync"] = {"shift_s": shift, "r_before": r0, "r": r}
        except Exception as e:   # too short / no overlap: keep the computer-clock offset
            data["imu_error"] = f"sync: {e}"
    if data["tracking"] is not None:
        data["pushes"] = find_pushes(data["tracking"], data["imu_result"])
    if (folder / "markers.json").exists():
        data["markers"] = json.loads((folder / "markers.json").read_text())
    if data["tracking"] is None and data["imu"] is None:
        raise ValueError(f"No tracking.csv or IMU file in {folder}")
    return data


def tracking_speed(df):
    # Speed in m/s: use the recorded column if present, else None (older recordings)
    if "speed_mps" in df and df["speed_mps"].notna().any():
        return df["speed_mps"]
    return None


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
        if tracked.any():
            x = df["x_filt"][tracked]
            y = df["y_filt"][tracked]
            out["x range"] = f"{x.max() - x.min():.0f} px"
            out["y range"] = f"{y.max() - y.min():.0f} px"
            step = np.hypot(x.diff(), y.diff())
            out["Path length"] = f"{step.sum():.0f} px"
        speed = tracking_speed(df)
        if speed is not None:
            out["Max speed (video)"] = f"{speed.max():.2f} m/s"
            out["Mean speed (video)"] = f"{speed.mean():.2f} m/s"
    result, info = data["imu_result"], data["imu_info"]
    if result is not None:
        out["IMU start"] = f"{info['t_start']:.2f} s"
        out["Max speed (IMU)"] = f"{result['v'].max():.2f} m/s"
        out["Max accel. (IMU)"] = f"{result['a_filt'].abs().max():.1f} m/s²"
        out["IMU drift v_end"] = f"{info['v_end']:.2f} m/s"
    if data["sync"] is not None:
        out["IMU sync"] = f"{1000 * data['sync']['shift_s']:+.0f} ms (r {data['sync']['r']:.2f})"
    pushes = data["pushes"]
    out["Pushes (detected)"] = f"{len(pushes)}"
    if pushes and "imu_pause" in pushes[0]:
        out["Pushes seen by IMU"] = f"{sum(p['imu_pause'] is not None for p in pushes)}/{len(pushes)}"
    out["Markers"] = f"{len(data['markers'])}"
    return out


class Dashboard(ttk.Frame):
    # Dashboard for one saved analysis: pick a folder, see KPI cards and plots

    def __init__(self, master, analyses_dir):
        super().__init__(master)
        self.analyses_dir = Path(analyses_dir)
        self.data = None

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, text="Analysis:").pack(side="left")
        self.choice = tk.StringVar()
        self.combo = ttk.Combobox(controls, textvariable=self.choice, width=48, state="readonly")
        self.combo.pack(side="left", padx=5)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.load(self.analyses_dir / self.choice.get()))
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left", padx=5)
        ttk.Button(controls, text="Open folder...", command=self.browse).pack(side="left", padx=5)
        ttk.Button(controls, text="Save pushes as markers", command=self.save_pushes).pack(side="left", padx=5)
        self.status = ttk.Label(controls, text="")
        self.status.pack(side="left", padx=15)

        self.cards = ttk.Frame(self)
        self.cards.pack(fill="x", padx=10)

        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.fig.subplots_adjust(left=0.06, right=0.95, bottom=0.08, top=0.95, wspace=1.2, hspace=0.4)
        # Top row: video (markers), bottom row: IMU. Kept in separate charts for now.
        gs = self.fig.add_gridspec(2, 6)
        self.ax_pos = self.fig.add_subplot(gs[0, 0:3])
        self.ax_dist = self.fig.add_subplot(gs[0, 3:6])
        self.ax_imu_v = self.fig.add_subplot(gs[1, 0:2])
        self.ax_acc = self.fig.add_subplot(gs[1, 2:4])
        self.ax_gyro = self.fig.add_subplot(gs[1, 4:6])
        # Position (px, left axis) and speed (m/s, right axis) share one chart
        self.ax_speed = self.ax_pos.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

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
        self.show_plots()

    def save_pushes(self):
        # Write the detected pushes to markers.json (replaces earlier auto pushes, keeps manual markers)
        if self.data is None:
            return
        folder = self.data["folder"]
        kept = [m for m in self.data["markers"] if m.get("source") != "auto"]
        auto = [{"t": round(p["t"], 3), "label": "push", "kind": "push", "source": "auto"}
                for p in self.data["pushes"]]
        (folder / "markers.json").write_text(json.dumps(kept + auto, indent=2))
        self.load(folder)
        self.status.config(text=f"Saved {len(auto)} pushes to {folder / 'markers.json'}")

    def show_cards(self, values):
        for child in self.cards.winfo_children():
            child.destroy()
        for i, (label, value) in enumerate(values.items()):
            card = ttk.Frame(self.cards, relief="groove", padding=6)
            card.grid(row=i // 6, column=i % 6, sticky="ew", padx=3, pady=3)
            self.cards.columnconfigure(i % 6, weight=1)
            ttk.Label(card, text=label).pack(anchor="w")
            ttk.Label(card, text=value, font=("TkDefaultFont", 14, "bold")).pack(anchor="w")

    def show_plots(self):
        d = self.data
        df = d["tracking"]
        for ax in self.fig.axes:
            ax.clear()
            if ax is not self.ax_speed:   # one grid per chart (the speed axis sits on top of position)
                ax.grid(alpha=0.3)
        self.ax_speed.yaxis.tick_right()
        self.ax_speed.yaxis.set_label_position("right")

        def no_data(ax, title, text="no data"):
            ax.set_title(title)
            ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")

        if df is not None and df["x_filt"].notna().any():
            t = df["t"]
            self.ax_pos.set_title("Marker position and speed")
            self.ax_pos.plot(t, df["x_filt"], color="tab:blue", label="x (Kalman)")
            self.ax_pos.plot(t, df["y_filt"], color="tab:orange", label="y (Kalman)")
            self.ax_pos.set_xlabel("t (s)")
            self.ax_pos.set_ylabel("position (px)")

            self.ax_dist.set_title("Distance between markers")
            self.ax_dist.plot(t, df["distance_filt"], label="Kalman")
            self.ax_dist.plot(t, df["distance"], ".", alpha=0.3, markersize=2, label="raw")
            self.ax_dist.set_xlabel("t (s)")
            self.ax_dist.set_ylabel("px")
            self.ax_dist.legend(fontsize=8)

            speed = tracking_speed(df)
            if speed is not None:
                self.ax_speed.plot(t, speed, color="black", lw=1.2, label="speed, video (m/s)")
        else:
            for ax, title in ((self.ax_pos, "Marker position and speed"),
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

        # Speed axis + one legend for both axes of the combined chart
        if self.ax_speed.lines:
            self.ax_speed.set_ylabel("speed (m/s)")
            self.ax_speed.set_ylim(bottom=0)   # video speed is a magnitude, never negative
        else:
            self.ax_speed.set_yticks([])
        handles = self.ax_pos.get_legend_handles_labels()
        more = self.ax_speed.get_legend_handles_labels()
        if handles[0] or more[0]:
            self.ax_speed.legend(handles[0] + more[0], handles[1] + more[1], fontsize=8, loc="upper left")

        # Detected pushes: red triangles on the position, red band = IMU pause around the push
        for p in d["pushes"]:
            if df is not None:
                self.ax_pos.plot(p["t"], np.interp(p["t"], df["t"], df["x_filt"]), "rv")
            if p.get("imu_pause"):
                self.ax_acc.axvspan(*p["imu_pause"], color="red", alpha=0.15)

        # User markers as vertical lines on the time plots
        for m in d["markers"]:
            for ax in (self.ax_pos, self.ax_dist, self.ax_imu_v, self.ax_acc, self.ax_gyro):
                ax.axvline(m["t"], color="orange", ls=":")
            self.ax_pos.annotate(m.get("label", ""), (m["t"], 1), xycoords=("data", "axes fraction"),
                                 rotation=90, va="top", fontsize=8, color="darkorange")
        self.canvas.draw_idle()
