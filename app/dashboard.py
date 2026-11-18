import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_analysis import analyse_start, load_imu

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
            "imu_info": None, "imu_error": None, "markers": []}
    if (folder / "tracking.csv").exists():
        data["tracking"] = pd.read_csv(folder / "tracking.csv")
    for name in IMU_FILES:
        if (folder / name).exists():
            try:
                data["imu"] = load_imu(folder / name)
                data["imu_result"], data["imu_info"] = analyse_start(data["imu"])
            except Exception as e:   # unknown format, no start found, ...: show what we have
                data["imu_error"] = str(e)
            break
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
    out["Markers"] = f"{len(data['markers'])}"
    return out


class Dashboard(ttk.Frame):
    # Dashboard for one saved analysis: pick a folder, see KPI cards and plots

    def __init__(self, master, analyses_dir):
        super().__init__(master)
        self.analyses_dir = Path(analyses_dir)
        self.data = None
        self.colorbar = None

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

        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.95, wspace=0.22, hspace=0.4)
        axes = self.fig.subplots(2, 3)
        (self.ax_pos, self.ax_traj, self.ax_dist), (self.ax_speed, self.ax_acc, self.ax_gyro) = axes
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
        if self.colorbar is not None:   # ax.clear() would leave the old colorbar axes behind
            self.colorbar.remove()
            self.colorbar = None
        for ax in self.fig.axes:
            ax.clear()
            ax.grid(alpha=0.3)

        def no_data(ax, title, text="no data"):
            ax.set_title(title)
            ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")

        if df is not None and df["x_filt"].notna().any():
            t = df["t"]
            self.ax_pos.set_title("Marker position (Kalman)")
            self.ax_pos.plot(t, df["x_filt"], label="x")
            self.ax_pos.plot(t, df["y_filt"], label="y")
            self.ax_pos.set_xlabel("t (s)")
            self.ax_pos.set_ylabel("px")
            self.ax_pos.legend(fontsize=8)

            self.ax_traj.set_title("Trajectory (image)")
            sc = self.ax_traj.scatter(df["x_filt"], df["y_filt"], c=t, s=4, cmap="viridis")
            self.ax_traj.invert_yaxis()   # image y grows downwards
            self.ax_traj.set_aspect("equal", adjustable="datalim")
            self.ax_traj.set_xlabel("x (px)")
            self.ax_traj.set_ylabel("y (px)")
            self.colorbar = self.fig.colorbar(sc, ax=self.ax_traj, label="t (s)")

            self.ax_dist.set_title("Distance between markers")
            self.ax_dist.plot(t, df["distance_filt"], label="Kalman")
            self.ax_dist.plot(t, df["distance"], ".", alpha=0.3, markersize=2, label="raw")
            self.ax_dist.set_xlabel("t (s)")
            self.ax_dist.set_ylabel("px")
            self.ax_dist.legend(fontsize=8)

            speed = tracking_speed(df)
            if speed is not None:
                self.ax_speed.set_title("Speed")
                self.ax_speed.plot(t, speed, label="video")
                self.ax_speed.set_ylabel("m/s")
            else:
                no_data(self.ax_speed, "Speed", "no speed in this recording")
        else:
            for ax, title in ((self.ax_pos, "Marker position"), (self.ax_traj, "Trajectory"),
                              (self.ax_dist, "Distance between markers"), (self.ax_speed, "Speed")):
                no_data(ax, title, "no tracking data")

        result = d["imu_result"]
        if result is not None:
            self.ax_speed.plot(result["t"], result["v"], label="IMU (integrated)")
            self.ax_speed.set_title("Speed")
            self.ax_speed.set_ylabel("m/s")
            self.ax_speed.set_xlabel("t (s)")
            self.ax_speed.legend(fontsize=8)
            self.ax_acc.set_title("Forward acceleration (IMU)")
            self.ax_acc.plot(result["t"], result["a_fwd"], alpha=0.4, label="raw")
            self.ax_acc.plot(result["t"], result["a_filt"], label="low-pass")
            self.ax_acc.axvline(d["imu_info"]["t_start"], color="red", ls="--", label="start")
            self.ax_acc.set_xlabel("t (s)")
            self.ax_acc.set_ylabel("m/s²")
            self.ax_acc.legend(fontsize=8)
        else:
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

        # User markers as vertical lines on the time plots
        for m in d["markers"]:
            for ax in (self.ax_pos, self.ax_dist, self.ax_speed, self.ax_acc, self.ax_gyro):
                ax.axvline(m["t"], color="orange", ls=":")
            self.ax_pos.annotate(m.get("label", ""), (m["t"], 1), xycoords=("data", "axes fraction"),
                                 rotation=90, va="top", fontsize=8, color="darkorange")
        self.canvas.draw_idle()
