import base64
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from video_analysis import MarkerTracker

ANALYSES_DIR = Path(__file__).resolve().parent.parent / "analyses"
RESULTS_WIDTH = 420   # px reserved for the results panel; the video gets the rest
PLOT_HEIGHT = 260     # px height of the plots at the bottom
PLOT_SECONDS = 10     # time window shown in the plots
PLOT_EVERY = 6        # redraw plots every N polls (poll = 40 ms)
PLOT_COLORS = {"yellow": "goldenrod", "red": "red"}


def format_results(row, names):
    # Current values as multi-line text for the results panel
    lines = [f"frame {row['frame']}    t = {row['t']:.3f} s", ""]
    for name in names:
        lines += [name,
                  f"  raw     {row[name + '_x']:8.2f} {row[name + '_y']:8.2f}",
                  f"  kalman  {row[name + '_x_filt']:8.2f} {row[name + '_y_filt']:8.2f}",
                  f"  size    {row[name + '_size']:8.0f}  ({row[name + '_w']:.0f} x {row[name + '_h']:.0f})",
                  ""]
    lines += [f"distance  {row['distance']:.0f}   (kalman {row['distance_filt']:.0f})",
              f"middle    {row['mid_x']:.2f}  {row['mid_y']:.2f}"]
    return "\n".join(lines)


class Gui:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Luge Start Tracker")
        w = min(1700, int(self.window.winfo_screenwidth() * 0.92))
        h = min(1000, int(self.window.winfo_screenheight() * 0.9))
        self.window.geometry(f"{w}x{h}+20+20")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        # Live tracking runs in a worker thread; it only writes these fields,
        # the GUI reads them in poll() (tkinter must only be used from the main thread)
        self.stop_event = threading.Event()
        self.thread = None
        self.tracker = None
        self.folder = None
        self.latest_frame = None
        self.latest_row = None
        self.rows = []
        self.error = None
        self.polls = 0

        self.tabs = ttk.Notebook(self.window)
        self.tabs.pack(fill="both", expand=True)

        # Measurement tab: synchronised IMU + video signals (must have)
        self.measurement_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.measurement_tab, text="Measurement")
        self.build_measurement_tab()

        # Analysis tab: load saved analyses, video player, pose estimation (optional)
        self.analysis_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.analysis_tab, text="Analysis")
        ttk.Label(self.analysis_tab, text="Analysis: saved measurements").pack(pady=20)

    def build_measurement_tab(self):
        tab = self.measurement_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)   # video + results get all the free space
        # row 2 (plots) keeps a fixed height

        # Top: controls
        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        ttk.Label(controls, text="Camera:").pack(side="left")
        self.camera_var = tk.StringVar(value="0")
        ttk.Spinbox(controls, from_=0, to=9, width=3, textvariable=self.camera_var).pack(side="left", padx=5)
        self.start_button = ttk.Button(controls, text="Start analysis", command=self.start)
        self.start_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(controls, text="Stop analysis", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        self.status = ttk.Label(controls, text="Ready")
        self.status.pack(side="left", padx=15)

        # Middle: video on the left, current results on the right
        self.middle = ttk.Frame(tab)
        self.middle.grid(row=1, column=0, sticky="nsew", padx=10)
        self.middle.pack_propagate(False)   # size comes from the grid, not from the image
        self.results = tk.Label(self.middle, text="", font=("Courier New", 12), width=38,
                                justify="left", anchor="nw")
        self.results.pack(side="right", fill="y", padx=(20, 0))
        self.preview = ttk.Label(self.middle)
        self.preview.pack(side="left", anchor="nw")

        # Bottom: plots of the last PLOT_SECONDS seconds
        self.fig = Figure(figsize=(12, 3), dpi=100, layout="constrained")
        self.ax_x, self.ax_y, self.ax_d = self.fig.subplots(1, 3)
        self.ax_x.set_title("x (px)")
        self.ax_y.set_title("y (px)")
        self.ax_y.invert_yaxis()   # image y grows downwards
        self.ax_d.set_title("distance (px)")
        for ax in (self.ax_x, self.ax_y, self.ax_d):
            ax.set_xlabel("t (s)")
            ax.grid(alpha=0.3)
        self.lines = {}   # (plot key, column) -> matplotlib line
        for name in ("yellow", "red"):
            c = PLOT_COLORS.get(name, name)
            for key, ax in (("x", self.ax_x), ("y", self.ax_y)):
                self.lines[name + "_" + key + "_filt"] = ax.plot([], [], color=c, label=name + " (Kalman)")[0]
                self.lines[name + "_" + key] = ax.plot([], [], ".", color=c, alpha=0.4, markersize=3)[0]
        self.lines["distance_filt"] = self.ax_d.plot([], [], color="green", label="Kalman")[0]
        self.lines["distance"] = self.ax_d.plot([], [], ".", color="green", alpha=0.4, markersize=3)[0]
        self.ax_x.legend(loc="upper left", fontsize=8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.get_tk_widget().config(height=PLOT_HEIGHT)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="ew", padx=10, pady=10)

    def start(self):
        try:
            camera = int(self.camera_var.get())
        except ValueError:
            messagebox.showerror("Luge Start Tracker", "Camera must be a number.")
            return
        self.folder = ANALYSES_DIR / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_measurement")
        self.tracker = MarkerTracker()
        self.stop_event.clear()
        self.latest_frame = None
        self.latest_row = None
        self.rows = []
        self.error = None

        self.thread = threading.Thread(target=self.worker, args=(camera,), daemon=True)
        self.thread.start()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.poll()

    def stop(self):
        self.stop_event.set()
        self.stop_button.config(state="disabled")
        self.status.config(text="Stopping...")

    def worker(self, camera):
        # Records video + tracking.csv into the analysis folder
        try:
            self.tracker.track_camera(camera, save_path=self.folder / "video.mp4",
                                      csv_path=self.folder / "tracking.csv",
                                      on_frame=self.on_frame, verbose=False,
                                      stop_event=self.stop_event, preview=False)
        except Exception as e:
            self.error = str(e)

    def on_frame(self, frame, detections, row):
        self.latest_frame = self.tracker.draw(frame, detections)
        self.latest_row = row
        self.rows.append(row)

    def poll(self):
        # Refresh preview, results and plots while the worker runs
        self.polls += 1
        if self.latest_frame is not None:
            self.show(self.latest_frame)
        if self.latest_row is not None:
            self.results.config(text=format_results(self.latest_row, self.tracker.markers))
        if self.polls % PLOT_EVERY == 0:
            self.update_plots()

        if self.thread.is_alive():
            if not self.stop_event.is_set():
                self.status.config(text=f"Recording... {len(self.rows)} frames")
            self.window.after(40, self.poll)
            return

        self.update_plots()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        if self.error:
            self.status.config(text="Error")
            messagebox.showerror("Luge Start Tracker", self.error)
        else:
            self.status.config(text=f"Saved {len(self.rows)} frames to {self.folder}")

    def show(self, frame):
        # Scale to the largest size that fits next to the results panel
        h, w = frame.shape[:2]
        avail_w = self.middle.winfo_width() - RESULTS_WIDTH
        avail_h = self.middle.winfo_height()
        scale = max(min(avail_w / w, avail_h / h), 0.2)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)))
        ok, png = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self.photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        self.preview.config(image=self.photo)

    def update_plots(self):
        rows = list(self.rows)
        if not rows:
            return
        t = np.array([r["t"] for r in rows])
        keep = t >= t[-1] - PLOT_SECONDS
        t = t[keep]
        for key, line in self.lines.items():
            line.set_data(t, np.array([r[key] for r in rows])[keep])
        for ax in (self.ax_x, self.ax_y, self.ax_d):
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=3)
        self.window.destroy()

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    Gui().run()
