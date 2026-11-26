import json
import threading
import time # for running the camera in a separate thread
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

import style
from dashboard import Dashboard, DevelopmentTab
from data_analysis import samples_to_dataframe, save_imu
from imu_panel import ImuPanel
from video_analysis import MarkerTracker

ANALYSES_DIR = Path(__file__).resolve().parent.parent / "analyses"
RESULTS_WIDTH = 420   # px (at 100 % scaling) reserved for the results panel; the video gets the rest
PLOT_HEIGHT = 260     # px (at 100 % scaling) height of the plots at the bottom
PLOT_SECONDS = 10     # time window shown in the plots
PLOT_INTERVAL = 0.6   # s between plot redraws; a matplotlib redraw blocks the video
PREVIEW_ROWS = 600     # rows kept while only previewing
PLOT_COLORS = style.MARKER_COLORS


def format_results(row, names):
    # Current values as multi-line text for the results panel
    lines = [f"frame {row['frame']}    t = {row['t']:.3f} s", ""]
    for name in names:
        lines += [name,
                  f"  raw     {row[name + '_x']:8.2f} {row[name + '_y']:8.2f}",
                  f"  kalman  {row[name + '_x_filt']:8.2f} {row[name + '_y_filt']:8.2f}",
                  f"  size    {row[name + '_size']/10000:.2f}  ({row[name + '_w']:.0f} x {row[name + '_h']:.0f})",
                  ""]
    lines += [f"distance  {row['distance']:.0f}   (kalman {row['distance_filt']:.0f})",
              f"middle    {row['mid_x']:.2f}  {row['mid_y']:.2f}"]
    return "\n".join(lines)


class Gui:
    def __init__(self):
        style.enable_sharp_rendering()   # before the window exists, else Windows blurs it
        self.window = tk.Tk()
        style.apply_theme(self.window)   # before any widget or plot is created
        self.window.title("Luge Start Tracker")
        w = min(style.px(1700), int(self.window.winfo_screenwidth() * 0.92))
        h = min(style.px(1000), int(self.window.winfo_screenheight() * 0.9))
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
        self.shown_frame = None
        self.last_plot = 0.0
        self.mode = "idle"      # idle | preview | recording
        self.frozen = False

        self.tabs = ttk.Notebook(self.window)
        self.tabs.pack(fill="both", expand=True)

        # Measurement tab: synchronised IMU + video signals (must have)
        self.measurement_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.measurement_tab, text="Measurement")
        self.build_measurement_tab()

        # Analysis tab: load saved analyses, video player, pose estimation (optional)
        self.analysis_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.analysis_tab, text="Analysis")
        self.dashboard = Dashboard(self.analysis_tab, ANALYSES_DIR)
        self.dashboard.pack(fill="both", expand=True)

        # Development tab: diagnostics (x/y range, path length, IMU drift) for the
        # recording that is open in the Analysis tab
        self.development_tab = DevelopmentTab(self.tabs)
        self.tabs.add(self.development_tab, text="Development")
        self.dashboard.on_load = self.development_tab.show
        if self.dashboard.data is not None:   # the newest recording was loaded on start
            self.development_tab.show(self.dashboard.data)
        # Pick up a freshly saved recording when the user switches to the tab
        self.tabs.bind("<<NotebookTabChanged>>",
                       lambda e: self.dashboard.refresh() if self.tabs.select() == str(self.analysis_tab) else None)

        self.poll()
        self.window.after(300, self.start_preview)   # show the camera right away

    def build_measurement_tab(self):
        tab = self.measurement_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)   # video + results get all the free space
        # row 2 (plots) keeps a fixed height

        # Top: controls
        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        ttk.Label(controls, text="Camera", style="Title.TLabel").pack(side="left")
        self.camera_var = tk.StringVar(value="0")
        ttk.Spinbox(controls, from_=0, to=9, width=3, textvariable=self.camera_var,
                    command=self.on_camera_change).pack(side="left", padx=(8, 12))
        self.start_button = ttk.Button(controls, text="Start analysis", command=self.start,
                                       style="Accent.TButton")
        self.start_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(controls, text="Stop analysis", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        self.status = ttk.Label(controls, text="Ready", style="Muted.TLabel")
        self.status.pack(side="left", padx=15)

        # Middle: video on the left, current results on the right
        self.middle = ttk.Frame(tab)
        self.middle.grid(row=1, column=0, sticky="nsew", padx=12)
        self.middle.pack_propagate(False)   # size comes from the grid, not from the image
        self.results = tk.Label(self.middle, text="No tracking values yet", font=style.MONO, width=38,
                                justify="left", anchor="nw", bg=style.SURFACE, fg=style.TEXT,
                                padx=14, pady=12, highlightthickness=1,
                                highlightbackground=style.BORDER)
        self.results.pack(side="right", fill="y", padx=(12, 0))
        self.preview = ttk.Label(self.middle, text="Waiting for the camera...", style="Video.TLabel",
                                 anchor="center", justify="center")
        self.preview.pack(side="left", fill="both", expand=True)

        # Bottom: plots of the last PLOT_SECONDS seconds
        self.fig = Figure(figsize=(12, PLOT_HEIGHT / 100), dpi=100)   # fixed layout: constrained layout is slow to redraw
        self.ax_x, self.ax_y, self.ax_d = self.fig.subplots(1, 3)
        self.fig.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.86, wspace=0.18)
        self.ax_x.set_title("Marker x (px)")
        self.ax_y.set_title("Marker y (px)")
        self.ax_y.invert_yaxis()   # image y grows downwards
        self.ax_d.set_title("Distance between markers (px)")
        # Grey note shown while there is nothing to plot yet
        self.empty_notes = {ax: ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                                        transform=ax.transAxes, color=style.MUTED)
                            for ax in (self.ax_x, self.ax_y, self.ax_d)}
        for ax in (self.ax_x, self.ax_y, self.ax_d):
            ax.set_xlabel("t (s)")
        self.lines = {}   # (plot key, column) -> matplotlib line
        for name in ("GREEN", "PINK"):
            c = PLOT_COLORS.get(name, name)
            for key, ax in (("x", self.ax_x), ("y", self.ax_y)):
                self.lines[name + "_" + key + "_filt"] = ax.plot([], [], color=c, label=name + " (Kalman)")[0]
                self.lines[name + "_" + key] = ax.plot([], [], ".", color=c, alpha=0.4, markersize=3, rasterized=True)[0]
        self.lines["distance_filt"] = self.ax_d.plot([], [], color=style.POS_X, label="Kalman")[0]
        self.lines["distance"] = self.ax_d.plot([], [], ".", color=style.POS_X, alpha=0.4, markersize=3)[0]
        self.ax_x.legend(loc="upper left", ncol=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        # Unscaled on purpose: matplotlib scales its canvas to the screen itself when it is shown
        self.canvas.get_tk_widget().config(height=PLOT_HEIGHT)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="ew", padx=12, pady=12)

        # Bottom: live Movella DOT IMU (scan, connect, values, plot)
        self.imu_panel = ImuPanel(tab)
        self.imu_panel.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))

    def run_thread(self, target, *args):
        self.stop_event.clear()
        self.error = None
        self.thread = threading.Thread(target=target, args=args, daemon=True)
        self.thread.start()

    def stop_thread(self):
        # Stop whatever the worker is doing and wait until the camera is released
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)

    def camera_index(self):
        try:
            return int(self.camera_var.get())
        except ValueError:
            return None

    def start_preview(self, keep_data=False):
        # Live camera view + tracking values, nothing is saved
        camera = self.camera_index()
        if camera is None:
            return
        if not keep_data:
            self.reset_data()
        self.frozen = keep_data   # keep showing the finished analysis in the plots
        self.mode = "preview"
        self.tracker = MarkerTracker()
        self.run_thread(self.preview_worker, camera)
        self.status.config(text="Preview (not recording)")

    def on_camera_change(self):
        # Spinbox changed: only switch cameras while previewing
        if self.mode == "preview" or self.mode == "idle":
            self.stop_thread()
            self.start_preview()

    def reset_data(self):
        # Back to the starting point: no values, empty plots
        self.latest_frame = None
        self.latest_row = None
        self.rows = []
        self.results.config(text="")
        self.update_plots()

    def start(self):
        camera = self.camera_index()
        if camera is None:
            messagebox.showerror("Luge Start Tracker", "Camera must be a number.")
            return
        if not self.imu_panel.is_connected():   # every analysis needs video + IMU
            messagebox.showerror("Luge Start Tracker",
                                 "The IMU is not connected. Scan for the Movella DOT, connect it, "
                                 "then start the analysis.")
            return
        self.stop_thread()   # release the camera used by the preview
        self.reset_data()
        self.folder = ANALYSES_DIR / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_measurement")
        self.tracker = MarkerTracker()
        self.frozen = False
        self.mode = "recording"
        self.imu_panel.start_recording()   # IMU samples are saved with the video when stopping
        self.run_thread(self.worker, camera)
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

    def stop(self):
        self.stop_event.set()
        self.stop_button.config(state="disabled")
        self.status.config(text="Stopping...")

    def preview_worker(self, camera):
        try:
            self.tracker.track_camera(camera, on_frame=self.on_preview_frame,
                                      stop_event=self.stop_event)
        except Exception as e:
            self.error = str(e)

    def worker(self, camera):
        # Records video + tracking.csv into the analysis folder
        try:
            self.tracker.track_camera(camera, save_path=self.folder / "video.mp4",
                                      csv_path=self.folder / "tracking.csv",
                                      on_frame=self.on_frame,
                                      stop_event=self.stop_event)
        except Exception as e:
            self.error = str(e)

    def on_frame(self, frame, detections, row):
        self.latest_frame = self.tracker.draw(frame, detections)
        self.latest_row = row
        self.rows.append(row)

    def on_preview_frame(self, frame, detections, row):
        self.latest_frame = self.tracker.draw(frame, detections)
        self.latest_row = row
        if not self.frozen:
            self.rows.append(row)
            del self.rows[:-PREVIEW_ROWS]   # rolling buffer, preview data is not kept

    def poll(self):
        # Refresh preview, results and plots (runs for the whole life of the window)
        self.polls += 1
        frame = self.latest_frame
        if frame is not None and frame is not self.shown_frame:   # only redraw new frames
            self.shown_frame = frame
            self.show(frame)
        if self.latest_row is not None:
            self.results.config(text=format_results(self.latest_row, self.tracker.markers))
        if time.perf_counter() - self.last_plot >= PLOT_INTERVAL:
            self.last_plot = time.perf_counter()
            self.update_plots()

        alive = self.thread is not None and self.thread.is_alive()
        if self.mode == "recording":
            if alive:
                if not self.stop_event.is_set():
                    self.status.config(text=f"Recording... {len(self.rows)} frames")
            else:
                self.update_plots()
                self.start_button.config(state="normal")
                self.stop_button.config(state="disabled")
                imu_samples = self.imu_panel.stop_recording()
                if self.error:
                    self.status.config(text="Error")
                    messagebox.showerror("Luge Start Tracker", self.error)
                    self.start_preview()
                else:
                    saved = f"Saved {len(self.rows)} frames"
                    saved += self.save_imu_recording(imu_samples) + f" to {self.folder}"
                    self.start_preview(keep_data=True)
                    self.status.config(text=saved)
        elif self.mode == "preview" and not alive and self.error:
            self.status.config(text=f"Camera unavailable: {self.error}")
            self.preview.config(text=f"No camera image\n\n{self.error}")
            self.mode = "idle"
        self.window.after(10, self.poll)

    def save_imu_recording(self, samples):
        # imu_raw.csv + meta.json next to the video; returns text for the status line
        if not samples:
            return " (no IMU data: sensor not connected)"
        imu = samples_to_dataframe(samples)
        save_imu(imu, self.folder / "imu_raw.csv")
        video_t0 = getattr(self.tracker, "t0", None)
        imu_t0 = float(imu["t_host"].iloc[0])
        meta = {
            "imu_sensor": self.imu_panel.connected_mac(),
            "imu_samples": len(imu),
            # Both clocks are time.perf_counter() on this computer. IMU time 0 (first sample)
            # is at video time imu_offset_s; Bluetooth latency (~tens of ms) is not removed.
            "video_t0_host": video_t0,
            "imu_t0_host": imu_t0,
            "imu_offset_s": imu_t0 - video_t0 if video_t0 is not None else None,
        }
        (self.folder / "meta.json").write_text(json.dumps(meta, indent=2))
        return f" + {len(imu)} IMU samples"

    def show(self, frame):
        # Scale to the largest size that fits next to the results panel
        h, w = frame.shape[:2]
        avail_w = self.middle.winfo_width() - style.px(RESULTS_WIDTH)
        avail_h = self.middle.winfo_height()
        scale = max(min(avail_w / w, avail_h / h), 0.2)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.preview.config(image=self.photo)

    def update_plots(self):
        rows = list(self.rows)
        for ax, note in self.empty_notes.items():
            note.set_visible(not rows)
            ax.tick_params(labelleft=bool(rows))   # no meaningless y values on an empty plot
        if not rows:
            for line in self.lines.values():
                line.set_data([], [])
            for ax in self.empty_notes:
                ax.set_xlim(0, PLOT_SECONDS)
            self.canvas.draw_idle()
            return
        rows = rows[-PLOT_SECONDS * 60:]   # enough for the window at up to 60 fps
        t = np.array([r["t"] for r in rows])
        keep = t >= t[-1] - PLOT_SECONDS
        t = t[keep]
        rows = [r for r, k in zip(rows, keep) if k]
        for key, line in self.lines.items():
            line.set_data(t, [r[key] for r in rows])
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
