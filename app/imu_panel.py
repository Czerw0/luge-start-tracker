"""Live IMU panel for the GUI: find a Movella DOT, connect, show and plot its data live."""

import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_analysis import DotStreamer, scan_dots

POLL_MS = 50          # how often new samples are fetched from the streamer
PLOT_EVERY = 4        # redraw the plot every N polls
PLOT_SECONDS = 10     # time window shown in the plot
PLOT_HEIGHT = 200     # px
AXIS_COLORS = {"ax": "tab:red", "ay": "tab:green", "az": "tab:blue"}


class ImuPanel(ttk.Frame):
    """Scan / connect controls, current values and a live plot of the free acceleration.

    All Bluetooth work runs on background threads (DotStreamer, scan thread); this
    widget only reads their results in poll(), because tkinter must stay on the main thread.
    """

    def __init__(self, master):
        super().__init__(master)
        self.streamer = None
        self.sensors = []        # result of the last scan
        self.scan_result = None  # set by the scan thread
        self.scanning = False
        self.connect_started = None
        self.polls = 0
        # Samples of the last PLOT_SECONDS s: (t_host, ax, ay, az); 60 Hz -> 600 per 10 s
        self.buffer = deque(maxlen=PLOT_SECONDS * 120)

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text="IMU:").pack(side="left")
        self.scan_button = ttk.Button(controls, text="Scan", command=self.scan)
        self.scan_button.pack(side="left", padx=5)
        self.sensor_choice = ttk.Combobox(controls, width=28, state="readonly")
        self.sensor_choice.pack(side="left", padx=5)
        self.connect_button = ttk.Button(controls, text="Connect", command=self.toggle_connect,
                                         state="disabled")
        self.connect_button.pack(side="left", padx=5)
        self.status = ttk.Label(controls, text="Not connected")
        self.status.pack(side="left", padx=15)
        self.values = ttk.Label(controls, text="", font=("Courier New", 12))
        self.values.pack(side="left", padx=15)

        self.fig = Figure(figsize=(12, 2), dpi=100, layout="constrained")
        self.ax = self.fig.add_subplot()
        self.ax.set_ylabel("free acc (m/s²)")
        self.ax.set_xlabel("t (s)")
        self.ax.grid(alpha=0.3)
        self.lines = {c: self.ax.plot([], [], color=col, lw=1, label=c)[0]
                      for c, col in AXIS_COLORS.items()}
        self.ax.legend(loc="upper left", fontsize=8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().config(height=PLOT_HEIGHT)
        self.canvas.get_tk_widget().pack(fill="x", pady=(5, 0))

        # Window closed: release the sensor so it can be used again right away
        self.bind("<Destroy>", self.on_destroy)
        self.poll()
        self.scan()   # look for sensors right away

    # ----------------------------------------------------------------- scanning

    def scan(self):
        if self.scanning or self.streamer is not None:
            return
        self.scanning = True
        self.scan_button.config(state="disabled")
        self.connect_button.config(state="disabled")
        self.status.config(text="Scanning for Movella DOTs...")

        def worker():
            try:
                self.scan_result = scan_dots()
            except Exception as e:
                self.scan_result = e

        threading.Thread(target=worker, daemon=True).start()

    def finish_scan(self):
        result, self.scan_result = self.scan_result, None
        self.scanning = False
        self.scan_button.config(state="normal")
        if isinstance(result, Exception):
            self.status.config(text=f"Scan failed: {result}")
            return
        previous = self.selected_sensor()
        self.sensors = result
        self.sensor_choice["values"] = [f"{s['mac']}  ({s['rssi']} dBm)" for s in result]
        if not result:
            self.sensor_choice.set("")
            self.status.config(text="No DOT found (sensor on? not connected to the phone app?)")
            return
        # Keep the previously chosen sensor if it is still there, otherwise the closest one
        macs = [s["mac"] for s in result]
        index = macs.index(previous["mac"]) if previous and previous["mac"] in macs else 0
        self.sensor_choice.current(index)
        self.connect_button.config(state="normal")
        self.status.config(text=f"{len(result)} DOT(s) found, strongest signal first")

    def selected_sensor(self):
        i = self.sensor_choice.current()
        return self.sensors[i] if 0 <= i < len(self.sensors) else None

    # ----------------------------------------------------------------- connection

    def toggle_connect(self):
        if self.streamer is None:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        sensor = self.selected_sensor()
        if sensor is None:
            return
        self.buffer.clear()
        self.streamer = DotStreamer(sensor["address"])
        self.streamer.start()
        self.connect_started = time.perf_counter()
        self.connect_button.config(text="Disconnect")
        self.scan_button.config(state="disabled")
        self.sensor_choice.config(state="disabled")
        self.status.config(text=f"Connecting to {sensor['mac']}...")

    def disconnect(self, message="Not connected"):
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None
        self.connect_button.config(text="Connect")
        self.scan_button.config(state="normal")
        self.sensor_choice.config(state="readonly")
        self.status.config(text=message)
        self.values.config(text="")

    def on_destroy(self, event):
        if event.widget is self and self.streamer is not None:
            self.streamer.stop()
            self.streamer = None

    # ----------------------------------------------------------------- live update

    def poll(self):
        self.polls += 1
        if self.scan_result is not None:
            self.finish_scan()

        s = self.streamer
        if s is not None:
            for t_host, _ts, ax, ay, az in s.samples_so_far():
                self.buffer.append((t_host, ax, ay, az))
            if not s.is_running():
                # Thread ended without us stopping it: connection failed or dropped
                self.disconnect(f"Connection lost: {s.error}" if s.error else "Connection lost")
            elif s.connected.is_set():
                self.show_values()
            elif time.perf_counter() - self.connect_started > 20:
                self.disconnect("Could not connect within 20 s")
        if self.polls % PLOT_EVERY == 0 and self.streamer is not None:
            self.update_plot()
        self.after(POLL_MS, self.poll)

    def show_values(self):
        if not self.buffer:
            self.status.config(text="Connected, waiting for data...")
            return
        _, ax, ay, az = self.buffer[-1]
        recent = [b[0] for b in self.buffer if b[0] > self.buffer[-1][0] - 1]
        rate = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0
        self.status.config(text=f"Connected to {self.selected_sensor()['mac']}, {rate:.0f} Hz")
        self.values.config(text=f"ax {ax:+6.2f}  ay {ay:+6.2f}  az {az:+6.2f}  "
                                f"|a| {np.sqrt(ax*ax + ay*ay + az*az):5.2f} m/s²")

    def update_plot(self):
        if not self.buffer:
            return
        data = np.array(self.buffer)
        t = data[:, 0] - data[-1, 0]     # seconds relative to now (0 = latest sample)
        keep = t >= -PLOT_SECONDS
        for i, c in enumerate(("ax", "ay", "az"), start=1):
            self.lines[c].set_data(t[keep], data[keep, i])
        self.ax.set_xlim(-PLOT_SECONDS, 0)
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        self.canvas.draw_idle()
