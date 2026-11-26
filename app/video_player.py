import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import style

SPEEDS = ("0.1x", "0.25x", "0.5x", "1x")


def frame_times(n_frames, fps, tracking=None):
    # Time of every video frame on the analysis time axis (= tracking.csv "t").
    # Live recordings have an uneven frame rate, so the per-frame times from tracking.csv
    # are used when available; frames outside it are extrapolated with the nominal fps.
    i = np.arange(n_frames)
    if tracking is not None and len(tracking) >= 2 and {"frame", "t"} <= set(tracking.columns):
        frames = tracking["frame"].to_numpy(float)
        t = tracking["t"].to_numpy(float)
        out = np.interp(i, frames, t)
        out[i > frames[-1]] = t[-1] + (i[i > frames[-1]] - frames[-1]) / fps
        out[i < frames[0]] = t[0] - (frames[0] - i[i < frames[0]]) / fps
        return out
    return i / fps


class VideoPlayer(ttk.Frame):
    # Frame-accurate player for a saved video: play/pause, step, slider, jump to frame.
    # on_frame(i, t) is called after every shown frame so the plots can follow.

    def __init__(self, master, on_frame=None):
        super().__init__(master)
        self.on_frame = on_frame
        self.cap = None
        self.n = 0
        self.cur = -1
        self.times = np.array([])
        self.overlay = None     # optional function(frame index) -> (x, y) px or None
        self.playing = False
        self.photo = None

        self.view = ttk.Frame(self)
        self.view.pack(fill="both", expand=True)
        self.view.pack_propagate(False)
        self.screen = ttk.Label(self.view, text="no video", style="Video.TLabel", anchor="center")
        self.screen.pack(fill="both", expand=True)

        self.slider = ttk.Scale(self, from_=0, to=0, orient="horizontal",
                                command=lambda v: self.seek(int(float(v))) if int(float(v)) != self.cur else None)
        self.slider.pack(fill="x", pady=(6, 2))

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(4, 0))
        for text, step in (("|<", None), ("-10", -10), ("-1", -1)):
            ttk.Button(buttons, text=text, width=4,
                       command=(lambda s=step: self.seek(0) if s is None else self.step(s))).pack(side="left")
        self.play_button = ttk.Button(buttons, text="Play", width=7, command=self.toggle, style="Accent.TButton")
        self.play_button.pack(side="left", padx=4)
        for text, step in (("+1", 1), ("+10", 10), (">|", None)):
            ttk.Button(buttons, text=text, width=4,
                       command=(lambda s=step: self.seek(self.n - 1) if s is None else self.step(s))).pack(side="left")
        self.speed = tk.StringVar(value="0.25x")
        speed_box = ttk.Combobox(buttons, textvariable=self.speed, values=SPEEDS, width=6, state="readonly")
        speed_box.pack(side="left", padx=8)
        # Restart the playback clock from the current frame when the speed changes
        speed_box.bind("<<ComboboxSelected>>", lambda e: self.play() if self.playing else None)

        goto = ttk.Frame(self)
        goto.pack(fill="x", pady=(6, 0))
        ttk.Label(goto, text="Frame", style="Muted.TLabel").pack(side="left")
        self.goto_var = tk.StringVar()
        entry = ttk.Entry(goto, textvariable=self.goto_var, width=7)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda e: self.goto())
        ttk.Button(goto, text="Go", width=4, command=self.goto).pack(side="left")
        # Fixed width: a label that changes size on every frame makes Tk re-layout the whole window,
        # which was the main cause of laggy playback
        self.info = ttk.Label(goto, text="", style="Muted.TLabel", width=30, anchor="w")
        self.info.pack(side="left", padx=10)

    def load(self, path, tracking=None, overlay=None):
        self.close()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.screen.config(image="", text=f"cannot open {path}")
            return False
        self.cap = cap
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.times = frame_times(self.n, fps, tracking)
        self.overlay = overlay
        self.slider.config(to=max(self.n - 1, 0))
        self.seek(0)
        return True

    def close(self, text="no video"):
        self.pause()
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.n = 0
        self.cur = -1
        self.photo = None
        self.screen.config(image="", text=text)
        self.info.config(text="")

    def seek(self, i):
        if self.cap is None or self.n == 0:
            return
        i = int(np.clip(i, 0, self.n - 1))
        if i != self.cur + 1:   # sequential reads are fast, only seek when jumping
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        if not ok:   # frame count in the header can be larger than the real video
            self.pause()
            self.cur = -1   # position unknown, seek next time
            return
        self.cur = i
        self.show(frame)
        self.slider.set(i)
        self.info.config(text=f"{i} / {self.n - 1}    t = {self.times[i]:.3f} s")
        if self.on_frame is not None:
            self.on_frame(i, self.times[i])

    def seek_time(self, t):
        if self.n:
            self.seek(int(np.argmin(np.abs(self.times - t))))

    def step(self, n):
        self.pause()
        self.seek(self.cur + n)

    def goto(self):
        try:
            self.pause()
            self.seek(int(self.goto_var.get()))
        except ValueError:
            pass

    def show(self, frame):
        if self.overlay is not None:
            pos = self.overlay(self.cur if self.cur >= 0 else 0)
            if pos is not None:
                cv2.circle(frame, (round(pos[0]), round(pos[1])), 12, (0, 0, 255), 2)
        h, w = frame.shape[:2]
        scale = min(self.view.winfo_width() / w, self.view.winfo_height() / h)
        scale = scale if scale > 0.05 else 0.5   # widget not laid out yet
        small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        image = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if self.photo is not None and (self.photo.width(), self.photo.height()) == image.size:
            self.photo.paste(image)   # same size: reuse the Tk image instead of creating a new one
        else:
            self.photo = ImageTk.PhotoImage(image)
            self.screen.config(image=self.photo, text="")

    # --- playback: follows the recorded frame times, scaled by the chosen speed ---

    def toggle(self):
        self.pause() if self.playing else self.play()

    def play(self):
        if self.cap is None:
            return
        if self.cur >= self.n - 1:
            self.seek(0)
        self.wall0 = time.perf_counter()
        self.t0 = self.times[max(self.cur, 0)]
        if not self.playing:   # already playing (speed change): only the clock is reset
            self.playing = True
            self.play_button.config(text="Pause")
            self.tick()

    def pause(self):
        self.playing = False
        self.play_button.config(text="Play")

    def tick(self):
        if not self.playing or self.cap is None:
            return
        speed = float(self.speed.get().rstrip("x"))
        target_t = self.t0 + (time.perf_counter() - self.wall0) * speed
        target = int(np.searchsorted(self.times, target_t, side="right")) - 1
        if target >= self.n - 1:
            self.seek(self.n - 1)
            self.pause()
            return
        if target > self.cur:
            while self.cur + 1 < target:   # behind: skip frames without decoding them fully
                if not self.cap.grab():
                    break
                self.cur += 1
            self.seek(target)
        self.after(5, self.tick)
