"""Marker colour calibration: measure the HSV range of the front/back marker under the
current camera and light, and store it for the tracker (marker_colors.json).

How it works:
1. The chosen base colour gives a rough hue range (BASE_COLORS).
2. While the marker is moved in front of the camera, every frame the blob in that rough
   range that MOVES (frame difference) is taken as the marker, so still objects of a
   similar colour are not mistaken for it; pixels inside it (edges left out) and pixels
   from the rest of the image (background, including those still objects) are sampled.
3. fit_ranges() fits a tight hue / saturation / brightness range to the marker pixels and
   then tightens it until the background no longer falls into it, as long as the marker
   stays covered.
"""

import json
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "marker_colors.json"
SLOTS = ("front", "back")   # tracker order: front marker first

# Rough hue ranges in OpenCV units (0-179); lo > hi means the range wraps around red.
# display = colour used for this marker in plots.
BASE_COLORS = {
    "GREEN": {"hue": (35, 90), "display": "green"},
    "PINK": {"hue": (145, 179), "display": "deeppink"},
    "RED": {"hue": (170, 10), "display": "red"},
    "ORANGE": {"hue": (5, 22), "display": "darkorange"},
    "YELLOW": {"hue": (20, 35), "display": "goldenrod"},
    "BLUE": {"hue": (95, 130), "display": "blue"},
}
ROUGH_MIN_S, ROUGH_MIN_V = 70, 60   # rough search: ignore greyish / dark pixels

PROC_WIDTH = 640            # same processing width as the tracker
MIN_BLOB_PX = 50            # at PROC_WIDTH; smaller blobs are not the marker
MOTION_DIFF = 25            # colour change between frames (largest of B, G, R) that counts as movement
MIN_MOTION_PX = 20          # a blob needs this many moving pixels to count as the marker
EDGE_PX = 2                 # marker pixels closer than this to the edge are left out (mixed colours)
BG_MARGIN_PX = 10           # background = everything further than this from the marker
MARKER_SAMPLES = 500        # pixels sampled per frame
BG_SAMPLES = 3000
CALIBRATION_S = 6.0         # how long the marker is moved
MIN_FRAMES = 15             # frames with a detected marker needed for a result

HUE_PERCENTILES = (1, 99)
HUE_MARGIN = 4              # extra hue (OpenCV units) on both sides, shrunk if background leaks in
SV_PERCENTILE = 2
S_MARGIN, V_MARGIN = 20, 25
MIN_SV = 30                 # never accept almost grey / black pixels
MIN_COVERAGE = 0.95         # share of marker pixels that must stay inside the range
MAX_BACKGROUND = 0.0005     # share of background pixels allowed inside the range (0.05 %)


# ----------------------------------------------------------------------------- storage

def default_markers():
    # Hand-tuned ranges used when nothing is calibrated (same as before calibration existed)
    from video_analysis import GREEN, PINK
    return {"GREEN": GREEN, "PINK": PINK}


def load_calibration():
    # Content of marker_colors.json ({} if missing or unreadable)
    try:
        return json.loads(CALIBRATION_FILE.read_text())
    except (OSError, ValueError):
        return {}


def to_ranges(stored):
    # JSON [[lo, hi], ...] -> [(np.array lo, np.array hi), ...] as the tracker uses them
    return [(np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)) for lo, hi in stored]


def load_marker_colors():
    """{name: ranges} for the tracker, front marker first.

    Uses the calibrated colours from marker_colors.json; a slot that is not calibrated
    keeps its default (GREEN in front, PINK at the back).
    """
    defaults = list(default_markers().items())
    cal = load_calibration()
    markers = {}
    for slot, (default_name, default_ranges) in zip(SLOTS, defaults):
        entry = cal.get(slot)
        if entry:
            markers[entry["color"]] = to_ranges(entry["ranges"])
        else:
            markers[default_name] = default_ranges
    if len(markers) < 2:   # same colour in both slots: fall back instead of tracking one marker twice
        return dict(defaults)
    return markers


def save_slot(slot, color, ranges, stats):
    cal = load_calibration()
    cal[slot] = {
        "color": color,
        "ranges": [[lo.tolist(), hi.tolist()] for lo, hi in ranges],
        "calibrated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **{k: round(float(v), 4) for k, v in stats.items()},
    }
    CALIBRATION_FILE.write_text(json.dumps(cal, indent=2))


def reset_slot(slot):
    cal = load_calibration()
    cal.pop(slot, None)
    if cal:
        CALIBRATION_FILE.write_text(json.dumps(cal, indent=2))
    elif CALIBRATION_FILE.exists():
        CALIBRATION_FILE.unlink()


def display_color(name):
    return BASE_COLORS.get(name, {}).get("display", "gray")


def describe(ranges):
    # Short text like "H 38-71  S≥95  V≥80"
    hues = " + ".join(f"{int(lo[0])}-{int(hi[0])}" for lo, hi in ranges)
    return f"H {hues}  S≥{int(ranges[0][0][1])}  V≥{int(ranges[0][0][2])}"


# ----------------------------------------------------------------------------- algorithm

def hue_mask(h, lo, hi):
    return (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)


def rough_mask(hsv, color):
    lo, hi = BASE_COLORS[color]["hue"]
    m = hue_mask(hsv[..., 0], lo, hi) & (hsv[..., 1] >= ROUGH_MIN_S) & (hsv[..., 2] >= ROUGH_MIN_V)
    m = m.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


def largest_blob(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= MIN_BLOB_PX else None


def motion_image(frame):
    # Lightly blurred frame for the movement test (camera noise must not count as movement)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def moving_blob(mask, image, prev_image):
    """The blob of `mask` with the most moving pixels (None if nothing in the colour range moves).

    Movement = colour change between this and the previous frame (images from motion_image).
    Colour, not brightness: a coloured marker can be exactly as bright as the wall behind it.
    """
    if prev_image is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = (cv2.absdiff(image, prev_image).max(axis=2) > MOTION_DIFF).astype(np.uint8)
    best, best_score = None, MIN_MOTION_PX - 1
    for c in contours:
        if cv2.contourArea(c) < MIN_BLOB_PX:
            continue
        region = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(region, [c], -1, 1, -1)
        score = int((region & motion).sum())
        if score > best_score:
            best, best_score = c, score
    return best


def sample_frame(hsv, color, rng, image=None, prev_image=None):
    """Marker and background HSV samples of one (PROC_WIDTH) frame + the marker outline.

    With image / prev_image (motion_image of this and the previous frame) the moving blob is
    the marker; without, the largest one. Returns (marker_pixels, background_pixels, contour)
    or None.
    """
    mask = rough_mask(hsv, color)
    contour = moving_blob(mask, image, prev_image) if image is not None else largest_blob(mask)
    if contour is None:
        return None
    filled = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    inner = cv2.erode(filled, np.ones((2 * EDGE_PX + 1,) * 2, np.uint8))
    outer = cv2.dilate(filled, np.ones((2 * BG_MARGIN_PX + 1,) * 2, np.uint8))
    marker = hsv[inner > 0]
    background = hsv[outer == 0]
    if len(marker) == 0:
        return None
    marker = marker[rng.choice(len(marker), min(MARKER_SAMPLES, len(marker)), replace=False)]
    background = background[rng.choice(len(background), min(BG_SAMPLES, len(background)), replace=False)]
    return marker, background, contour


def inside(pixels, center, lo_off, hi_off, s_min, v_min):
    # Which pixels fall into the range (hue as offsets around center, handles the wrap at red)
    d = (pixels[:, 0].astype(int) - center + 90) % 180 - 90
    return (d >= lo_off) & (d <= hi_off) & (pixels[:, 1] >= s_min) & (pixels[:, 2] >= v_min)


def fit_ranges(marker, background):
    """Tight HSV range for the marker pixels, tightened against the background.

    Returns (ranges, stats) with ranges as the tracker uses them and stats
    {"coverage": share of marker pixels inside, "background": share of background inside}.
    """
    h = marker[:, 0].astype(float) * np.pi / 90.0   # OpenCV hue 0-179 -> angle 0-2π
    center = int(round(np.degrees(np.arctan2(np.sin(h).mean(), np.cos(h).mean())) / 2)) % 180
    d = (marker[:, 0].astype(int) - center + 90) % 180 - 90
    lo_core, hi_core = np.percentile(d, HUE_PERCENTILES)
    s_min = int(max(MIN_SV, np.percentile(marker[:, 1], SV_PERCENTILE) - S_MARGIN))
    v_min = int(max(MIN_SV, np.percentile(marker[:, 2], SV_PERCENTILE) - V_MARGIN))
    margin = HUE_MARGIN

    def evaluate(margin, s_min, v_min):
        lo, hi = int(np.floor(lo_core)) - margin, int(np.ceil(hi_core)) + margin
        cov = inside(marker, center, lo, hi, s_min, v_min).mean()
        bg = inside(background, center, lo, hi, s_min, v_min).mean() if len(background) else 0.0
        return cov, bg

    cov, bg = evaluate(margin, s_min, v_min)
    # Tighten while background leaks in: hue margin first, then saturation, then brightness
    while bg > MAX_BACKGROUND:
        options = []
        if margin > 0:
            options.append((margin - 1, s_min, v_min))
        if s_min < 250:
            options.append((margin, s_min + 5, v_min))
        if v_min < 250:
            options.append((margin, s_min, v_min + 5))
        scored = [(evaluate(*o), o) for o in options]
        scored = [(c, b, o) for (c, b), o in scored if c >= MIN_COVERAGE]
        if not scored:
            break   # cannot get cleaner without losing the marker
        cov, bg, (margin, s_min, v_min) = min(scored, key=lambda x: (x[1], -x[0]))

    lo = (center + int(np.floor(lo_core)) - margin) % 180
    hi = (center + int(np.ceil(hi_core)) + margin) % 180
    hue_parts = [(lo, hi)] if lo <= hi else [(lo, 179), (0, hi)]
    ranges = [(np.array([a, s_min, v_min], np.uint8), np.array([b, 255, 255], np.uint8))
              for a, b in hue_parts]
    return ranges, {"coverage": cov, "background": bg}


# ----------------------------------------------------------------------------- GUI

class CalibrationTab(ttk.Frame):
    """Pick front/back and a base colour, move the marker, get a calibrated colour range.

    The camera is only opened while this tab is shown (activate / deactivate are called
    by the main window, which hands its camera over).
    """

    def __init__(self, master, on_saved=None):
        super().__init__(master)
        self.on_saved = on_saved   # called after a calibration was saved or reset
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None          # newest frame (PROC_WIDTH, BGR) from the camera thread
        self.collecting = None     # list of samples while calibrating, else None
        self.calib_end = 0.0
        self.rng = np.random.default_rng()
        self.preview_ranges = None   # range to show in the mask after a calibration
        self.prev_image = None       # previous frame (motion_image), to find the moving marker

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, text="Marker:").pack(side="left")
        self.slot = tk.StringVar(value="front")
        for slot in SLOTS:
            ttk.Radiobutton(controls, text=slot.capitalize(), value=slot, variable=self.slot,
                            command=self.on_choice).pack(side="left", padx=4)
        ttk.Label(controls, text="Base colour:").pack(side="left", padx=(15, 4))
        self.color = ttk.Combobox(controls, values=list(BASE_COLORS), width=10, state="readonly")
        self.color.set("GREEN")
        self.color.pack(side="left")
        self.color.bind("<<ComboboxSelected>>", lambda e: self.on_choice())
        self.start_button = ttk.Button(controls, text="Start calibration", command=self.start_calibration)
        self.start_button.pack(side="left", padx=10)
        ttk.Button(controls, text="Reset to default", command=self.reset).pack(side="left")
        self.status = ttk.Label(controls, text="")
        self.status.pack(side="left", padx=15)

        self.settings = ttk.Label(self, text="", font=("Courier New", 12), justify="left")
        self.settings.pack(anchor="w", padx=10)
        ttk.Label(self, text="Left: camera, outline = detected marker.   Right: pixels inside the "
                             "colour range (white). After calibrating, only the marker should be white.",
                  foreground="gray").pack(anchor="w", padx=10, pady=(4, 0))
        views = ttk.Frame(self)
        views.pack(fill="both", expand=True, padx=10, pady=10)
        self.video_view = ttk.Label(views)
        self.video_view.pack(side="left", padx=(0, 10))
        self.mask_view = ttk.Label(views)
        self.mask_view.pack(side="left")

        self.on_choice()
        self.poll()

    # ------------------------------------------------------------ camera

    def activate(self, camera_index):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.cap = None
            self.status.config(text=f"Cannot open camera {camera_index}")
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.grab_loop, daemon=True)
        self.thread.start()
        self.status.config(text="Camera on. Pick marker + base colour, then Start calibration.")

    def deactivate(self):
        self.collecting = None
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_button.config(state="normal")

    def grab_loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            scale = PROC_WIDTH / frame.shape[1]
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.frame = small

    # ------------------------------------------------------------ calibration

    def on_choice(self):
        self.preview_ranges = None   # show the rough range for the newly chosen colour
        self.show_settings()

    def start_calibration(self):
        if self.cap is None:
            self.status.config(text="No camera")
            return
        self.collecting = []
        self.calib_end = time.perf_counter() + CALIBRATION_S
        self.start_button.config(state="disabled")

    def finish_calibration(self):
        samples, self.collecting = self.collecting, None
        self.start_button.config(state="normal")
        if len(samples) < MIN_FRAMES:
            self.status.config(text=f"Moving marker found in only {len(samples)} frames (need {MIN_FRAMES}). "
                                    "Is the base colour right? Keep the marker moving while calibrating.")
            return
        marker = np.vstack([s[0] for s in samples])
        background = np.vstack([s[1] for s in samples])
        ranges, stats = fit_ranges(marker, background)
        slot, color = self.slot.get(), self.color.get()
        other = load_calibration().get("back" if slot == "front" else "front", {}).get("color")
        if other == color:
            self.status.config(text=f"The other marker is already {color}: pick a different base colour.")
            return
        save_slot(slot, color, ranges, {**stats, "frames": len(samples)})
        self.preview_ranges = ranges
        self.status.config(text=f"Saved {slot} = {color}: covers {100 * stats['coverage']:.1f} % of the marker, "
                                f"{100 * stats['background']:.3f} % of the background ({len(samples)} frames)")
        self.show_settings()
        if self.on_saved is not None:
            self.on_saved()

    def reset(self):
        reset_slot(self.slot.get())
        self.preview_ranges = None
        self.status.config(text=f"{self.slot.get().capitalize()} marker back to the default colour")
        self.show_settings()
        if self.on_saved is not None:
            self.on_saved()

    def show_settings(self):
        cal = load_calibration()
        names = list(load_marker_colors().items())
        lines = []
        for slot, (name, ranges) in zip(SLOTS, names):
            entry = cal.get(slot)
            when = f"calibrated {entry['calibrated']}" if entry else "default"
            lines.append(f"{slot:5s}  {name:7s} {describe(ranges):28s} ({when})")
        self.settings.config(text="\n".join(lines))

    # ------------------------------------------------------------ live view

    def poll(self):
        with self.lock:
            frame = self.frame
        if frame is not None and self.cap is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            image = motion_image(frame)
            color = self.color.get()
            calib_contour = None
            if self.collecting is not None:
                left = self.calib_end - time.perf_counter()
                sample = sample_frame(hsv, color, self.rng, image, self.prev_image)
                if sample is not None:
                    self.collecting.append(sample[:2])
                    calib_contour = sample[2]
                self.status.config(text=f"Calibrating: keep moving the marker ({max(left, 0):.1f} s), "
                                        f"found in {len(self.collecting)} frames")
                if left <= 0:
                    self.finish_calibration()
            self.prev_image = image
            # Mask: calibrated range if there is one, else the rough base-colour range
            if self.preview_ranges is not None:
                mask = np.zeros(hsv.shape[:2], np.uint8)
                for lo, hi in self.preview_ranges:
                    mask |= cv2.inRange(hsv, lo, hi)
            else:
                mask = rough_mask(hsv, color)
            contour = calib_contour if calib_contour is not None else largest_blob(mask)
            shown = frame.copy()
            if contour is not None:
                cv2.drawContours(shown, [contour], -1, (255, 255, 0), 2)
            self.show_image(self.video_view, cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
            self.show_image(self.mask_view, mask)
        self.after(40, self.poll)

    def show_image(self, label, array):
        photo = ImageTk.PhotoImage(Image.fromarray(array))
        label.config(image=photo)
        label.image = photo   # keep a reference
