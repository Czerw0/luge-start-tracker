"""Marker colour calibration: measure the HSV range of the front/back marker under the
current camera and light, and store it for the tracker (marker_colors.json).

How it works:
1. The chosen base colour gives a rough hue range (BASE_COLORS).
2. While the marker is moved in front of the camera, every frame the blob in that rough
   range that is SQUARE and MOVES (frame difference) is taken as the marker, so neither
   the hand holding it (skin) nor still objects of a similar colour are mistaken for it;
   pixels inside it (edges left out) and pixels from the rest of the image (background,
   including hand and still objects) are sampled.
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
    "GREEN": {"hue": (35, 90), "display": "green", "bgr": (0, 160, 0)},
    "PINK": {"hue": (145, 179), "display": "deeppink", "bgr": (180, 105, 255)},
    "RED": {"hue": (170, 10), "display": "red", "bgr": (0, 0, 255)},
    "ORANGE": {"hue": (5, 22), "display": "darkorange", "bgr": (0, 140, 255)},
    "YELLOW": {"hue": (20, 35), "display": "goldenrod", "bgr": (0, 215, 255)},
    "BLUE": {"hue": (95, 130), "display": "blue", "bgr": (255, 0, 0)},
}
ROUGH_MIN_S, ROUGH_MIN_V = 70, 60   # rough search: ignore greyish / dark pixels

PROC_WIDTH = 640            # same processing width as the tracker
MIN_BLOB_PX = 50            # at PROC_WIDTH; smaller blobs are not the marker
MOTION_DIFF = 25            # colour change between frames (largest of B, G, R) that counts as movement
MIN_MOTION_PX = 20          # a blob needs this many moving pixels to count as the marker
# Square test for the marker (skin / hands fail it); at PROC_WIDTH
MIN_ASPECT = 0.6            # shorter / longer side of the rotated bounding box (perspective tilts it)
MIN_EXTENT = 0.75           # blob area / rotated bounding box area (a square fills it)
MIN_SOLIDITY = 0.9          # blob area / convex hull area (no dents such as fingers)
SAT_STEPS = (70, 100, 130, 160, 190)   # rough saturation minimum, raised until marker and hand separate
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
    """Store a calibration for `slot`. Returns a note for the user ("" if nothing special).

    Both markers must have different colours. If the other marker currently has `color`,
    the two swap: the other marker takes over this slot's previous calibration (or goes
    back to its default if this slot had none), so colours can be swapped without first
    calibrating a third colour.
    """
    cal = load_calibration()
    other = SLOTS[1 - SLOTS.index(slot)]
    defaults = dict(zip(SLOTS, default_markers()))   # slot -> default colour name
    note = ""
    if cal.get(other, {}).get("color") == color:
        previous = cal.get(slot)
        if previous and previous["color"] != color:
            cal[other] = previous
            note = f" The {other} marker took over the previous {slot} colour ({previous['color']})."
        else:
            cal.pop(other)
            note = f" The {other} marker is back to its default ({defaults[other]})."
        if other not in cal and defaults[other] == color:
            note += f" That is {color} too: calibrate the {other} marker in another colour."
    elif other not in cal and defaults[other] == color:
        note = f" The {other} marker still uses its default {color}: calibrate it in another colour."
    cal[slot] = {
        "color": color,
        "ranges": [[lo.tolist(), hi.tolist()] for lo, hi in ranges],
        "calibrated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **{k: round(float(v), 4) for k, v in stats.items()},
    }
    CALIBRATION_FILE.write_text(json.dumps(cal, indent=2))
    return note


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


def rough_mask(hsv, color, min_s=ROUGH_MIN_S):
    lo, hi = BASE_COLORS[color]["hue"]
    m = hue_mask(hsv[..., 0], lo, hi) & (hsv[..., 1] >= min_s) & (hsv[..., 2] >= ROUGH_MIN_V)
    m = m.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


def is_square(contour):
    """True for a (possibly tilted, slightly perspective-distorted) square marker.

    Hands, arms and other skin-coloured areas fail at least one of the tests.
    """
    area = cv2.contourArea(contour)
    if area < MIN_BLOB_PX:
        return False
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    if min(w, h) == 0:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return (min(w, h) / max(w, h) >= MIN_ASPECT          # sides about equally long
            and area / (w * h) >= MIN_EXTENT             # fills its rotated bounding box
            and area / hull_area >= MIN_SOLIDITY)        # no dents (fingers, gaps)


def square_blobs(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if is_square(c)]


def motion_image(frame):
    # Lightly blurred frame for the movement test (camera noise must not count as movement)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def motion_score(contour, motion):
    region = np.zeros(motion.shape, np.uint8)
    cv2.drawContours(region, [contour], -1, 1, -1)
    return int((region & motion).sum())


def find_marker_blob(hsv, color, image=None, prev_image=None):
    """Outline of the marker: a SQUARE blob of the base colour that MOVES.

    Movement = colour change between this and the previous frame (images from motion_image;
    colour, not brightness, because a marker can be as bright as the wall behind it).
    If the marker touches the hand holding it, both merge into one blob that is not square;
    the saturation threshold is then raised step by step (skin is much less saturated than
    a coloured marker) until the marker separates. Without images: the largest square.
    Returns None if no suitable blob is found.
    """
    motion = None
    if image is not None:
        if prev_image is None:
            return None
        motion = (cv2.absdiff(image, prev_image).max(axis=2) > MOTION_DIFF).astype(np.uint8)
    for min_s in SAT_STEPS:
        squares = square_blobs(rough_mask(hsv, color, min_s))
        if motion is not None:
            scored = [(motion_score(c, motion), c) for c in squares]
            scored = [(m, c) for m, c in scored if m >= MIN_MOTION_PX]
            if scored:
                return max(scored, key=lambda x: x[0])[1]
        elif squares:
            return max(squares, key=cv2.contourArea)
    return None


def sample_frame(hsv, color, rng, image=None, prev_image=None):
    """Marker and background HSV samples of one (PROC_WIDTH) frame + the marker outline.

    The marker is found with find_marker_blob (moving square). Everything further than
    BG_MARGIN_PX from it is background, including the hand holding it, so the fitted range
    is tightened against skin too. Returns (marker_pixels, background_pixels, contour) or None.
    """
    contour = find_marker_blob(hsv, color, image, prev_image)
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

    Start: hue 1st-99th percentile of the marker + HUE_MARGIN, saturation / brightness a
    little below the marker's lowest values. Then, while background pixels fall into the
    range, take the single step that removes the most background: move the lower or the
    upper hue limit inwards (each side on its own, so e.g. the orange/skin side of red can
    be cut without the pink side), or raise the saturation or brightness minimum. Steps
    that would leave less than MIN_COVERAGE of the marker inside are not taken.

    Returns (ranges, stats) with ranges as the tracker uses them and stats
    {"coverage": share of marker pixels inside, "background": share of background inside}.
    """
    h = marker[:, 0].astype(float) * np.pi / 90.0   # OpenCV hue 0-179 -> angle 0-2π
    center = int(round(np.degrees(np.arctan2(np.sin(h).mean(), np.cos(h).mean())) / 2)) % 180
    d = (marker[:, 0].astype(int) - center + 90) % 180 - 90
    lo_core, hi_core = np.percentile(d, HUE_PERCENTILES)
    state = (int(np.floor(lo_core)) - HUE_MARGIN, int(np.ceil(hi_core)) + HUE_MARGIN,
             int(max(MIN_SV, np.percentile(marker[:, 1], SV_PERCENTILE) - S_MARGIN)),
             int(max(MIN_SV, np.percentile(marker[:, 2], SV_PERCENTILE) - V_MARGIN)))

    def evaluate(lo, hi, s_min, v_min):
        cov = inside(marker, center, lo, hi, s_min, v_min).mean()
        bg = inside(background, center, lo, hi, s_min, v_min).mean() if len(background) else 0.0
        return cov, bg

    cov, bg = evaluate(*state)
    while bg > MAX_BACKGROUND:
        lo, hi, s_min, v_min = state
        options = []
        if lo < hi:
            options += [(lo + 1, hi, s_min, v_min), (lo, hi - 1, s_min, v_min)]
        if s_min < 250:
            options.append((lo, hi, s_min + 5, v_min))
        if v_min < 250:
            options.append((lo, hi, s_min, v_min + 5))
        scored = [(*evaluate(*o), o) for o in options]
        scored = [x for x in scored if x[0] >= MIN_COVERAGE]
        if not scored:
            break   # cannot get cleaner without losing the marker
        cov, bg, state = min(scored, key=lambda x: (x[1], -x[0]))

    lo_off, hi_off, s_min, v_min = state
    lo, hi = (center + lo_off) % 180, (center + hi_off) % 180
    hue_parts = [(lo, hi)] if lo <= hi else [(lo, 179), (0, hi)]
    ranges = [(np.array([a, s_min, v_min], np.uint8), np.array([b, 255, 255], np.uint8))
              for a, b in hue_parts]
    return ranges, {"coverage": cov, "background": bg}


# ----------------------------------------------------------------------------- GUI

class CalibrationTab(ttk.Frame):
    """Set the colour range of the front / back marker by hand, with a live view of exactly
    what the tracker detects with those values.

    Sliders: hue min/max (if min > max the range wraps through red, e.g. 170 -> 10),
    saturation min/max, brightness (value) min/max. Nothing changes for the tracking until
    Save. "Auto-calibrate" only proposes slider values (move the marker for a few seconds).
    The camera is only opened while this tab is shown (activate / deactivate are called by
    the main window, which hands its camera over).
    """

    SLIDERS = (("h_min", "Hue min", 179), ("h_max", "Hue max", 179),
               ("s_min", "Saturation min", 255), ("s_max", "Saturation max", 255),
               ("v_min", "Brightness min", 255), ("v_max", "Brightness max", 255))

    def __init__(self, master, on_saved=None):
        super().__init__(master)
        from video_analysis import MarkerTracker   # its detection is what the live view shows
        self.detector = MarkerTracker(markers={})
        self.on_saved = on_saved   # called after a colour was saved or reset
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None          # newest frame (PROC_WIDTH, BGR) from the camera thread
        self.scale = 1.0           # PROC_WIDTH / camera width (the tracker scales its filters with it)
        self.collecting = None     # samples while auto-calibrating, else None
        self.calib_end = 0.0
        self.rng = np.random.default_rng()
        self.prev_image = None     # previous frame (motion_image), to find the moving marker
        self.saved_values = None   # slider values as last loaded / saved (to show "unsaved")
        self.other_ranges = None   # saved range of the other marker, drawn for comparison
        self.other_name = None

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, text="Marker:").pack(side="left")
        self.slot = tk.StringVar(value="front")
        for slot in SLOTS:
            ttk.Radiobutton(controls, text=slot.capitalize(), value=slot, variable=self.slot,
                            command=self.on_choice).pack(side="left", padx=4)
        ttk.Label(controls, text="Colour:").pack(side="left", padx=(15, 4))
        self.color = ttk.Combobox(controls, values=list(BASE_COLORS), width=10, state="readonly")
        self.color.pack(side="left")
        self.color.bind("<<ComboboxSelected>>", lambda e: self.on_choice(keep_color=True))
        ttk.Button(controls, text="Save", command=self.save).pack(side="left", padx=(15, 4))
        ttk.Button(controls, text="Revert", command=lambda: self.on_choice(keep_color=True)).pack(side="left", padx=4)
        ttk.Button(controls, text="Reset to default", command=self.reset).pack(side="left", padx=4)
        self.start_button = ttk.Button(controls, text="Auto-calibrate (move marker)",
                                       command=self.start_calibration)
        self.start_button.pack(side="left", padx=(15, 4))

        self.status = ttk.Label(self, text="")
        self.status.pack(anchor="w", padx=10)
        self.settings = ttk.Label(self, text="", font=("Courier New", 12), justify="left")
        self.settings.pack(anchor="w", padx=10, pady=(4, 0))

        sliders = ttk.Frame(self)
        sliders.pack(fill="x", padx=10, pady=(8, 0))
        self.vars = {}
        for i, (key, text, top) in enumerate(self.SLIDERS):
            self.vars[key] = tk.IntVar(value=0)
            ttk.Label(sliders, text=text).grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=(0, 6))
            tk.Scale(sliders, from_=0, to=top, orient="horizontal", variable=self.vars[key], length=360,
                     command=lambda v: self.on_slider()).grid(row=i // 2, column=(i % 2) * 2 + 1,
                                                                sticky="ew", padx=(0, 20))
        sliders.columnconfigure(1, weight=1)
        sliders.columnconfigure(3, weight=1)
        ttk.Label(self, text="Hue: 0 red, 15 orange, 30 yellow, 60 green, 105 blue, 165 pink, 179 red. "
                             "If hue min > hue max the range wraps through red (e.g. 170 to 10).",
                  foreground="gray").pack(anchor="w", padx=10)

        self.detected = ttk.Label(self, text="", font=("TkDefaultFont", 13, "bold"))
        self.detected.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(self, text="Left: camera; thick cyan outline = what the tracker detects with the sliders, "
                             "thin outline = the other marker (saved values).   Right: what the tracker sees "
                             "(white = in the colour range, after its noise filtering).",
                  foreground="gray").pack(anchor="w", padx=10)
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
        self.status.config(text="Camera on. Move the sliders until only the marker is outlined, then Save.")

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
                self.frame, self.scale = small, scale

    # ------------------------------------------------------------ slider values <-> ranges

    def slider_values(self):
        return tuple(self.vars[k].get() for k, _, _ in self.SLIDERS)

    def set_sliders(self, ranges):
        lo, hi = ranges[0][0], ranges[-1][1]
        if len(ranges) == 2:   # wrapped through red: [lo..179] + [0..hi]
            h_min, h_max = int(ranges[0][0][0]), int(ranges[1][1][0])
        else:
            h_min, h_max = int(lo[0]), int(hi[0])
        values = (h_min, h_max, int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2]))
        for (key, _, _), v in zip(self.SLIDERS, values):
            self.vars[key].set(v)

    def current_ranges(self):
        h_min, h_max, s_min, s_max, v_min, v_max = self.slider_values()
        hue_parts = [(h_min, h_max)] if h_min <= h_max else [(h_min, 179), (0, h_max)]
        return [(np.array([a, s_min, v_min], np.uint8), np.array([b, s_max, v_max], np.uint8))
                for a, b in hue_parts]

    def start_ranges(self, slot, color):
        # Values to start from: the saved ones for this marker and colour, else the built-in
        # default of this marker, else the rough base-colour range
        entry = load_calibration().get(slot)
        if entry and entry["color"] == color:
            return to_ranges(entry["ranges"]), "saved"
        default_name, default_ranges = list(default_markers().items())[SLOTS.index(slot)]
        if color == default_name:
            return default_ranges, "default"
        lo, hi = BASE_COLORS[color]["hue"]
        hue_parts = [(lo, hi)] if lo <= hi else [(lo, 179), (0, hi)]
        return [(np.array([a, ROUGH_MIN_S, ROUGH_MIN_V], np.uint8), np.array([b, 255, 255], np.uint8))
                for a, b in hue_parts], "rough"

    # ------------------------------------------------------------ choices, save, reset

    def on_choice(self, keep_color=False):
        slot = self.slot.get()
        names = list(load_marker_colors())
        if not keep_color:   # marker switched: show the colour it currently uses
            self.color.set(names[SLOTS.index(slot)])
        ranges, source = self.start_ranges(slot, self.color.get())
        self.set_sliders(ranges)
        self.saved_values = self.slider_values() if source != "rough" else None
        other = SLOTS[1 - SLOTS.index(slot)]
        self.other_name, self.other_ranges = list(load_marker_colors().items())[SLOTS.index(other)]
        self.status.config(text={"saved": "Saved values loaded.", "default": "Built-in default values loaded.",
                                 "rough": "Rough range for this colour loaded (not saved yet)."}[source])
        self.show_settings()

    def on_slider(self):
        if self.saved_values is not None and self.slider_values() == self.saved_values:
            self.status.config(text="No changes.")
        else:
            self.status.config(text="Unsaved changes: Save to use them for tracking.")

    def save(self):
        slot, color = self.slot.get(), self.color.get()
        ranges = self.current_ranges()
        note = save_slot(slot, color, ranges, {})
        self.saved_values = self.slider_values()
        self.status.config(text=f"Saved {slot} = {color} ({describe(ranges)}).{note}")
        self.show_settings()
        if self.on_saved is not None:
            self.on_saved()

    def reset(self):
        reset_slot(self.slot.get())
        self.on_choice()
        self.status.config(text=f"{self.slot.get().capitalize()} marker back to the default colour.")
        if self.on_saved is not None:
            self.on_saved()

    def show_settings(self):
        cal = load_calibration()
        lines = []
        for slot, (name, ranges) in zip(SLOTS, load_marker_colors().items()):
            entry = cal.get(slot)
            when = f"saved {entry['calibrated']}" if entry else "default"
            hues = " + ".join(f"{int(lo[0])}-{int(hi[0])}" for lo, hi in ranges)
            lines.append(f"{slot:5s}  {name:7s} H {hues:13s} S {int(ranges[0][0][1])}-{int(ranges[0][1][1])}  "
                         f"V {int(ranges[0][0][2])}-{int(ranges[0][1][2])}   ({when})")
        self.settings.config(text="In use for tracking:\n" + "\n".join(lines))

    # ------------------------------------------------------------ auto-calibration (proposal)

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
            self.status.config(text=f"Auto-calibrate: moving square marker found in only {len(samples)} frames "
                                    f"(need {MIN_FRAMES}). Set the sliders by hand instead.")
            return
        ranges, stats = fit_ranges(np.vstack([s[0] for s in samples]), np.vstack([s[1] for s in samples]))
        self.set_sliders(ranges)
        self.status.config(text=f"Auto-calibrate proposal: covers {100 * stats['coverage']:.1f} % of the marker, "
                                f"{100 * stats['background']:.3f} % of the background. Adjust if needed, then Save.")

    # ------------------------------------------------------------ live view

    def poll(self):
        with self.lock:
            frame, scale = self.frame, self.scale
        if frame is not None and self.cap is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            image = motion_image(frame)
            shown = frame.copy()
            if self.collecting is not None:
                left = self.calib_end - time.perf_counter()
                sample = sample_frame(hsv, self.color.get(), self.rng, image, self.prev_image)
                if sample is not None:
                    self.collecting.append(sample[:2])
                    cv2.drawContours(shown, [sample[2]], -1, (0, 255, 255), 1)
                self.status.config(text=f"Auto-calibrating: keep moving the marker ({max(left, 0):.1f} s), "
                                        f"found in {len(self.collecting)} frames")
                if left <= 0:
                    self.finish_calibration()
            self.prev_image = image

            # Exactly the tracker's detection: same mask filtering, largest blob, minimum size
            ranges = self.current_ranges()
            mask = self.detector.color_mask(hsv, ranges, scale)
            det = self.detector.find_marker(hsv, ranges, scale=scale)
            other = self.detector.find_marker(hsv, self.other_ranges, scale=scale) if self.other_ranges else None
            if other is not None:
                cv2.polylines(shown, [(other.hull * scale).round().astype(np.int32)], True,
                              BASE_COLORS.get(self.other_name, {}).get("bgr", (200, 200, 200)), 1)
            if det is not None:
                cv2.polylines(shown, [(det.hull * scale).round().astype(np.int32)], True, (255, 255, 0), 3)
                self.detected.config(text=f"Detected: {self.color.get()} at x {det.cx:.0f}, y {det.cy:.0f} "
                                          f"(area {det.area:.0f} px, camera resolution)", foreground="dark green")
            else:
                self.detected.config(text="Not detected with these values", foreground="red")
            self.show_image(self.video_view, cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
            self.show_image(self.mask_view, mask)
        self.after(40, self.poll)

    def show_image(self, label, array):
        photo = ImageTk.PhotoImage(Image.fromarray(array))
        label.config(image=photo)
        label.image = photo   # keep a reference
