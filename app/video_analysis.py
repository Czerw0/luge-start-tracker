import csv
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PINK =[(np.array([159, 100, 100]), np.array([179, 255, 255]))]
GREEN = [(np.array([30, 60, 60]), np.array([90, 255, 255]))]

MIN_AREA = 150
KERNEL = 30   
PROC_WIDTH = 640      # colour detection runs on a frame scaled to this width (speed); results are scaled back to full-res px
MAX_LOST_FRAMES = 10  
PROCESS_NOISE = (1, 1, 25, 25) # 1 - 1 px/frame, 25 px/frame^2
MEASUREMENT_NOISE = 9

# Sled leaving the frame (tested on the saved recordings): the distance between the two markers
# varies ~1 % at rest and up to ±11 % while the sled moves (perspective). When the sled leaves,
# a marker is lost or the tracker jumps to background objects.
REST_S = 2.0           # first seconds of a recording: sled at rest, reference distance
EXIT_TOLERANCE = 0.2   # a frame is invalid if a marker is missing or the distance is >20 % off the reference
EXIT_TIME_S = 0.3      # invalid this long in a row = the sled has left; shorter dropouts (motion blur) are ignored

# cx, cy: centroid, area: px², bbox: (x, y, w, h); all in full-res px
Detection = namedtuple("Detection", "cx cy area bbox")


def valid_frames(df):
    """Frames where the tracking can be trusted: both markers found and their distance within
    EXIT_TOLERANCE of the distance at rest (median of the first REST_S seconds)."""
    d = df["distance"]
    rest = d[df["t"] < df["t"].iloc[0] + REST_S].median()
    return d.notna() & ((d - rest).abs() <= EXIT_TOLERANCE * rest)


def exit_time(df):
    """Time when the sled has left the frame: the first invalid frame (see valid_frames) after which
    no valid frame follows for at least EXIT_TIME_S. None if the sled stays in view."""
    t = df["t"].to_numpy()
    valid = valid_frames(df).to_numpy()
    good_t = np.append(t[valid], np.inf)
    next_good = good_t[np.searchsorted(good_t, t, side="right")]   # next valid frame after each frame
    exits = ~valid & (next_good - t >= EXIT_TIME_S)
    return float(t[np.argmax(exits)]) if exits.any() else None


def load_tracking(path):
    """Read a tracking.csv and end the video signals where the sled leaves the frame.

    From exit_time() on every column except frame and t is NaN, so plots, sync and push detection
    stop there (the background objects tracked afterwards are noise). All rows are kept, so the
    per-frame times still match the video.
    """
    df = pd.read_csv(path)
    t_exit = exit_time(df)
    if t_exit is not None:
        df.loc[df["t"] >= t_exit, df.columns.difference(["frame", "t"])] = np.nan
    return df


def new_kalman(x, y):
    # Constant velocity, step = frame; state x, y, vx, vy
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.diag(PROCESS_NOISE).astype(np.float32)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    kf.errorCovPost = np.diag([MEASUREMENT_NOISE, MEASUREMENT_NOISE, 100, 100]).astype(np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


def round_sig(v, digits=3):
    # Round to significant digits
    if np.isnan(v) or v == 0:
        return v
    return float(round(v, digits - 1 - int(np.floor(np.log10(abs(v))))))


def relax(ranges):
    # Lower min saturation/brightness (blur washes colour out)
    return [(np.maximum(lo - (0, 60, 20), 0), hi) for lo, hi in ranges]


def search_area(det, frame_shape):
    # Area 3x marker size, clipped to frame
    h, w = frame_shape[:2]
    bx, by, bw, bh = det.bbox
    m = 3 * max(bw, bh)
    return max(bx - m, 0), max(by - m, 0), min(bx + bw + m, w), min(by + bh + m, h)


class MarkerTracker:
    def __init__(self, markers=None):
        # name - HSV ranges
        if markers is None:
            markers = {"GREEN": GREEN, "PINK": PINK}
        self.markers = dict(markers)
        self._tracks = {}  # name - Kalman filter, size, frames lost
        self._filt = {}    # name - filtered position, last frame

    def color_mask(self, hsv, ranges, scale=1.0):
        # hsv: already converted (and possibly downscaled) image
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        k = max(3, round(KERNEL * scale))
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def find_marker(self, hsv, ranges, roi=None, scale=1.0):
        # Largest colour blob or None. hsv is the downscaled image, roi = (x0, y0, x1, y1)
        # in full-res px, returned Detection is in full-res px too.
        x0, y0 = 0, 0
        if roi is not None:
            x0, y0, x1, y1 = (int(v * scale) for v in roi)
            hsv = hsv[y0:y1, x0:x1]
            if hsv.size == 0:
                return None
        mask = self.color_mask(hsv, ranges, scale)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour, convex hull, area
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)) + (x0, y0)
        area = cv2.contourArea(hull)
        if area < MIN_AREA * scale ** 2:
            return None

        # Sub-pixel centroid
        m = cv2.moments(hull)
        bx, by, bw, bh = cv2.boundingRect(hull)
        return Detection(m["m10"] / m["m00"] / scale, m["m01"] / m["m00"] / scale, area / scale ** 2,
                         (round(bx / scale), round(by / scale), round(bw / scale), round(bh / scale)))

    def process_frame(self, frame):
        # Raw detections {name: Detection or None}; filtered in self._filt
        # Lost marker: relaxed search around filter prediction
        result = {}
        self._filt = {}
        # Downscale + HSV once per frame (shared by all markers)
        scale = min(1.0, PROC_WIDTH / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        for name, ranges in self.markers.items():
            track = self._tracks.get(name)
            if track:
                # Predict next position from Kalman filter
                pred = track["kf"].predict()
                px, py = int(pred[0, 0]), int(pred[1, 0])
            # find marker in frame, or if lost, in relaxed search area around predicted position
            det = self.find_marker(hsv, ranges, scale=scale)
            if det is None and track:
                w, h = track["size"]
                around = Detection(px, py, 0, (px - w // 2, py - h // 2, w, h))
                det = self.find_marker(hsv, relax(ranges), roi=search_area(around, frame.shape), scale=scale)

            # Update Kalman filter and store filtered position
            state = None
            if det is not None:
                if track is None:
                    track = self._tracks[name] = {"kf": new_kalman(det.cx, det.cy)}
                else:
                    track["kf"].correct(np.array([[det.cx], [det.cy]], np.float32))
                track["size"] = det.bbox[2:]
                track["lost"] = 0
                state = track["kf"].statePost
            elif track:
                track["lost"] += 1
                if track["lost"] > MAX_LOST_FRAMES:
                    del self._tracks[name]
                else:
                    kf = track["kf"]
                    kf.statePost, kf.errorCovPost = kf.statePre.copy(), kf.errorCovPre.copy()
                    state = kf.statePre
            # Store filtered position (NaN if lost)
            self._filt[name] = (float(state[0, 0]), float(state[1, 0])) if state is not None else (np.nan, np.nan)
            result[name] = det
        return result

    def median_position(self, detections):
        # Detects a median position beetwen detected markers
        # Actual position of the sledges if there is any difference in marksers 
        found = [d for d in detections.values() if d is not None]
        if not found:
            return np.nan, np.nan
        return (float(np.median([d[0] for d in found])),
                float(np.median([d[1] for d in found])))

    def pair_geometry(self, points):
            # (distance, mid x, mid y) of first two markers, NaN if missing
            a, b = (points[n] for n in list(self.markers)[:2])
            if np.isnan(a[0]) or np.isnan(b[0]):
                return np.nan, np.nan, np.nan
            return float(np.hypot(a[0] - b[0], a[1] - b[1])), (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    def track_video(self, video_path, csv_path=None, on_frame=None, stop_event=None):
        # Track markers in a video file (t = frame / fps), return DataFrame with results
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return self._track(cap, lambda i: i / fps, csv_path=csv_path, on_frame=on_frame,
                           stop_event=stop_event)

    def track_camera(self, camera_index=0, save_path=None, csv_path=None, on_frame=None,
                     stop_event=None):
        # Track markers in a live camera (t = seconds since start), return DataFrame with results
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {camera_index}")
        # MJPG gives much higher frame rates than the default format on most webcams
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.t0 = time.perf_counter()   # host clock at video t = 0, saved in meta.json to sync the IMU
        return self._track(cap, lambda i: time.perf_counter() - self.t0, save_path, csv_path,
                           on_frame, stop_event)

    def _track(self, cap, frame_time, save_path=None, csv_path=None, on_frame=None, stop_event=None):
        # Frame loop shared by camera and video file: frame_time(i) -> t in s.
        # Optionally writes the video (save_path) and streams rows to a CSV (csv_path);
        # on_frame(frame, detections, row) is called after every frame.
        # Runs until stop_event is set or no frame is left; always releases cap.
        self._tracks.clear()
        writer = None
        csv_file = csv_writer = None
        if csv_path is not None:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(csv_path, "w", newline="")
        rows = []
        i = 0
        try:
            while stop_event is None or not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_time(i)
                if save_path is not None and writer is None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if writer is not None:
                    writer.write(frame)
                # Process frame and store results
                det = self.process_frame(frame)
                row = self._row(i, t, det)
                rows.append(row)
                if csv_file is not None:
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    if i % 30 == 0:
                        csv_file.flush()
                i += 1
                if on_frame is not None:
                    on_frame(frame, det, row)
        # Release resources and return DataFrame
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()
        return pd.DataFrame(rows)

    def _row(self, i, t, det):
        # Row per frame: <name>_x/_y centre (px), _x_filt/_y_filt Kalman,
        # _size area, _w/_h box; distance, mid_*; x, y, x_filt, y_filt marker median
        x, y = self.median_position(det)
        fx, fy = self.median_position({n: Detection(*p, 0, None) for n, p in self._filt.items()
                                       if not np.isnan(p[0])})
        raw = {n: (d.cx, d.cy) if d else (np.nan, np.nan) for n, d in det.items()}
        dist, mx, my = self.pair_geometry(raw)
        fdist, fmx, fmy = self.pair_geometry(self._filt)

        # Build row dictionary with all values
        row = {"frame": i, "t": t}
        for name, d in det.items():
            row[f"{name}_x"], row[f"{name}_y"] = raw[name]
            row[f"{name}_x_filt"], row[f"{name}_y_filt"] = self._filt[name]
            # size: 3 significant digits
            row[f"{name}_size"] = round_sig(d.area) if d else np.nan
            row[f"{name}_w"], row[f"{name}_h"] = (round_sig(d.bbox[2]), round_sig(d.bbox[3])) if d else (np.nan, np.nan)
        # distance: whole px; positions keep decimals
        row.update(distance=np.round(dist), mid_x=mx, mid_y=my,
                   distance_filt=np.round(fdist), mid_x_filt=fmx, mid_y_filt=fmy,
                   x=x, y=y, x_filt=fx, y_filt=fy)
        return row

    def draw(self, frame, detections):
        out = frame.copy()
        for name, d in detections.items():
            # GREEN ring = Kalman position
            fx, fy = self._filt.get(name, (np.nan, np.nan))
            if not np.isnan(fx):
                cv2.circle(out, (int(fx), int(fy)), 10, (0, 255, 255), 2)
            if d is None:
                continue
            cx, cy, area, (x, y, w, h) = d
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.circle(out, (round(cx), round(cy)), 4, (0, 0, 255), -1)
            cv2.putText(out, f"{name} ({cx:.1f},{cy:.1f}) A={area:.0f}", (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Line between centres, midpoint, distance
        raw = {n: (d.cx, d.cy) if d else (np.nan, np.nan) for n, d in detections.items()}
        dist, mx, my = self.pair_geometry(raw)
        if not np.isnan(dist):
            a, b = (raw[n] for n in list(self.markers)[:2])
            cv2.line(out, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), (0, 255, 0), 1)
            cv2.drawMarker(out, (round(mx), round(my)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(out, f"d={dist:.0f}px", (round(mx) + 12, round(my) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out
