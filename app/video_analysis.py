import csv
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


RED = [(np.array([0, 140, 40]), np.array([12, 255, 255])),
       (np.array([165, 140, 40]), np.array([180, 255, 255]))]
GREEN = [(np.array([40, 100, 70]), np.array([85, 255, 255]))]
YELLOW = [(np.array([20, 100, 100]), np.array([35, 255, 255]))]

MIN_AREA = 150
KERNEL = 30   
MAX_LOST_FRAMES = 10  
PROCESS_NOISE = (1, 1, 25, 25)   
MEASUREMENT_NOISE = 9

Detection = namedtuple("Detection", "cx cy area bbox")


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


def save_tracking(df, folder):
    # Save table to <folder>/tracking.csv
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "tracking.csv"
    df.to_csv(path, index=False)
    return path


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
        # name -> HSV ranges
        if markers is None:
            markers = {"yellow": YELLOW, "red": RED}
        self.markers = dict(markers)
        self._tracks = {}  # name -> Kalman filter, size, frames lost
        self._filt = {}    # name -> filtered position, last frame

    def color_mask(self, frame, ranges):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        kernel = np.ones((KERNEL, KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def find_marker(self, frame, ranges, roi=None):
        # Largest colour blob or None; roi = (x0, y0, x1, y1)
        x0, y0 = 0, 0
        if roi is not None:
            x0, y0, x1, y1 = roi
            frame = frame[y0:y1, x0:x1]
            if frame.size == 0:
                return None
        mask = self.color_mask(frame, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour, convex hull, area
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)) + (x0, y0) 
        area = cv2.contourArea(hull)
        if area < MIN_AREA:
            return None

        # Sub-pixel centroid
        m = cv2.moments(hull)
        return Detection(m["m10"] / m["m00"], m["m01"] / m["m00"], area, cv2.boundingRect(hull)) 

    def process_frame(self, frame):
        # Raw detections {name: Detection or None}; filtered in self._filt
        # Lost marker: relaxed search around filter prediction
        result = {}
        self._filt = {}
        for name, ranges in self.markers.items():
            track = self._tracks.get(name)
            if track:
                pred = track["kf"].predict()
                px, py = int(pred[0, 0]), int(pred[1, 0])

            det = self.find_marker(frame, ranges)
            if det is None and track:
                w, h = track["size"]
                around = Detection(px, py, 0, (px - w // 2, py - h // 2, w, h))
                det = self.find_marker(frame, relax(ranges), roi=search_area(around, frame.shape))

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

            self._filt[name] = (float(state[0, 0]), float(state[1, 0])) if state is not None else (np.nan, np.nan)
            result[name] = det
        return result

    def median_position(self, detections):
        found = [d for d in detections.values() if d is not None]
        if not found:
            return np.nan, np.nan
        return (float(np.median([d[0] for d in found])),
                float(np.median([d[1] for d in found])))

    def track_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._tracks.clear()
        rows = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rows.append(self._row(i, i / fps, self.process_frame(frame)))
            i += 1
        cap.release()
        return pd.DataFrame(rows)

    def pair_geometry(self, points):
        # (distance, mid x, mid y) of first two markers, NaN if missing
        a, b = (points[n] for n in list(self.markers)[:2])
        if np.isnan(a[0]) or np.isnan(b[0]):
            return np.nan, np.nan, np.nan
        return float(np.hypot(a[0] - b[0], a[1] - b[1])), (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    def _row(self, i, t, det):
        # Row per frame: <name>_x/_y centre (px), _x_filt/_y_filt Kalman,
        # _size area, _w/_h box; distance, mid_*; x, y, x_filt, y_filt marker median
        x, y = self.median_position(det)
        fx, fy = self.median_position({n: Detection(*p, 0, None) for n, p in self._filt.items()
                                       if not np.isnan(p[0])})
        raw = {n: (d.cx, d.cy) if d else (np.nan, np.nan) for n, d in det.items()}
        dist, mx, my = self.pair_geometry(raw)
        fdist, fmx, fmy = self.pair_geometry(self._filt)

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

    def _format_line(self, row):
        parts = [f"frame {row['frame']:5d}  t={row['t']:7.3f}s"]
        for name in self.markers:
            parts.append(f"{name}: raw=({row[name + '_x']:.2f},{row[name + '_y']:.2f}) "
                         f"kalman=({row[name + '_x_filt']:.2f},{row[name + '_y_filt']:.2f}) "
                         f"A={row[name + '_size']:.0f} {row[name + '_w']:.0f}x{row[name + '_h']:.0f}")
        parts.append(f"dist={row['distance']:.0f} (kalman {row['distance_filt']:.0f}) "
                     f"mid=({row['mid_x']:.2f},{row['mid_y']:.2f})")
        return "  |  ".join(parts)

    def track_camera(self, camera_index=0, save_path=None, csv_path=None, on_frame=None, verbose=True):
        # Live tracking until 'q' or Ctrl+C; save_path = mp4 video,
        # csv_path = rows written as tracked
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {camera_index}")

        self._tracks.clear()
        writer = None
        csv_file = csv_writer = None
        if csv_path is not None:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(csv_path, "w", newline="")
        rows = []
        i = 0
        t0 = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = time.perf_counter() - t0
                if save_path is not None and writer is None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if writer is not None:
                    writer.write(frame)

                det = self.process_frame(frame)
                row = self._row(i, t, det)
                rows.append(row)
                if csv_file is not None:
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_file.flush()
                if verbose:
                    print(self._format_line(row), flush=True)
                i += 1
                if on_frame is not None:
                    on_frame(frame, det)
                cv2.imshow("Live tracking (q = stop)", self.draw(frame, det))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()
            cv2.destroyAllWindows()
        return pd.DataFrame(rows)

    def draw(self, frame, detections):
        out = frame.copy()
        for name, d in detections.items():
            # Yellow ring = Kalman position
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


if __name__ == "__main__":
    # Live recording to analyses/<date>_camera/
    import sys
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    folder = Path(__file__).resolve().parent.parent / "analyses" / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_camera")
    folder.mkdir(parents=True, exist_ok=True)
    df = MarkerTracker().track_camera(cam, save_path=folder / "video.mp4", csv_path=folder / "tracking.csv")
    print(f"{len(df)} frames tracked, saved to {folder}")
