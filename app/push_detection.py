"""Push detection from the video position, synced and checked with the IMU.

Method and tests: notebooks/push_detection.ipynb.

- A push is where the sled stops and turns around (speed 0) = a peak of the video position x.
- Video and IMU are synced by sliding the IMU position over the video position (correlation).
- Everything on the video side works in pixels with relative thresholds, so the camera position
  and the mm/px scale do not matter.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

FS = 60.0              # common sample rate for the sync, Hz
HIGHPASS_HZ = 0.2      # removes IMU drift / position trend, keeps the movement (~0.4 Hz and faster)
MAX_SHIFT_S = 0.5      # largest sync correction that is searched
PROMINENCE = 0.2       # a push peak must stand out by 20 % of the movement range (tested 0.15-0.3: same result)
MIN_GAP_S = 1.0        # at least this much time between two pushes
STILL = 0.1            # "almost still" = |speed| below 10 % of the largest speed
PAUSE_WINDOW_S = 0.6   # look for the pause this far around a push


def _highpass(y, fs=FS):
    b, a = butter(2, HIGHPASS_HZ / (fs / 2), btype="high")
    return filtfilt(b, a, y)   # forward + backward: no time shift


def _zscore(y):
    return (y - y.mean()) / y.std()


def refine_offset(video, imu_result):
    """Extra time shift for the IMU so that its position matches the video position.

    video: tracking DataFrame (t, x in px). imu_result: analyse_start result (t, x in m), already on
    the video time axis via imu_offset_s. Both get the same high-pass (IMU drift out) and a z-score
    (px vs m out); then the IMU is slid by up to MAX_SHIFT_S and the Pearson r is computed per shift.
    Returns (shift_s, r_without_shift, r_best); positive shift = the IMU must move later.
    """
    video = video.dropna(subset=["x"])
    t = np.arange(max(video.t.min(), imu_result.t.min()), min(video.t.max(), imu_result.t.max()), 1 / FS)
    v = _zscore(_highpass(np.interp(t, video.t, video.x)))
    i = _zscore(_highpass(np.interp(t, imu_result.t, imu_result.x)))
    n = int(MAX_SHIFT_S * FS)
    shifts = np.arange(-n, n + 1)
    r = np.array([np.corrcoef(v[n:-n], np.roll(i, s)[n:-n])[0, 1] for s in shifts])   # ends cut: np.roll wraps
    k = int(np.argmax(r))
    return shifts[k] / FS, float(r[n]), float(r[k])


def detect_pushes(t, x):
    """Times of the pushes: peaks of the (smoothed) position x.

    The peak must stand out by PROMINENCE of the movement range (5th-95th percentile, so single
    tracking jumps do not set the scale) and be at least MIN_GAP_S after the previous one.
    """
    ok = ~np.isnan(x)
    t, x = np.asarray(t)[ok], np.asarray(x)[ok]
    if len(t) < 10:
        return np.array([])
    fps = (len(t) - 1) / (t[-1] - t[0])
    x_smooth = savgol_filter(x, 9, 2)   # ~0.3 s at 30 fps: removes pixel jitter, keeps the peaks
    span = np.percentile(x_smooth, 95) - np.percentile(x_smooth, 5)
    peaks, _ = find_peaks(x_smooth, prominence=PROMINENCE * span, distance=max(1, int(MIN_GAP_S * fps)))
    return t[peaks]


def pause_around(t, speed, t_push):
    """(start, end) of the stretch around t_push where the sled is almost still, or None."""
    still = np.abs(speed) < STILL * np.abs(speed).max()
    in_pause = t[still & (np.abs(t - t_push) < PAUSE_WINDOW_S)]
    return (float(in_pause.min()), float(in_pause.max())) if len(in_pause) else None


def find_pushes(video, imu_result=None):
    """Pushes of one recording as a list of dicts.

    t: push time (peak of the video position, s on the video axis).
    With IMU (imu_result already synced): imu_pause = (start, end) of the IMU pause around the push,
    None if the IMU shows no pause there (then the video peak may be a tracking error).
    """
    video = video.dropna(subset=["x"])
    t_video = video.t.to_numpy()
    pushes = [{"t": float(t)} for t in detect_pushes(t_video, video.x.to_numpy())]
    if imu_result is not None:
        speed_imu = _highpass(imu_result.v.to_numpy(), fs=1 / np.median(np.diff(imu_result.t)))
        for p in pushes:
            p["imu_pause"] = pause_around(imu_result.t.to_numpy(), speed_imu, p["t"])
    return pushes
