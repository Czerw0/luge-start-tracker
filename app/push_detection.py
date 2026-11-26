"""Push detection from the video position, synced and checked with the IMU.

Method and tests: notebooks/push_detection.ipynb.

- A push starts where the sled stops at the back and is pushed forward = the back turning point of
  the video position x, where the forward acceleration peaks.
- Video and IMU are synced by sliding the IMU position over the video position (correlation).
- Everything on the video side works in pixels with relative thresholds, so the camera position
  does not matter. No speed is taken from the video; push strength comes from the IMU only.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

FS = 60.0              # common sample rate for the sync, Hz
HIGHPASS_HZ = 0.2      # removes IMU drift / position trend, keeps the movement (~0.4 Hz and faster)
MAX_SHIFT_S = 0.5      # largest sync correction that is searched
PROMINENCE = 0.2       # a push peak must stand out by 20 % of the movement range (tested 0.15-0.3: same result)
MIN_GAP_S = 1.0        # at least this much time between two pushes
MIN_PUSH_ACC = 1.0     # m/s², IMU forward acceleration that counts as a push (pushes seen: 1.6-7 m/s²)
MAX_GAP_ERROR = 0.3    # a frame is valid if both markers are found and their distance is within 30 % of its local median
MAX_GAP_DRIFT = 1.0    # ... and at most 100 % away from the distance at rest (seen: wrong objects at 4-5x)


def _highpass(y, fs=FS):
    b, a = butter(2, HIGHPASS_HZ / (fs / 2), btype="high")
    return filtfilt(b, a, y)   # forward + backward: no time shift


def _zscore(y):
    return (y - y.mean()) / y.std()


def valid_frames(video):
    """Frames where the tracking can be trusted: both markers found and their distance normal.

    When the sled leaves the image or a marker is lost, the tracker can jump to other yellow objects;
    the distance between the two markers then becomes wrong or missing. Two checks:
    - within MAX_GAP_ERROR of the median distance of the surrounding second (rolling; perspective can
      change it slowly), catches short jumps;
    - within MAX_GAP_DRIFT of the distance in the first 2 s (sled at rest), catches the tracker staying
      on wrong objects, where the rolling median adapts to the wrong distance.
    """
    if "distance" not in video:
        return video.x.notna()
    d = video.distance
    local = d.rolling(31, center=True, min_periods=5).median()
    start = d[video.t < video.t.iloc[0] + 2.0].median()
    ok = video.x.notna() & d.notna() & ((d - local).abs() <= MAX_GAP_ERROR * local)
    if np.isfinite(start):
        ok &= (d - start).abs() <= MAX_GAP_DRIFT * start
    return ok


def refine_offset(video, imu_result):
    """Extra time shift for the IMU so that its position matches the video position.

    video: tracking DataFrame (t, x in px). imu_result: analyse_start result (t, x in m), already on
    the video time axis via imu_offset_s. Both get the same high-pass (IMU drift out) and a z-score
    (px vs m out); then the IMU is slid by up to MAX_SHIFT_S and the Pearson r is computed per shift.
    Only valid video frames are used (see valid_frames).

    The camera can see the sled moving to the right (+x) or to the left (-x). Both are tried; the one
    with the higher r is the forward direction of the sled in the image.
    Returns (shift_s, r_without_shift, r_best, direction); positive shift = the IMU must move later,
    direction = +1 if forward is +x in the image, -1 if forward is -x.
    """
    video = video[valid_frames(video)]
    t = np.arange(max(video.t.min(), imu_result.t.min()), min(video.t.max(), imu_result.t.max()), 1 / FS)
    v = _zscore(_highpass(np.interp(t, video.t, video.x)))
    i = _zscore(_highpass(np.interp(t, imu_result.t, imu_result.x)))
    n = int(MAX_SHIFT_S * FS)
    shifts = np.arange(-n, n + 1)
    best = None
    for direction in (1, -1):
        r = np.array([np.corrcoef(direction * v[n:-n], np.roll(i, s)[n:-n])[0, 1] for s in shifts])   # ends cut: np.roll wraps
        k = int(np.argmax(r))
        if best is None or r[k] > best[2]:
            best = (shifts[k] / FS, float(r[n]), float(r[k]), direction)
    return best


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


def stroke_end(t, x_forward, t_push, max_len_s=1.5):
    """End of the forward stroke that starts at t_push: furthest forward position in the next max_len_s."""
    after = (t > t_push) & (t <= t_push + max_len_s)
    if not after.any():
        return None
    return float(t[after][np.argmax(x_forward[after])])


def find_pushes(video, imu_result=None, direction=-1):
    """Pushes of one recording as a list of dicts.

    A push starts at the BACK turning point: the sled stops at the back and is pushed forward; this is
    where the forward acceleration peaks. With direction = forward direction in the image (+1 = +x,
    -1 = -x, from refine_offset) the back turning points are the peaks of -direction * x.
    Without IMU the direction is unknown; the default -1 matches the camera setup of the labelled
    recordings (pushes at the peaks of x).

    t_video: back turning point in the video (s).
    t: push time. With IMU (imu_result already synced): the forward-acceleration peak of the IMU close
        to t_video (-0.3 ... +0.5 s), so the marker sits on the push itself; else t_video.
    imu_ok: IMU shows a clear forward push there (acceleration peak >= MIN_PUSH_ACC);
        None without IMU. False = the video peak may be a tracking error.
    t_end: end of the forward stroke (furthest forward position within 1.5 s).
    speed: push strength in m/s = largest forward IMU speed during the stroke. The stroke starts at
        rest, so this is also the speed the push added (delta v). High-passed IMU speed, only a
        ~1.5 s window so drift stays small. NaN without IMU (no speed is taken from the video).
    rel: speed in % of the strongest push of the recording. Only confirmed pushes set the 100 %, so
        a tracking jump cannot become the reference. NaN without IMU.
    """
    video = video.dropna(subset=["x"])   # detect on all tracked frames, then drop peaks on invalid frames
    t_video = video.t.to_numpy()
    x_forward = direction * video.x.to_numpy()   # grows when the sled moves forward
    x_smooth = savgol_filter(x_forward, 9, 2)
    if imu_result is not None:
        t_imu = imu_result.t.to_numpy()
        a_imu = imu_result.a_filt.to_numpy()
        speed_imu = _highpass(imu_result.v.to_numpy(), fs=1 / np.median(np.diff(t_imu)))

    pushes = []
    valid = valid_frames(video).to_numpy()
    for t_turn in detect_pushes(t_video, -x_forward):   # peaks of -x_forward = back turning points
        if not valid[np.argmin(np.abs(t_video - t_turn))]:
            continue   # peak on a frame with broken tracking (marker lost / other object): not a push
        p = {"t_video": float(t_turn), "t": float(t_turn), "imu_ok": None, "speed": np.nan,
             "t_end": stroke_end(t_video, x_smooth, t_turn)}
        end = p["t_end"] if p["t_end"] is not None else t_turn + 1.5
        if imu_result is not None:
            near = (t_imu >= t_turn - 0.3) & (t_imu <= t_turn + 0.5)
            if near.any():
                k = np.argmax(a_imu[near])
                p["t"] = float(t_imu[near][k])
                p["imu_ok"] = bool(a_imu[near][k] >= MIN_PUSH_ACC)
            else:
                p["imu_ok"] = False
            stroke = (t_imu >= p["t"]) & (t_imu <= end)
            if stroke.any():
                p["speed"] = float(speed_imu[stroke].max())
        pushes.append(p)

    reference = [p["speed"] for p in pushes if p["imu_ok"] is not False and np.isfinite(p["speed"])]
    best = max(reference) if reference else np.nan
    for p in pushes:
        p["rel"] = 100 * p["speed"] / best if best > 0 else np.nan
    return pushes
