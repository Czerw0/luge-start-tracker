"""IMU data loading and processing (accelerations a_x..a_z, angular velocities ω_x..ω_z).

Movella DOT streaming, loading and the processing chain (bias removal, direction of
motion, low-pass, start detection, integration) are adapted from the course material
in mci-luge-start: 03_Movella_DOT_IMU.ipynb, dot_stream.py and imu_kinematics.py
(MCI Device Development in Top Class Sports). Based on the Movella DOT BLE Service
Specification (XD0506P).
"""

import asyncio
import queue
import struct
import threading
import time

import numpy as np
import pandas as pd
from bleak import BleakClient, BleakScanner
from scipy import integrate
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

G = 9.81   # nominal gravity; analyse_start measures the sensor's own value instead
MAX_ORIENTATION_DRIFT_DEG = 1.0   # orientation counts as settled if its tilt changes less while still (seen: <=0.1° ok, 4° not)

# Columns of every IMU DataFrame in this project. t = sensor time in s (starts at 0),
# t_host = time.perf_counter() on this computer when the sample arrived (for syncing
# with camera frames recorded on the same machine), q* = orientation quaternion
# (sensor -> earth, earth z up), a* in m/s², g* in rad/s.
# a* and g* are in the SENSOR's own axes and a* still contains gravity (lying flat,
# az ≈ +9.8); use gravity_free() to remove it.
IMU_COLUMNS = ["t", "t_host", "ts_us", "qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz"]
# Fields of one live sample, as delivered by DotStreamer
SAMPLE_FIELDS = ["t_host", "ts_us", "qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz"]


# ----------------------------------------------------------------------------- Movella DOT (BLE)

MEASUREMENT_CHAR = "15172001-4947-11e9-8646-d663bd873d93"
LONG_PAYLOAD_CHAR = "15172002-4947-11e9-8646-d663bd873d93"
# Measurement control: type=1, start=1/stop=0, payload mode 26 = orientation (quaternion)
# + acceleration + angular velocity, delivered on the long payload characteristic
START_MEASUREMENT = b"\x01\x01\x1a"
STOP_MEASUREMENT = b"\x01\x00\x1a"


def parse_measurement(data):
    """Long payload in mode 26, little endian: uint32 timestamp [µs], float32 quaternion
    w, x, y, z, float32 acc x/y/z [m/s²], float32 gyro x/y/z [deg/s] (rest is padding).

    Returns (ts_us, qw, qx, qy, qz, ax, ay, az, gx, gy, gz) with g* in rad/s.
    Acceleration and angular velocity are in the sensor's own axes, acceleration includes
    gravity. The quaternion rotates sensor axes into the earth frame (z up).
    Checked on a DOT lying still: |q| = 1.000; rotating a into the earth frame gives
    (0.01, 0.06, 9.61) m/s², i.e. pure gravity, and the same horizontal values as the
    sensor's own free acceleration (mode 6). The other quaternion orders leave 1-8 m/s²
    horizontal. Gyro at rest ≈ 0.5 deg/s (0.5 rad/s would be implausible).
    """
    ts_us, qw, qx, qy, qz, ax, ay, az, gx, gy, gz = struct.unpack_from("<I10f", data, 0)
    return ts_us, qw, qx, qy, qz, ax, ay, az, np.radians(gx), np.radians(gy), np.radians(gz)


MOVELLA_COMPANY_ID = 0x0886   # Bluetooth company ID in the DOT's advertisement


def scan_dots(timeout=5.0):
    """Scan for DOTs (blocking). Returns a list of dicts, strongest signal first:
    {"address": id to connect with, "mac": e.g. "D4:22:CD:00:45:74", "rssi": dBm}.

    All DOTs are called "Movella DOT"; the MAC address (printed on the sensor and shown
    in the DOT app) tells them apart. It is not readable on macOS directly, but the DOT
    sends it, byte-reversed, in its advertisement. The strongest signal is usually the
    sensor closest to the computer. On macOS the address is a CoreBluetooth UUID (no MAC
    address) and differs per Mac, so we scan instead of hard-coding it.
    """
    async def scan():
        return await BleakScanner.discover(timeout=timeout, return_adv=True)

    found = []
    for device, adv in asyncio.run(scan()).values():
        if not (device.name and "dot" in device.name.lower()):
            continue
        raw = adv.manufacturer_data.get(MOVELLA_COMPANY_ID, b"")
        mac = ":".join(f"{b:02X}" for b in reversed(raw[:6])) if len(raw) >= 6 else "?"
        found.append({"address": device.address, "mac": mac, "rssi": adv.rssi})
    return sorted(found, key=lambda d: d["rssi"], reverse=True)


class DotStreamer:
    """Streams orientation + acceleration + angular velocity (sensor axes, gravity
    included) from a Movella DOT on a background thread.

    The asyncio/Bluetooth loop runs in its own thread so the tkinter GUI never blocks.
    Samples land in self.samples as tuples in SAMPLE_FIELDS order; call samples_so_far()
    to fetch them. Before running: sensor switched on and NOT connected to the DOT phone
    app (only one connection at a time). address: from scan_dots(). On first use macOS
    asks for Bluetooth permission for VS Code / Terminal; without it the scan finds nothing.
    """

    def __init__(self, address):
        self.address = address
        self.samples = queue.Queue()
        self.connected = threading.Event()
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.connected.clear()
        self.error = None
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def samples_so_far(self):
        """Remove and return all samples received since the last call."""
        rows = []
        while True:
            try:
                rows.append(self.samples.get_nowait())
            except queue.Empty:
                return rows

    async def _run(self):
        try:
            async with BleakClient(self.address) as client:
                def on_data(_, data):
                    self.samples.put((time.perf_counter(), *parse_measurement(data)))

                await client.start_notify(LONG_PAYLOAD_CHAR, on_data)
                await client.write_gatt_char(MEASUREMENT_CHAR, START_MEASUREMENT, response=True)
                self.connected.set()
                while not self._stop.is_set():
                    await asyncio.sleep(0.05)
                await client.write_gatt_char(MEASUREMENT_CHAR, STOP_MEASUREMENT, response=True)
                await client.stop_notify(LONG_PAYLOAD_CHAR)
        except Exception as e:
            self.error = str(e) or type(e).__name__


def samples_to_dataframe(rows):
    """Sample tuples from the DOT (SAMPLE_FIELDS order) -> IMU DataFrame (IMU_COLUMNS)."""
    df = pd.DataFrame(list(rows), columns=SAMPLE_FIELDS)
    ts = df["ts_us"].to_numpy(np.int64)
    if len(ts):
        ts = ts + np.concatenate([[0], np.cumsum(np.diff(ts) < 0)]) * 2**32   # uint32 wraps every ~71 min
        df["ts_us"] = ts
        df["t"] = (ts - ts[0]) / 1e6   # sensor time [s], starts at 0
    else:
        df["t"] = pd.Series(dtype=float)
    return df[IMU_COLUMNS].astype(float)


# ----------------------------------------------------------------------------- files

def load_imu(path):
    """Load an IMU file and return a DataFrame with IMU_COLUMNS.

    This is the only place that knows about the file format; add new formats here.
    Supported: CSVs written by save_imu() (imu_raw.csv of a saved analysis).
    """
    raw = pd.read_csv(path)
    missing = [c for c in IMU_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Unknown IMU file format in {path}. Missing columns: {missing}")
    return raw[IMU_COLUMNS].astype(float)


def save_imu(df, path):
    """Save an IMU DataFrame as CSV (readable again with load_imu)."""
    df[IMU_COLUMNS].to_csv(path, index=False)


# ----------------------------------------------------------------------------- gravity

def orientation(df):
    """Per-sample rotation sensor axes -> earth frame (scipy Rotation).

    The DOT sends w, x, y, z; scipy wants x, y, z, w.
    """
    return Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy(float))


def orientation_drift(df, t_still):
    """How much the sensor's own tilt estimate changes while the sled rests (degrees).

    The sled does not move in the still window, so the true tilt is constant. Right after the DOT is
    switched on or reconnected its orientation filter may still be settling (seen: roll drifting from
    -20° to -9° over 8 s). Gravity removed with such a quaternion leaves a fake, slowly changing
    acceleration of several m/s². Settled recordings change by ~0.1° or less.
    """
    still = still_mask(df["t"].to_numpy(), t_still)
    if still.sum() < 2:
        return 0.0
    g = orientation(df[still]).inv().apply([0.0, 0.0, 1.0])   # gravity direction in sensor axes
    return float(np.degrees(np.arccos(np.clip(g[0] @ g[-1], -1.0, 1.0))))


def estimate_gravity(df, t_still):
    """Gravity as THIS sensor measures it: mean |a| while the sled rests.

    Accelerometers have a small scale error (one DOT read 9.61 instead of 9.81 m/s²).
    analyse_start uses it to rescale the acceleration to the true G before removing
    gravity; otherwise the error stays as a fake vertical acceleration (drift) and every
    acceleration is that much too small.
    """
    still = still_mask(df["t"].to_numpy(), t_still)
    if not still.any():
        return G
    return float(np.linalg.norm(df.loc[still, ["ax", "ay", "az"]].to_numpy(float), axis=1).mean())


def gravity_free(df, g=G):
    """Linear acceleration in the SENSOR's axes (n x 3): a minus gravity rotated into
    the sensor frame with the orientation of each sample.

    Correct even while the sled pitches and rolls, unlike subtracting a constant.
    """
    a = df[["ax", "ay", "az"]].to_numpy(float)
    return a - orientation(df).inv().apply([0.0, 0.0, g])


# ----------------------------------------------------------------------------- processing

def resample_uniform(df, fs, columns=("ax", "ay", "az")):
    """BLE IMUs deliver jittery timestamps; filters and integration assume constant dt.

    Returns t (uniform, 1/fs steps) and an (n, len(columns)) array.
    """
    t = np.arange(df["t"].iloc[0], df["t"].iloc[-1], 1.0 / fs)
    values = np.column_stack([np.interp(t, df["t"], df[c]) for c in columns])
    return t, values


def lowpass(y, fs, fc, order=4):
    """Zero-phase Butterworth low-pass (offline only: filtfilt has no lag but needs all data)."""
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, y, axis=0)


def still_mask(t, t_still):
    """Samples inside the stationary window (sled resting before the start)."""
    return (t >= t_still[0]) & (t <= t_still[1])


def direction_of_motion(acc, t, t_still, k=6.0):
    """Dominant direction of the acceleration during the movement (principal component).

    Works in whatever frame `acc` is in. With orientation data analyse_start passes
    gravity-free acceleration in the sensor's axes, so the result is the sled's forward
    axis as seen by the sensor (the sensor need not be mounted exactly straight).
    Sign: net acceleration is forward. `acc` must already have the bias removed.
    """
    still = still_mask(t, t_still)
    moving = np.linalg.norm(acc, axis=1) > k * np.linalg.norm(acc[still].std(axis=0))
    if moving.sum() < 2:
        raise ValueError("No movement found in the IMU data.")
    # Unlike the course notebook, the samples are not mean-centred before the SVD:
    # centring removes the push itself when the acceleration is roughly constant, and the
    # direction then follows the noise (tested: up to ~90° off for a steady push,
    # < 0.3° without centring; for pulsed pushes both are fine).
    _, _, vt = np.linalg.svd(acc[moving], full_matrices=False)
    direction = vt[0]
    if (acc[moving] @ direction).sum() < 0:
        direction = -direction
    return direction


def detect_start(a, t, t_still, k=6.0):
    """Index of the first sample after the still window where |a| > k * its noise std.

    Returns None if the signal never leaves the noise band.
    """
    sigma = a[still_mask(t, t_still)].std()
    above = (np.abs(a) > k * sigma) & (t > t_still[1])
    return int(np.argmax(above)) if above.any() else None


def integrate_from(a, t, i0):
    """a -> v -> x with v(t0)=0, x(t0)=0 (sled at rest at the start).

    Everything before i0 stays zero so pure noise is not integrated. Every constant
    error b left in a becomes b·t in v and ½·b·t² in x (drift), so keep the evaluation
    window short.
    """
    v = np.zeros_like(a)
    x = np.zeros_like(a)
    v[i0:] = integrate.cumulative_trapezoid(a[i0:], t[i0:], initial=0.0)
    x[i0:] = integrate.cumulative_trapezoid(v[i0:], t[i0:], initial=0.0)
    return v, x


def analyse_start(df, fs=60.0, t_still=(0.3, 1.8), k=6.0, fc=15.0):
    """Forward acceleration, speed and distance of the sled (chain from the course notebook).

    df: IMU DataFrame (load_imu / samples_to_dataframe). fs: resampling rate, set close to the
    sensor rate (DOT default 60 Hz). t_still: window in s where the sled is at rest.
    The forward axis is found from the data (principal component of the movement).

    The acceleration is first rescaled so that the sensor reads exactly G at rest (fixes its
    scale error), then gravity is removed per sample in the sensor's axes with the DOT
    orientation; the rest of the still-window mean is the accelerometer bias, which lives in
    the sensor's axes too.
    When the DOT orientation is still settling during the still window (see orientation_drift),
    it is not used: the still-window mean then removes bias and gravity together, which is
    only right as long as the tilt does not change.
    Returns (result DataFrame with t, lx, ly, lz, a_fwd, a_filt, v, x; info dict).
    """
    drift = orientation_drift(df, t_still)
    if drift <= MAX_ORIENTATION_DRIFT_DEG:
        g = estimate_gravity(df, t_still)
        scaled = df.copy()
        scaled[["ax", "ay", "az"]] *= G / g   # sensor scale error, measured at rest
        lin = gravity_free(scaled, G)
        frame = "sensor axes, gravity removed with the DOT orientation"
    else:
        g = None
        lin = df[["ax", "ay", "az"]].to_numpy(float)
        frame = (f"orientation not settled ({drift:.1f}° change while still), not used: "
                 "still-window mean removed (tilt assumed constant)")
    t, acc = resample_uniform(pd.DataFrame({"t": df["t"], "lx": lin[:, 0], "ly": lin[:, 1],
                                            "lz": lin[:, 2]}), fs, ("lx", "ly", "lz"))
    still = still_mask(t, t_still)
    if not still.any():
        raise ValueError(f"No samples in the still window {t_still} s.")
    acc = acc - acc[still].mean(axis=0)   # remaining accelerometer bias per axis

    direction = direction_of_motion(acc, t, t_still, k)
    a_fwd = acc @ direction
    a_filt = lowpass(a_fwd, fs, fc=min(fc, 0.4 * fs))
    i0 = detect_start(a_filt, t, t_still, k)
    if i0 is None:
        raise ValueError("No start detected: acceleration never leaves the noise of the still window.")
    v, x = integrate_from(a_filt, t, i0)

    result = pd.DataFrame({"t": t, "lx": acc[:, 0], "ly": acc[:, 1], "lz": acc[:, 2],
                           "a_fwd": a_fwd, "a_filt": a_filt, "v": v, "x": x})
    info = {
        "frame": frame,
        "g": g,                   # gravity measured by this sensor at rest (m/s²), None if the orientation was not used
        "orientation_drift_deg": drift,   # tilt change of the DOT orientation while still
        "scale": G / g if g else None,   # factor applied to the acceleration
        "direction": direction,   # forward unit vector in the frame above
        "t_start": float(t[i0]),
        "i_start": i0,
        # Sanity check: if the sled is still again at the end, v should be ~0.
        # Whatever is left is the accumulated drift.
        "v_end": float(v[-1]),
    }
    return result, info
