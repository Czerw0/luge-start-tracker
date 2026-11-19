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

AXES = ("x", "y", "z")
G = 9.81   # nominal gravity; analyse_start measures the sensor's own value instead

# Columns of every IMU DataFrame in this project. t = sensor time in s (starts at 0),
# t_host = time.perf_counter() on this computer when the sample arrived (for syncing
# with camera frames recorded on the same machine), q* = orientation quaternion
# (sensor -> earth, earth z up), a* in m/s², g* in rad/s.
# a* and g* are in the SENSOR's own axes and a* still contains gravity (lying flat,
# az ≈ +9.8); use gravity_free() / to_earth() to remove it. Older recordings have no
# q* (and the free-acceleration ones no g*); those columns are then NaN.
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


async def find_dots(timeout=5.0):
    """Return all BLE devices whose name contains 'DOT'.

    On macOS the address is a CoreBluetooth UUID (no MAC address) and differs per Mac,
    so we scan instead of hard-coding it.
    """
    devices = await BleakScanner.discover(timeout=timeout)
    return [d for d in devices if d.name and "dot" in d.name.lower()]


MOVELLA_COMPANY_ID = 0x0886   # Bluetooth company ID in the DOT's advertisement


def scan_dots(timeout=5.0):
    """Scan for DOTs (blocking). Returns a list of dicts, strongest signal first:
    {"address": id to connect with, "mac": e.g. "D4:22:CD:00:45:74", "rssi": dBm}.

    All DOTs are called "Movella DOT"; the MAC address (printed on the sensor and shown
    in the DOT app) tells them apart. It is not readable on macOS directly, but the DOT
    sends it, byte-reversed, in its advertisement. The strongest signal is usually the
    sensor closest to the computer.
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
    app (only one connection at a time). On first use macOS asks for Bluetooth
    permission for VS Code / Terminal; without it the scan finds nothing.
    """

    def __init__(self, address=None):
        self.address = address   # None = use the first DOT found by a scan
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
            address = self.address
            if address is None:
                dots = await find_dots()
                if not dots:
                    raise RuntimeError(
                        "No DOT found. Sensor on (LED blinking)? Not connected to the phone app? "
                        "Bluetooth permission granted?"
                    )
                address = dots[0].address
            async with BleakClient(address) as client:
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


def record_dot(seconds, address=None, connect_timeout=20.0):
    """Record `seconds` of orientation, acceleration and angular velocity from a DOT and
    return an IMU DataFrame.

    Blocks until done, so call it from a script or a worker thread, not the GUI thread.
    The recording time starts once the sensor is connected (scan + connect take a few s).
    Keep the sled still for the first ~2 s: that window is used to estimate the bias.
    """
    streamer = DotStreamer(address)
    streamer.start()
    if not streamer.connected.wait(connect_timeout):
        streamer.stop()
        raise RuntimeError(streamer.error or f"DOT did not connect within {connect_timeout:.0f} s")
    time.sleep(seconds)
    streamer.stop()
    if streamer.error:
        raise RuntimeError(streamer.error)
    return samples_to_dataframe(streamer.samples_so_far())


def samples_to_dataframe(rows):
    """Sample tuples from the DOT -> IMU DataFrame (IMU_COLUMNS).

    Full samples follow SAMPLE_FIELDS. Shorter tuples from older recordings are accepted
    too (missing values become NaN): (t_host, ts_us, ax, ay, az, gx, gy, gz) without
    orientation, and (t_host, ts_us, ax, ay, az) from free-acceleration mode.
    """
    layouts = {
        12: SAMPLE_FIELDS,
        8: ["t_host", "ts_us", "ax", "ay", "az", "gx", "gy", "gz"],
        5: ["t_host", "ts_us", "ax", "ay", "az"],
    }
    df = pd.DataFrame([dict(zip(layouts[len(r)], r)) for r in rows], columns=SAMPLE_FIELDS)
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

    This is the only place that knows about file formats; add new formats here.
    Supported:
    - files written by save_imu() / the course notebook (columns ts_us, ax, ay, az, ...)
    - CSVs from the course script dot_stream.py (SampleTimeFine_us, FreeAcc_X/Y/Z)
    """
    raw = pd.read_csv(path)
    cols = set(raw.columns)
    rows = pd.DataFrame(np.nan, index=raw.index, columns=SAMPLE_FIELDS)
    if {"ts_us", "ax", "ay", "az"} <= cols:
        for c in SAMPLE_FIELDS:
            if c in cols:
                rows[c] = raw[c]
        if "t_host" not in cols and "t_host_s" in cols:
            rows["t_host"] = raw["t_host_s"]
    elif {"SampleTimeFine_us", "FreeAcc_X", "FreeAcc_Y", "FreeAcc_Z"} <= cols:
        rows["t_host"] = raw.get("t_host_s", np.nan)
        rows["ts_us"] = raw["SampleTimeFine_us"]
        rows["ax"], rows["ay"], rows["az"] = raw["FreeAcc_X"], raw["FreeAcc_Y"], raw["FreeAcc_Z"]
    else:
        raise ValueError(f"Unknown IMU file format in {path}. Columns: {list(raw.columns)}")
    # Undo any earlier wrap correction so samples_to_dataframe sees raw uint32 values again.
    rows["ts_us"] = rows["ts_us"].astype(np.int64) % 2**32
    return samples_to_dataframe(rows.itertuples(index=False, name=None))


def save_imu(df, path):
    """Save an IMU DataFrame as CSV (readable again with load_imu)."""
    df[IMU_COLUMNS].to_csv(path, index=False)


# ----------------------------------------------------------------------------- gravity

def has_orientation(df):
    """True if every sample carries an orientation quaternion (mode 26 recordings)."""
    return len(df) > 0 and bool(df[["qw", "qx", "qy", "qz"]].notna().all().all())


def orientation(df):
    """Per-sample rotation sensor axes -> earth frame (scipy Rotation).

    The DOT sends w, x, y, z; scipy wants x, y, z, w.
    """
    return Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy(float))


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


def to_earth(df, g=G):
    """Linear acceleration in the EARTH frame (n x 3, z up, heading arbitrary), gravity removed."""
    a = df[["ax", "ay", "az"]].to_numpy(float)
    return orientation(df).apply(a) - [0.0, 0.0, g]


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


def axis_vector(forward):
    """'x', '-y', ... or a 3-vector -> unit vector."""
    if isinstance(forward, str):
        v = np.zeros(3)
        v["xyz".index(forward.lstrip("+-"))] = -1.0 if forward.startswith("-") else 1.0
        return v
    v = np.asarray(forward, dtype=float)
    return v / np.linalg.norm(v)


def analyse_start(df, fs=60.0, t_still=(0.3, 1.8), k=6.0, fc=15.0, forward=None):
    """Forward acceleration, speed and distance of the sled (chain from the course notebook).

    df: IMU DataFrame (load_imu / record_dot). fs: resampling rate, set close to the
    sensor rate (DOT default 60 Hz). t_still: window in s where the sled is at rest.
    forward: sled axis in sensor coordinates ('x', '-y', or a vector); None = find it
    from the data (principal component of the movement).

    With orientation (mode 26 recordings): the acceleration is first rescaled so that the
    sensor reads exactly G at rest (fixes its scale error), then gravity is removed per
    sample in the sensor's axes; the rest of the still-window mean is the accelerometer
    bias, which lives in the sensor's axes too.
    Without orientation (older files) the still-window mean removes bias and gravity
    together, which is only right as long as the tilt does not change.
    Returns (result DataFrame with t, lx, ly, lz, a_fwd, a_filt, v, x; info dict).
    """
    if has_orientation(df):
        g = estimate_gravity(df, t_still)
        scaled = df.copy()
        scaled[["ax", "ay", "az"]] *= G / g   # sensor scale error, measured at rest
        lin = gravity_free(scaled, G)
        frame = "sensor axes, gravity removed with the DOT orientation"
    else:
        g = None
        lin = df[["ax", "ay", "az"]].to_numpy(float)
        frame = "no orientation in data: still-window mean removed (tilt assumed constant)"
    t, acc = resample_uniform(pd.DataFrame({"t": df["t"], "lx": lin[:, 0], "ly": lin[:, 1],
                                            "lz": lin[:, 2]}), fs, ("lx", "ly", "lz"))
    still = still_mask(t, t_still)
    if not still.any():
        raise ValueError(f"No samples in the still window {t_still} s.")
    acc = acc - acc[still].mean(axis=0)   # remaining accelerometer bias per axis

    direction = direction_of_motion(acc, t, t_still, k) if forward is None else axis_vector(forward)
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
        "g": g,                   # gravity measured by this sensor at rest (m/s²), None without orientation
        "scale": G / g if g else None,   # factor applied to the acceleration
        "direction": direction,   # forward unit vector in the frame above
        "t_start": float(t[i0]),
        "i_start": i0,
        # Sanity check: if the sled is still again at the end, v should be ~0.
        # Whatever is left is the accumulated drift.
        "v_end": float(v[-1]),
    }
    return result, info


# ----------------------------------------------------------------------------- drift check

def cumulative_trapezoid(values, time):
    """Integrate `values` over `time` step by step (trapezoid rule), starting at 0."""
    steps = 0.5 * (values[1:] + values[:-1]) * np.diff(time)[:, None]
    return np.vstack([np.zeros((1, values.shape[1])), np.cumsum(steps, axis=0)])


class Imu:
    """IMU recording: time in seconds, accelerations in m/s² as columns x, y, z."""

    def __init__(self, time, acceleration):
        self.time = np.asarray(time, dtype=float)
        self.acceleration = np.asarray(acceleration, dtype=float)
        if self.acceleration.shape != (len(self.time), 3):
            raise ValueError("acceleration must have shape (len(time), 3)")

    @classmethod
    def from_dataframe(cls, df):
        """Build from an IMU DataFrame (load_imu / record_dot)."""
        return cls(df["t"], df[["ax", "ay", "az"]])

    def velocity(self):
        """Velocity in m/s: acceleration integrated once."""
        return cumulative_trapezoid(self.acceleration, self.time)

    def position(self):
        """Position in m: acceleration integrated twice."""
        return cumulative_trapezoid(self.velocity(), self.time)

    def drift(self):
        """Drift per axis: the position at the end of the recording.

        Use this on a recording where the sensor is lying still. The true position
        then stays 0, so everything the double integration produces is drift caused
        by sensor bias and noise.
        """
        final = self.position()[-1]
        return dict(zip(AXES, final))


if __name__ == "__main__":
    # Record from a Movella DOT, save the CSV, and print the start analysis:
    #   uv run app/data_analysis.py 10        (seconds; keep the sled still for the first 2 s)
    import sys

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    df = record_dot(seconds)
    fname = time.strftime("dot_%Y%m%d_%H%M%S.csv")
    save_imu(df, fname)
    print(f"{len(df)} samples, ~{(len(df) - 1) / df['t'].iloc[-1]:.1f} Hz, saved {fname}")
    result, info = analyse_start(df)
    print(f"frame: {info['frame']}" + (f", measured g = {info['g']:.3f} m/s²" if info["g"] else ""))
    print(f"start at t = {info['t_start']:.2f} s, forward direction {np.round(info['direction'], 3)}")
    print(f"max speed {result['v'].max():.2f} m/s, v at end (drift) {info['v_end']:.2f} m/s")
