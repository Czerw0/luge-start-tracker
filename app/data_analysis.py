"""IMU data loading and processing (accelerations a_x..a_z, angular velocities ω_x..ω_z)."""

import numpy as np

AXES = ("x", "y", "z")


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
    # Example: sensor lying still for 10 s at 100 Hz, with a small bias and noise.
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 0.01)
    bias = np.array([0.02, -0.01, 0.005])
    acc = bias + rng.normal(0, 0.05, size=(len(t), 3))

    imu = Imu(t, acc)
    for axis, value in imu.drift().items():
        print(f"drift {axis}: {value:.3f} m")
