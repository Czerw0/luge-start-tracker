# luge-start-tracker

Luge start tracking: computer vision (OpenCV + YOLO) combined with data from a sensor mounted on the sled.

- **Language:** Python 3.13
- **Computer vision:** OpenCV (YOLO to be added)
- **GUI:** tkinter (bundled with Python, no extra install)
- **Tooling:** [uv](https://docs.astral.sh/uv/) for Python and dependencies

## Repository structure

```
.
├── pyproject.toml   # Python dependencies (uv)
├── uv.lock
├── app/
│   ├── main.py             # application entry point (tkinter GUI)
│   ├── video_analysis.py   # video input and sled sticker tracking
│   ├── data_analysis.py    # Movella DOT streaming, IMU data loading and processing
│   ├── imu_panel.py        # live IMU panel of the Measurement tab
│   ├── push_detection.py   # video/IMU fine sync and push detection
│   ├── dashboard.py        # Analysis tab
│   └── video_player.py     # video player of the Analysis tab
├── notebooks/
│   └── push_detection.ipynb   # how the sync and push detection were developed and tested
└── analyses/               # saved analyses (created by the app, not committed)
```

New modules go in `app/` next to `main.py`, with `snake_case` file names.

## Running the app

From the repository root (same command on Windows and macOS):

```
uv run app/main.py
```

### Recording from the Movella DOT IMU

Switch the sensor on and make sure it is **not** connected to the Movella DOT phone app (only one connection at a time).

**Live in the app:** the IMU row at the bottom of the Measurement tab scans for sensors on start. Pick yours by its MAC address (printed on the sensor and shown in the DOT app; the list is sorted by signal strength), click **Connect**. Three live plots show, all in the sensor's own axes: the raw acceleration (with gravity, lying flat az ≈ +9.8 m/s²), the acceleration with gravity removed, and the angular velocity. Gravity is removed using the sensor's orientation, so it stays correct while the sled tilts; the sensor's own reading of gravity is calibrated automatically whenever it lies still.

If the DOT's own orientation is still settling during the first 2 s (its tilt changes by more than 1° while the sled rests, e.g. right after switching the sensor on), it is not used: gravity is then removed with the still-window average, and the Analysis tab shows **IMU orientation: not settled**. Giving the sensor a few seconds on the sled before recording avoids this.

**Recording camera + IMU together:** connect the sensor first, then click **Start analysis**; the analysis does not start while the IMU is not connected. Keep the sled still for the first 2 s (used to calibrate gravity and remove the sensor bias). When you click **Stop analysis**, the analysis folder gets `imu_raw.csv` and `meta.json` (time offset between camera and IMU) next to the video, and the **Analysis** tab shows both on one time axis.

On macOS, the first run asks for Bluetooth permission for VS Code / Terminal. Allow it, otherwise no sensor is found.

### Video player (Analysis tab)

The saved `video.*` of the selected analysis plays on the left of the plots, on the same time axis as the signals (each frame uses its recorded time from `tracking.csv`, so an uneven live frame rate stays in sync). The current frame is a magenta vertical line on every plot, with a dot on the x position; the tracked sled position is circled in red on the video.

- **Play/Pause** at 0.1x-1x speed, step **±1 / ±10** frames, drag the slider, or type a frame number and press **Go**.
- Click a plot to jump the video to that time; click a row in the push table to jump to that push.

### Push detection (Analysis tab)

When an analysis is loaded, the app:
- ends the video signals where the sled leaves the image. The distance between the two markers at rest (first 2 s) is the reference; a frame is invalid if a marker is lost or the distance is more than 20 % off. After the first 0.3 s of invalid frames in a row, the tracker is following background objects, so the position and distance are cut from there (plots, sync and push detection stop). Shorter dropouts (motion blur) are only left out. The video and the IMU signals continue to the end. Recordings are not changed; the cut is made when loading (constants `REST_S`, `EXIT_TOLERANCE`, `EXIT_TIME_S` in `video_analysis.py`);
- refines the camera/IMU sync by sliding the IMU position over the video position (the shift is shown in the **IMU sync** card). It also finds out whether the sled moves to the right or to the left in the image (**Forward in video** card), so the camera can film from either side. Only valid frames are used. If video and IMU still do not match (r < 0.9), the sync is shown as FAILED and not applied;
- detects pushes at the back turning points of the video position `x` (the sled stops at the back and is pushed forward), shown as red triangles; no camera calibration is needed. With IMU data the push time is moved to the IMU forward-acceleration peak next to it (red lines in the acceleration plot). Without IMU data the forward direction is unknown and assumed to be to the left (-x), as in the labelled recordings;
- with IMU data, checks that the IMU shows a forward push there (acceleration peak ≥ 1 m/s²; **Pushes seen by IMU**).

The push table lists each push with its strength: the largest forward IMU speed (m/s) during the stroke that follows the push (until the sled is furthest forward), and the same value in % of the strongest push of the recording (also shown next to the triangles). Only pushes confirmed by the IMU can set the 100 %. No speed is taken from the video, so without IMU data the pushes are listed without a strength.

Markers already saved in an analysis's `markers.json` (e.g. the manual push labels) are shown as orange dotted lines on the plots. The method and its tests are in `notebooks/push_detection.ipynb`.

## Setup from scratch

These steps assume a fresh machine with nothing installed. Each step gives commands for **Windows** (PowerShell) and **macOS** (Terminal, zsh). On macOS, install [Homebrew](https://brew.sh/) first if you don't have it. On Linux, use your package manager.

### 1. Install Git

Windows:

```powershell
winget install --id Git.Git -e
```

macOS:

```bash
brew install git
```

Close and reopen the terminal, then check:

```
git --version
```

Configure your identity (once per machine):

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 2. Install uv

Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS:

```bash
brew install uv
```

Close and reopen the terminal, then check:

```
uv --version
```

You do **not** need to install Python yourself. `uv` downloads the version from `.python-version` automatically.

### 3. Clone the repository

Windows:

```powershell
cd <your desired path>
git clone <repository-url> luge-start-tracker
cd luge-start-tracker
```

macOS:

```bash
cd <your desired path>
git clone <repository-url> luge-start-tracker
cd luge-start-tracker
```

### 4. Install dependencies

From the repository root:

```
uv sync
```

This creates `.venv/` in the root, downloads Python 3.13 if needed and installs all Python dependencies.

### 5. Select the Python interpreter in VS Code

Open the folder in VS Code, press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS), choose **Python: Select Interpreter** and pick `.venv\Scripts\python.exe` (macOS: `.venv/bin/python`) from the repository root. Use the same interpreter for notebooks.


## Managing dependencies

- Python (run from the repository root): `uv add <package>`
- Python dev-only group: `uv add --group <group> <package>`
- Always commit the updated `uv.lock`.

## Troubleshooting

- **`uv` not recognized:** close and reopen the terminal (VS Code too) so PATH is refreshed.
- **`import tkinter` fails:** the Python managed by `uv` includes tkinter. If you use a system Python instead (e.g. Homebrew), install it with `brew install python-tk@3.13`, or delete `.venv` and run `uv sync` so uv's own Python is used.
- **Dependency problems:** delete `.venv` and run `uv sync` again.
