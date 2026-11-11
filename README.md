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
│   └── data_analysis.py    # IMU data loading and processing
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

**Live in the app:** the IMU row at the bottom of the Measurement tab scans for sensors on start. Pick yours by its MAC address (printed on the sensor and shown in the DOT app; the list is sorted by signal strength), click **Connect**, and the free acceleration is shown and plotted live.

**From the command line**, from the repository root:

```
uv run app/data_analysis.py 10
```

This records 10 s of free acceleration over Bluetooth, saves it as `dot_<date>_<time>.csv` in the current folder, and prints the detected start and speed. Keep the sled still for the first 2 s; that window is used to remove the sensor bias.

On macOS, the first run asks for Bluetooth permission for VS Code / Terminal. Allow it, otherwise no sensor is found.

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
