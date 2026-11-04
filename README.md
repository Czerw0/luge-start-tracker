# luge-start-tracker

Luge start tracking: computer vision (OpenCV + YOLO) combined with data from a sensor mounted on the sled.

- **Backend:** Python 3.13, FastAPI, OpenCV (YOLO to be added)
- **Frontend:** React + TypeScript, built with Vite (Node.js)
- **Tooling:** [uv](https://docs.astral.sh/uv/) for Python, npm for JavaScript

## Repository structure

```
.
├── pyproject.toml   # Python dependencies (uv)
├── uv.lock
├── backend/main.py  # FastAPI app
└── frontend/        # Vite + React + TypeScript (dev proxy: /api, /ws -> :8000)
```

## Running the app

Use two terminals, both started from the repository root.

The commands are the same on Windows and macOS.

**Terminal 1: backend**

```
uv run fastapi dev backend/main.py
```

- API: http://127.0.0.1:8000/api/health
- Interactive API docs: http://127.0.0.1:8000/docs

**Terminal 2: frontend**

```
cd frontend
npm run dev
```

- App: http://localhost:5173

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

### 3. Install Node.js (22 LTS or newer)

Windows:

```powershell
winget install --id OpenJS.NodeJS.LTS -e
```

macOS:

```bash
brew install node@22
```

Close and reopen the terminal, then check:

```
node --version
npm --version
```

### 4. Clone the repository

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

### 5. Install backend dependencies

From the repository root:

```
uv sync
```

This creates `.venv/` in the root, downloads Python 3.13 if needed and installs all Python dependencies.

### 6. Install frontend dependencies

```
cd frontend
npm install
cd ..
```

### 7. Select the Python interpreter in VS Code

Open the folder in VS Code, press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS), choose **Python: Select Interpreter** and pick `.venv\Scripts\python.exe` (macOS: `.venv/bin/python`) from the repository root. Use the same interpreter for notebooks.


## Managing dependencies

- Python (run from the repository root): `uv add <package>`
- Python dev-only group: `uv add --group <group> <package>`
- JavaScript (run in `frontend/`): `npm install <package>`
- Always commit the updated `uv.lock` and `frontend/package-lock.json`.

## Troubleshooting

- **`uv` or `node` not recognized:** close and reopen the terminal (VS Code too) so PATH is refreshed. On macOS with `node@22`, you may also need `brew link --overwrite node@22`.
- **`import luge_start_tracker` fails:** delete `.venv` and run `uv sync` again.
- **`EBADENGINE` warnings from npm:** your Node.js is too old; install the current LTS.
- **Port already in use:** stop the process using port 8000 or 5173, or change the port.
