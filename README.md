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

**Terminal 1: backend**

```powershell
uv run fastapi dev backend/main.py
```

- API: http://127.0.0.1:8000/api/health
- Interactive API docs: http://127.0.0.1:8000/docs

**Terminal 2: frontend**

```powershell
cd frontend
npm run dev
```

- App: http://localhost:5173

## Setup from scratch

These steps assume a fresh Windows 11 machine with nothing installed. Commands are for PowerShell. On macOS or Linux, use the equivalent installers from each tool's website.

### 1. Install Git

```powershell
winget install --id Git.Git -e
```

Close and reopen PowerShell, then check:

```powershell
git --version
```

Configure your identity (once per machine):

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 2. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then check:

```powershell
uv --version
```

You do **not** need to install Python yourself. `uv` downloads the version from `.python-version` automatically.

### 3. Install Node.js (22 LTS or newer)

```powershell
winget install --id OpenJS.NodeJS.LTS -e
```

Close and reopen PowerShell, then check:

```powershell
node --version
npm --version
```

### 4. Clone the repository

```powershell
cd $<your desired path>
git clone <repository-url> luge-start-tracker
cd luge-start-tracker
```

### 5. Install backend dependencies

From the repository root:

```powershell
uv sync
```

This creates `.venv/` in the root, downloads Python 3.13 if needed and installs all Python dependencies.

### 6. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

### 7. Select the Python interpreter in VS Code

Open the folder in VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter** and pick `.venv\Scripts\python.exe` from the repository root. Use the same interpreter for notebooks.



## Managing dependencies

- Python (run from the repository root): `uv add <package>`
- Python dev-only group: `uv add --group <group> <package>`
- JavaScript (run in `frontend/`): `npm install <package>`
- Always commit the updated `uv.lock` and `frontend/package-lock.json`.

## Troubleshooting

- **`uv` or `node` not recognized:** close and reopen the terminal (VS Code too) so PATH is refreshed.
- **`import luge_start_tracker` fails:** delete `.venv` and run `uv sync` again.
- **`EBADENGINE` warnings from npm:** your Node.js is too old; install the current LTS.
- **Port already in use:** stop the process using port 8000 or 5173, or change the port.
