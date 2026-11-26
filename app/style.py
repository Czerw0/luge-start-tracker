"""Look of the app in one place: colours, fonts, the ttk theme and the matplotlib plot style.

Clean light style built on tkinter's own "clam" theme (no extra packages). Call apply_theme(root)
once, right after creating the Tk window and before any plot is created, so every tab and every
matplotlib figure uses the same look.
"""
import ctypes
import sys
from tkinter import font as tkfont, ttk

import matplotlib as mpl
from matplotlib import cycler

# Base colours
BG = "#eef1f5"           # window background
SURFACE = "#ffffff"      # cards, tables, plot areas
BORDER = "#c5cdd6"       # card and plot frames
GRID = "#e4e8ed"
TEXT = "#111827"
MUTED = "#4b5563"        # secondary text, axis labels
ACCENT = "#2563eb"       # main buttons, selected tab, selection
ACCENT_HOVER = "#1d4ed8"
ACCENT_LIGHT = "#dbe6fd" # selected table row, hovered buttons
VIDEO_BG = "#1c2026"     # behind the video image

# Signal colours: the same signal has the same colour in every tab
POS_X = "#2563eb"        # marker x position
POS_Y = "#f59e0b"        # marker y position
IMU_SPEED = "#dc2626"    # IMU speed
PUSH = "#dc2626"         # detected pushes
CURSOR = "#c026d3"       # current video frame
MARKER = "#ea7c0c"       # saved markers (markers.json)
AXES_COLORS = {"x": "#dc2626", "y": "#16a34a", "z": "#2563eb"}   # sensor axes x / y / z
MARKER_COLORS = {"GREEN": "#16a34a", "PINK": "#db2777"}          # tracked stickers
CYCLE = ["#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]

MONO = ("Consolas", 10)  # fixed-width text (falls back to the system monospace font)
SCALE = 1.0              # screen scaling (1.25 at 125 % in Windows), set by apply_theme


def enable_sharp_rendering():
    """Windows: draw at the screen's real resolution. Call before creating the Tk window.

    Without this, Windows renders the app at 96 dpi and stretches it to the display scaling
    (e.g. 125 %), which makes text and plots blurry. Fonts are in points, so they keep their size;
    sizes given in pixels are scaled with px(). Other platforms handle this themselves.
    """
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # system DPI aware
        except (AttributeError, OSError):   # very old Windows: keep the stretched rendering
            pass


def px(value):
    """A size in pixels at 100 % scaling -> pixels on this screen."""
    return round(value * SCALE)


def font_family():
    # The platform's UI font (Segoe UI on Windows, the system font on macOS)
    return tkfont.nametofont("TkDefaultFont").actual("family")


def apply_theme(root):
    """Style all ttk widgets of this Tk root and set the matplotlib defaults."""
    global SCALE
    if sys.platform == "win32":
        SCALE = max(1.0, root.winfo_fpixels("1i") / 96)   # 96 dpi = 100 % in Windows
    family = font_family()
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name).configure(size=10)
    root.configure(background=BG)
    root.option_add("*TCombobox*Listbox.background", SURFACE)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_LIGHT)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT)

    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=BG, foreground=TEXT, font=(family, 10), bordercolor=BORDER,
                lightcolor=BG, darkcolor=BG, troughcolor=GRID, focuscolor=ACCENT,
                selectbackground=ACCENT_LIGHT, selectforeground=TEXT)
    s.configure("TFrame", background=BG)
    s.configure("TLabel", background=BG, foreground=TEXT)
    s.configure("Muted.TLabel", foreground=MUTED)
    s.configure("Title.TLabel", font=(family, 12, "bold"))

    # Buttons: white with a thin border; Accent.TButton for the main action of a screen
    s.configure("TButton", background=SURFACE, bordercolor=BORDER, lightcolor=SURFACE,
                darkcolor=SURFACE, padding=(10, 4))
    s.map("TButton", background=[("disabled", BG), ("pressed", ACCENT_LIGHT), ("active", "#eef3fe")],
          foreground=[("disabled", "#a3acb6")], bordercolor=[("active", ACCENT)])
    s.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT,
                lightcolor=ACCENT, darkcolor=ACCENT, font=(family, 10, "bold"))
    s.map("Accent.TButton", background=[("disabled", "#a9c0f5"), ("pressed", ACCENT_HOVER), ("active", ACCENT_HOVER)],
          foreground=[("disabled", "white")], bordercolor=[("active", ACCENT_HOVER)])

    # Tabs: flat, the selected one white with accent text
    s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(10, 8, 10, 0))
    s.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(18, 7), borderwidth=0,
                font=(family, 10, "bold"), lightcolor=BG, bordercolor=BG)
    s.map("TNotebook.Tab", background=[("selected", SURFACE), ("active", "#e9edf2")],
          foreground=[("selected", ACCENT)], lightcolor=[("selected", SURFACE)])

    # Cards (KPI values)
    s.configure("Card.TFrame", background=SURFACE, bordercolor=BORDER, lightcolor=BORDER,
                darkcolor=BORDER, relief="solid", borderwidth=1)
    s.configure("CardTitle.TLabel", background=SURFACE, foreground=MUTED, font=(family, 8, "bold"))
    s.configure("CardValue.TLabel", background=SURFACE, foreground=TEXT, font=(family, 13, "bold"))

    # Tables, inputs, panes
    s.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, rowheight=22,
                bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE)
    s.map("Treeview", background=[("selected", ACCENT_LIGHT)], foreground=[("selected", TEXT)])
    s.configure("Treeview.Heading", background=BG, foreground=MUTED, font=(family, 9, "bold"),
                relief="flat", bordercolor=BORDER, lightcolor=BG, darkcolor=BG)
    s.map("Treeview.Heading", background=[("active", GRID)])
    for widget in ("TCombobox", "TSpinbox", "TEntry"):
        s.configure(widget, fieldbackground=SURFACE, background=SURFACE, arrowcolor=MUTED,
                    bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE, padding=3)
        s.map(widget, bordercolor=[("focus", ACCENT)], fieldbackground=[("readonly", SURFACE)])
    s.configure("TPanedwindow", background=BG)
    s.configure("Sash", sashthickness=8, gripcount=0, background=BG)
    s.configure("Video.TLabel", background=VIDEO_BG, foreground="#aab4bf", anchor="center")
    s.configure("Horizontal.TScale", background=ACCENT, troughcolor=GRID, bordercolor=BORDER,
                lightcolor=ACCENT, darkcolor=ACCENT, sliderthickness=14, gripcount=0)
    s.map("Horizontal.TScale", background=[("active", ACCENT_HOVER)])

    apply_plot_style()


def apply_plot_style():
    """matplotlib defaults for every figure: light, no top/right frame, muted axes, our colours."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BORDER,   # full light frame: every plot is a clearly separate panel
        "axes.linewidth": 1.0,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.titlecolor": TEXT,
        "axes.labelsize": 9,
        "axes.labelcolor": MUTED,
        "axes.prop_cycle": cycler(color=CYCLE),
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": True,     # white box: the legend stays readable over the signals
        "legend.framealpha": 0.9,
        "legend.edgecolor": GRID,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
    })


def no_data(ax, title, text="no data"):
    """Empty plot with a title and a grey note in the middle."""
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=MUTED)
