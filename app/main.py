import tkinter as tk
from tkinter import ttk


class Gui:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Luge Start Tracker")
        self.window.geometry("1280x800")

        self.tabs = ttk.Notebook(self.window)
        self.tabs.pack(fill="both", expand=True)

        # Measurement tab: synchronised IMU + video signals (must have)
        self.measurement_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.measurement_tab, text="Measurement")
        ttk.Label(self.measurement_tab, text="Measurement: synchronised signals").pack(pady=20)

        # Analysis tab: load saved analyses, video player, pose estimation (optional)
        self.analysis_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.analysis_tab, text="Analysis")
        ttk.Label(self.analysis_tab, text="Analysis: saved measurements").pack(pady=20)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    Gui().run()
