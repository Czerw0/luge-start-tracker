import tkinter as tk
from ipykernel import gui
import matplotlib.pyplot as plt

class Gui:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Ludge Start Tracker")
        self.window.geometry("1920x1080")

        self.label = tk.Label(self.window, text="Ludge Start Tracker ", font=("Arial", 24))
        self.label.pack(pady=20)

        self.window.mainloop()


if __name__ == "__main__":
    gui = Gui()