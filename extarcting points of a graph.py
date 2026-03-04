from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import simpledialog

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Text

"""
Manual graph digitizer (pixel sampling)

- Opens article1/image.png
- Left-click to add a point (records pixel coordinates)
- Right-click to undo last point
- Press 's' to save points to CSV
- Press 'c' to calibrate to axis values (then save data-values)
- Press 'q' or ESC to quit

Output CSV columns:
- If calibrated: index, x, y
- Otherwise: index, x_px, y_px
"""


@dataclass
class Point:
    x_px: float
    y_px: float


@dataclass
class LinearMap:
    a: float
    b: float

    def apply(self, px: float) -> float:
        return self.a * px + self.b

    @staticmethod
    def from_two_points(px1: float, v1: float, px2: float, v2: float) -> "LinearMap":
        if px2 == px1:
            raise ValueError("Calibration points must not have identical pixel coordinate")
        a = (v2 - v1) / (px2 - px1)
        b = v1 - a * px1
        return LinearMap(a=a, b=b)


class ClickSampler:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.points: List[Point] = []
        self._scat: Optional[PathCollection] = None
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        self._status_text: Optional[Text] = None
        self._x_map: Optional[LinearMap] = None
        self._y_map: Optional[LinearMap] = None
        self._calibration_clicks: List[Point] = []
        self._is_calibrating: bool = False

    def run(self) -> None:
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        img = mpimg.imread(self.image_path)

        fig, ax = plt.subplots()
        self._fig = fig
        self._ax = ax

        ax.set_title(
            "Left-click: add | Right-click: undo | 'c': calibrate | 's': save | 'q'/ESC: quit"
        )
        ax.imshow(img)
        ax.set_axis_off()

        # Scatter for points + status line
        self._scat = ax.scatter([], [], s=25)
        self._status_text = ax.text(
            0.01,
            0.99,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        self._update_plot()

        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        plt.show()

    def _on_click(self, event) -> None:
        if self._ax is None:
            return
        if event.inaxes != self._ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if self._is_calibrating and event.button == 1:
            self._calibration_clicks.append(Point(float(event.xdata), float(event.ydata)))
            if len(self._calibration_clicks) >= 4:
                self._finish_calibration()
            else:
                self._update_plot()
            return

        # Left click: add
        if event.button == 1:
            self.points.append(Point(float(event.xdata), float(event.ydata)))
            self._update_plot()
            return

        # Right click: undo
        if event.button == 3 and self.points:
            self.points.pop()
            self._update_plot()
            return

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()

        if self._fig is None:
            return

        if key in {"q", "escape"}:
            plt.close(self._fig)
            return

        if key == "c":
            self._start_calibration()
            return

        if key == "s":
            out_csv = self.image_path.with_name("samples.csv")
            self.save_csv(out_csv)
            self._update_plot(saved_to=out_csv)
            return

        if key in {"backspace", "delete"} and self.points:
            self.points.pop()
            self._update_plot()
            return

    def _update_plot(self, saved_to: Optional[Path] = None) -> None:
        if self._scat is None or self._status_text is None:
            return

        xs = [p.x_px for p in self.points]
        ys = [p.y_px for p in self.points]
        if self.points:
            offsets = np.column_stack([xs, ys])
        else:
            offsets = np.empty((0, 2))
        self._scat.set_offsets(offsets)

        msg = f"Points: {len(self.points)}"
        if self._is_calibrating:
            msg += f" | Calibrating: click {len(self._calibration_clicks) + 1}/4"
        else:
            msg += " | Calibrated" if self.is_calibrated else " | Not calibrated (press 'c')"

        if self.points:
            last = self.points[-1]
            if self.is_calibrated:
                x, y = self.px_to_data(last)
                msg += f" | Last: (x={x:.6g}, y={y:.6g})"
            else:
                msg += f" | Last px: ({last.x_px:.1f}, {last.y_px:.1f})"
        if saved_to is not None:
            msg += f" | Saved: {saved_to.name}"
        self._status_text.set_text(msg)

        if self._fig is not None:
            self._fig.canvas.draw_idle()

    def save_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if self.is_calibrated:
                w.writerow(["index", "x", "y"])
                for i, p in enumerate(self.points):
                    x, y = self.px_to_data(p)
                    w.writerow([i, x, y])
            else:
                w.writerow(["index", "x_px", "y_px"])
                for i, p in enumerate(self.points):
                    w.writerow([i, p.x_px, p.y_px])

    @property
    def is_calibrated(self) -> bool:
        return self._x_map is not None and self._y_map is not None

    def px_to_data(self, p: Point) -> tuple[float, float]:
        if not self.is_calibrated:
            raise RuntimeError("Not calibrated. Press 'c' and define axis mapping first.")
        assert self._x_map is not None
        assert self._y_map is not None
        return self._x_map.apply(p.x_px), self._y_map.apply(p.y_px)

    def _start_calibration(self) -> None:
        self._calibration_clicks = []
        self._is_calibrating = True
        self._update_plot()

    def _prompt_float(self, title: str, prompt: str) -> Optional[float]:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            value = simpledialog.askfloat(title, prompt, parent=root)
        finally:
            root.destroy()
        return value

    def _finish_calibration(self) -> None:
        self._is_calibrating = False
        if len(self._calibration_clicks) < 4:
            self._update_plot()
            return

        x1_px = self._calibration_clicks[0].x_px
        x2_px = self._calibration_clicks[1].x_px
        y1_px = self._calibration_clicks[2].y_px
        y2_px = self._calibration_clicks[3].y_px

        x1 = self._prompt_float("Calibrate X", "Enter the X value at the 1st clicked point")
        if x1 is None:
            self._update_plot()
            return
        x2 = self._prompt_float("Calibrate X", "Enter the X value at the 2nd clicked point")
        if x2 is None:
            self._update_plot()
            return

        y1 = self._prompt_float("Calibrate Y", "Enter the Y value at the 3rd clicked point")
        if y1 is None:
            self._update_plot()
            return
        y2 = self._prompt_float("Calibrate Y", "Enter the Y value at the 4th clicked point")
        if y2 is None:
            self._update_plot()
            return

        try:
            self._x_map = LinearMap.from_two_points(x1_px, x1, x2_px, x2)
            self._y_map = LinearMap.from_two_points(y1_px, y1, y2_px, y2)
        except Exception:
            self._x_map = None
            self._y_map = None
            raise

        self._calibration_clicks = []
        self._update_plot()


def main() -> None:
    # This script is located at: article1/extarcting points of a graph.py
    # Image is at: article1/image.png
    script_dir = Path(__file__).resolve().parent
    image_path = script_dir / "2D/figures/front_surface - Front Surface zF vs r.png"

    sampler = ClickSampler(image_path=image_path)
    sampler.run()


if __name__ == "__main__":
    main()