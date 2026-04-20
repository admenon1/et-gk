import matplotlib.pyplot as plt
import numpy as np


def draw_scene(ax, obstacles, goal_xy):
    ax.clear()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    for ob in obstacles:
        c = plt.Circle((ob["cx"], ob["cy"]), ob["r"], color="gray", alpha=0.45)
        ax.add_patch(c)

    ax.plot(goal_xy[0], goal_xy[1], "g*", markersize=14, label="Goal")
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)


def draw_robot(ax, x, mode="nominal-dormant"):
    color_map = {
        "nominal-dormant": "green",
        "nominal-awake": "orange",
        "backup": "red",
    }
    col = color_map.get(mode, "blue")
    ax.plot(x[0], x[1], "o", color=col, markersize=8)
    ax.arrow(
        x[0], x[1],
        0.4 * np.cos(x[2]), 0.4 * np.sin(x[2]),
        head_width=0.12, color=col, length_includes_head=True
    )