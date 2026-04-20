import numpy as np


class CircleObstacleField:
    def __init__(self, obstacles, robot_radius=0.25):
        """
        obstacles: list of dicts [{"cx":..., "cy":..., "r":...}, ...]
        """
        self.obstacles = obstacles
        self.robot_radius = robot_radius

    def min_signed_distance(self, x):
        px, py = x[0], x[1]
        dmin = np.inf
        idx = -1
        for i, ob in enumerate(self.obstacles):
            d = np.hypot(px - ob["cx"], py - ob["cy"]) - (ob["r"] + self.robot_radius)
            if d < dmin:
                dmin, idx = d, i
        return dmin, idx

    def in_collision(self, x, margin=0.0):
        dmin, _ = self.min_signed_distance(x)
        return dmin <= margin