import numpy as np
from models.unicycle import wrap_angle


class PurePursuitGoal:
    def __init__(self, k_v=0.8, k_w=1.8, v_max=1.2, goal_tol=0.25):
        self.k_v = k_v
        self.k_w = k_w
        self.v_max = v_max
        self.goal_tol = goal_tol

    def compute_control(self, x, goal_xy):
        px, py, th = x
        gx, gy = goal_xy
        dx, dy = gx - px, gy - py
        dist = np.hypot(dx, dy)

        if dist < self.goal_tol:
            return np.array([0.0, 0.0])

        heading_des = np.arctan2(dy, dx)
        e_th = wrap_angle(heading_des - th)

        v = np.clip(self.k_v * dist, 0.0, self.v_max)
        w = self.k_w * e_th
        return np.array([v, w], dtype=float)