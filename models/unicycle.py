import numpy as np


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Unicycle:
    def __init__(self, dt: float, v_bounds=(-1.0, 1.5), w_bounds=(-2.0, 2.0)):
        self.dt = dt
        self.v_min, self.v_max = v_bounds
        self.w_min, self.w_max = w_bounds

    def clamp_u(self, u):
        v, w = float(u[0]), float(u[1])
        v = np.clip(v, self.v_min, self.v_max)
        w = np.clip(w, self.w_min, self.w_max)
        return np.array([v, w], dtype=float)

    def f(self, x, u):
        px, py, th = x
        v, w = self.clamp_u(u)
        return np.array([v * np.cos(th), v * np.sin(th), w], dtype=float)

    def step(self, x, u):
        x_next = np.array(x, dtype=float) + self.dt * self.f(x, u)
        x_next[2] = wrap_angle(x_next[2])
        return x_next