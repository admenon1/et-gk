import numpy as np


def alpha_linear(h, k_alpha=1.0):
    return k_alpha * h


def h_circle(x, ob, robot_radius, margin):
    px, py = x[0], x[1]
    R = ob["r"] + robot_radius + margin
    return (px - ob["cx"]) ** 2 + (py - ob["cy"]) ** 2 - R ** 2


def hdot_circle_unicycle(x, u_nom, ob):
    px, py, th = x
    v = float(u_nom[0])
    return 2.0 * (px - ob["cx"]) * v * np.cos(th) + 2.0 * (py - ob["cy"]) * v * np.sin(th)


def gamma_min_cbf(x, u_nom, obstacles, robot_radius, margin=0.0, k_alpha=1.0):
    gammas = []
    hs = []
    for ob in obstacles:
        h = h_circle(x, ob, robot_radius, margin)
        hdot = hdot_circle_unicycle(x, u_nom, ob)
        gamma = hdot + alpha_linear(h, k_alpha)
        gammas.append(gamma)
        hs.append(h)
    i_min = int(np.argmin(gammas))
    return float(gammas[i_min]), float(hs[i_min]), i_min