import casadi as ca
import numpy as np


class BackupMPC:
    def __init__(self, dt, horizon=20, v_bounds=(-0.5, 1.0), w_bounds=(-2.0, 2.0)):
        self.dt = dt
        self.N = horizon
        self.v_min, self.v_max = v_bounds
        self.w_min, self.w_max = w_bounds

    def solve(self, x0, goal_xy, obstacles, robot_radius, safe_margin=0.15):
        nx, nu = 3, 2
        X = ca.SX.sym("X", nx, self.N + 1)
        U = ca.SX.sym("U", nu, self.N)

        x0_p = ca.SX.sym("x0", nx)
        gxy_p = ca.SX.sym("gxy", 2)

        w = []
        lbw, ubw = [], []
        g = []
        lbg, ubg = [], []
        J = 0

        # Decision vars
        for k in range(self.N + 1):
            w += [X[:, k]]
            lbw += [-ca.inf, -ca.inf, -ca.inf]
            ubw += [ca.inf, ca.inf, ca.inf]
        for k in range(self.N):
            w += [U[:, k]]
            lbw += [self.v_min, self.w_min]
            ubw += [self.v_max, self.w_max]

        # Initial condition
        g += [X[:, 0] - x0_p]
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

        # Dynamics + costs + obstacle constraints
        Q_pos, Q_th, R = 10.0, 0.1, 0.05
        for k in range(self.N):
            xk = X[:, k]
            uk = U[:, k]
            x_next = X[:, k + 1]

            f = ca.vertcat(
                uk[0] * ca.cos(xk[2]),
                uk[0] * ca.sin(xk[2]),
                uk[1]
            )
            g += [x_next - (xk + self.dt * f)]
            lbg += [0, 0, 0]
            ubg += [0, 0, 0]

            ex = xk[0] - gxy_p[0]
            ey = xk[1] - gxy_p[1]
            J += Q_pos * (ex * ex + ey * ey) + R * ca.dot(uk, uk)

            for ob in obstacles:
                dx = xk[0] - ob["cx"]
                dy = xk[1] - ob["cy"]
                min_r2 = (ob["r"] + robot_radius + safe_margin) ** 2
                g += [dx * dx + dy * dy - min_r2]
                lbg += [0.0]
                ubg += [ca.inf]

        # Terminal cost
        eN = X[:, self.N]
        J += Q_pos * ((eN[0] - gxy_p[0]) ** 2 + (eN[1] - gxy_p[1]) ** 2) + Q_th * (eN[2] ** 2)

        nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g), "p": ca.vertcat(x0_p, gxy_p)}
        solver = ca.nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": 0})

        w0 = np.zeros(sum(np.size(v) for v in w))
        p = np.array([x0[0], x0[1], x0[2], goal_xy[0], goal_xy[1]], dtype=float)

        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=p)
        z = np.array(sol["x"]).squeeze()

        # Extract first control
        offset_u = nx * (self.N + 1)
        u0 = z[offset_u: offset_u + nu]
        return np.array(u0, dtype=float)