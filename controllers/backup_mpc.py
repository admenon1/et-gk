import casadi as ca
import numpy as np


class BackupMPC:
    def __init__(
        self,
        dt,
        horizon=20,
        v_bounds=(-0.5, 1.0),
        w_bounds=(-2.0, 2.0),
        obstacles=None,
        robot_radius=0.25,
        safe_margin=0.15,
    ):
        self.dt = dt
        self.N = int(horizon)
        self.v_min, self.v_max = v_bounds
        self.w_min, self.w_max = w_bounds
        self.obstacles = obstacles if obstacles is not None else []
        self.robot_radius = float(robot_radius)
        self.safe_margin = float(safe_margin)

        self.nx, self.nu = 3, 2
        self._nw = self.nx * (self.N + 1) + self.nu * self.N + 1  # + s_term
        self._offset_u = self.nx * (self.N + 1)

        self.w_pos = 8.0
        self.w_u = 0.05
        self.w_heading = 1.5
        self.w_dv = 2.0
        self.w_dw = 0.5
        self.w_terminal = 40.0
        self.w_slack = 1e4

        # terminal sdf-like margin for closest obstacle (soft via slack)
        self.terminal_margin = self.safe_margin
        self._u_prev = np.array([0.0, 0.0], dtype=float)

        self._build_solver()

    def _build_solver(self):
        N, nx, nu = self.N, self.nx, self.nu
        X = ca.SX.sym("X", nx, N + 1)
        U = ca.SX.sym("U", nu, N)
        x0_p = ca.SX.sym("x0", nx)
        gxy_p = ca.SX.sym("gxy", 2)
        u_prev_p = ca.SX.sym("u_prev", self.nu)  # [v_prev, w_prev]

        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []
        J = 0
        def wrap_angle_sym(a):
            return ca.atan2(ca.sin(a), ca.cos(a))

        for k in range(N + 1):
            w += [X[:, k]]
            lbw += [-ca.inf] * nx
            ubw += [ca.inf] * nx
        for k in range(N):
            w += [U[:, k]]
            lbw += [self.v_min, self.w_min]
            ubw += [self.v_max, self.w_max]

        s_term = ca.SX.sym("s_term")
        w += [s_term]
        lbw += [0.0]
        ubw += [ca.inf]

        g += [X[:, 0] - x0_p]
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

        Q_pos, Q_th, R = 10.0, 0.1, 0.05
        for k in range(N):
            xk, uk, x_next = X[:, k], U[:, k], X[:, k + 1]
            f = ca.vertcat(
                uk[0] * ca.cos(xk[2]),
                uk[0] * ca.sin(xk[2]),
                uk[1],
            )
            g += [x_next - (xk + self.dt * f)]
            lbg += [0, 0, 0]
            ubg += [0, 0, 0]

            # Distance-to-goal
            ex = xk[0] - gxy_p[0]
            ey = xk[1] - gxy_p[1]
            dist_cost = self.w_pos * (ex * ex + ey * ey)

            # Control effort
            u_cost = self.w_u * ca.dot(uk, uk)

            # Heading-to-goal penalty
            goal_bearing = ca.atan2(gxy_p[1] - xk[1], gxy_p[0] - xk[0])
            e_heading = wrap_angle_sym(xk[2] - goal_bearing)
            heading_cost = self.w_heading * (e_heading * e_heading)

            # Input rate penalty
            if k == 0:
                du = uk - u_prev_p
            else:
                du = uk - U[:, k - 1]
            rate_cost = self.w_dv * (du[0] ** 2) + self.w_dw * (du[1] ** 2)

            J += dist_cost + u_cost + heading_cost + rate_cost

            for ob in self.obstacles:
                dx = xk[0] - float(ob["cx"])
                dy = xk[1] - float(ob["cy"])
                min_r2 = (float(ob["r"]) + self.robot_radius + self.safe_margin) ** 2
                g += [dx * dx + dy * dy - min_r2]
                lbg += [0.0]
                ubg += [ca.inf]

        eN = X[:, N]
        exN = eN[0] - gxy_p[0]
        eyN = eN[1] - gxy_p[1]
        J += self.w_terminal * (exN * exN + eyN * eyN)

        if len(self.obstacles) > 0:
            for ob in self.obstacles:
                dxN = eN[0] - float(ob["cx"])
                dyN = eN[1] - float(ob["cy"])
                min_r2 = (float(ob["r"]) + self.robot_radius + self.terminal_margin) ** 2
                g += [dxN * dxN + dyN * dyN - min_r2 + s_term]
                lbg += [0.0]
                ubg += [ca.inf]

        J += self.w_slack * (s_term ** 2)

        nlp = {
            "x": ca.vertcat(*w),
            "f": J,
            "g": ca.vertcat(*g),
            "p": ca.vertcat(x0_p, gxy_p, u_prev_p),
        }
        name = f"backup_mpc_{id(self)}"
        self._solver = ca.nlpsol(
            name,
            "ipopt",
            nlp,
            {"ipopt.print_level": 0, "print_time": 0},
        )
        self._lbw = np.array(lbw, dtype=float).flatten()
        self._ubw = np.array(ubw, dtype=float).flatten()
        self._lbg = np.array(lbg, dtype=float).flatten()
        self._ubg = np.array(ubg, dtype=float).flatten()
        self._w0 = np.zeros(self._nw)

    def _optimize(self, x0, goal_xy, u_prev):
        p = np.array(
            [x0[0], x0[1], x0[2], goal_xy[0], goal_xy[1], u_prev[0], u_prev[1]],
            dtype=float,
        )
        w0 = np.zeros(self._nw)  # fresh start each solve (robust for varying x0 during Ts search)
        sol = self._solver(
            x0=w0,
            lbx=self._lbw,
            ubx=self._ubw,
            lbg=self._lbg,
            ubg=self._ubg,
            p=p,
        )
        z = np.array(sol["x"]).squeeze()
        U_flat = z[self._offset_u : self._offset_u + self.nu * self.N]
        U = U_flat.reshape(self.N, self.nu)
        return U

    def solve_full_sequence(self, x0, goal_xy):
        """Full open-loop horizon from one NLP solve; shape (N, nu)."""
        return self._optimize(np.asarray(x0, dtype=float).flatten(), goal_xy, self._u_prev)

    def solve(self, x0, goal_xy):
        """First control only (same as before for callers)."""
        U = self._optimize(np.asarray(x0, dtype=float).flatten(), goal_xy, self._u_prev)
        u0 = np.array(U[0], dtype=float).copy()
        self._u_prev = u0
        return u0