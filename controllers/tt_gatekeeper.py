import numpy as np


class TTGatekeeper:
    def __init__(
        self,
        model,
        nominal_controller,
        backup_controller,
        env,
        dt,
        backup_horizon=1.5,
        nominal_horizon=1.5,
        horizon_discount=0.1,
        cbf_margin=0.10,
    ):
        self.model = model
        self.nominal = nominal_controller
        self.backup = backup_controller
        self.env = env
        self.dt = dt

        self.backup_horizon = backup_horizon
        self._backup_horizon_from_mpc = self.backup.N * self.dt
        if abs(self.backup_horizon - self._backup_horizon_from_mpc) > 1e-9:
            self.backup_horizon = self._backup_horizon_from_mpc

        self.nominal_horizon = nominal_horizon
        self.horizon_discount = horizon_discount
        self.cbf_margin = cbf_margin

    def _candidate_valid(self, x0, goal_xy, Ts_steps):
        x = np.array(x0, dtype=float)

        # nominal rollout
        for _ in range(Ts_steps):
            u = self.nominal.compute_control(x, goal_xy)
            x = self.model.step(x, u)
            if self.env.in_collision(x, margin=0.0):
                return False

        # one backup solve, open-loop rollout
        try:
            U_seq = self.backup.solve_full_sequence(x, goal_xy)
        except Exception:
            return False

        Tb_steps = int(np.ceil(self.backup_horizon / self.dt))
        N_mpc = self.backup.N
        for k in range(Tb_steps):
            uk = U_seq[min(k, N_mpc - 1)]
            x = self.model.step(x, uk)
            if self.env.in_collision(x, margin=0.0):
                return False

        return True

    def _find_Ts_star(self, x, goal_xy):
        max_steps = int(np.ceil(self.nominal_horizon / self.dt))
        disc = max(1, int(np.ceil(self.horizon_discount / self.dt)))

        for Ts_steps in range(max_steps, -1, -disc):
            if self._candidate_valid(x, goal_xy, Ts_steps):
                return Ts_steps
        return 0

    def compute_control(self, x, goal_xy):
        # time-triggered => run Ts* search every step
        Ts_star = self._find_Ts_star(x, goal_xy)
        if Ts_star > 0:
            u = self.nominal.compute_control(x, goal_xy)
            info = {"mode": "nominal-tt", "Ts_star": Ts_star}
            return u, info

        u_b = self.backup.solve(x0=x, goal_xy=goal_xy)
        info = {"mode": "backup", "Ts_star": 0}
        return u_b, info