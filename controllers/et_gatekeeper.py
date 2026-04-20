import numpy as np
from utils.cbf_triggers import gamma_min_cbf


class ETGatekeeper:
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
        k_alpha=1.0,
        cbf_margin=0.10,
        gamma_off=0.2,     # hysteresis off threshold
        min_awake_time=0.2 # seconds
    ):
        self.model = model
        self.nominal = nominal_controller
        self.backup = backup_controller
        self.env = env
        self.dt = dt

        self.backup_horizon = backup_horizon
        self.nominal_horizon = nominal_horizon
        self.horizon_discount = horizon_discount

        self.k_alpha = k_alpha
        self.cbf_margin = cbf_margin
        self.gamma_off = gamma_off
        self.min_awake_steps = int(np.ceil(min_awake_time / dt))

        self.awake = False
        self.awake_count = 0

    def _candidate_valid(self, x0, goal_xy, Ts_steps):
        x = np.array(x0, dtype=float)

        # Nominal part
        for _ in range(Ts_steps):
            u = self.nominal.compute_control(x, goal_xy)
            x = self.model.step(x, u)
            if self.env.in_collision(x, margin=0.0):
                return False

        # Backup part
        Tb_steps = int(np.ceil(self.backup_horizon / self.dt))
        for _ in range(Tb_steps):
            u = self.backup.solve(
                x0=x,
                goal_xy=goal_xy,
                obstacles=self.env.obstacles,
                robot_radius=self.env.robot_radius,
                safe_margin=self.cbf_margin
            )
            x = self.model.step(x, u)
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

    def _event_trigger(self, x, u_nom):
        gamma_min, h_min, _ = gamma_min_cbf(
            x=x,
            u_nom=u_nom,
            obstacles=self.env.obstacles,
            robot_radius=self.env.robot_radius,
            margin=self.cbf_margin,
            k_alpha=self.k_alpha
        )
        return gamma_min, h_min

    def compute_control(self, x, goal_xy):
        u_nom = self.nominal.compute_control(x, goal_xy)
        gamma_min, h_min = self._event_trigger(x, u_nom)

        # Wake logic
        if not self.awake and gamma_min <= 0.0:
            self.awake = True
            self.awake_count = 0

        # Sleep logic with hysteresis + dwell
        if self.awake:
            self.awake_count += 1
            if self.awake_count >= self.min_awake_steps and gamma_min >= self.gamma_off:
                self.awake = False

        # Dormant => nominal only
        if not self.awake:
            info = {"mode": "nominal-dormant", "gamma_min": gamma_min, "h_min": h_min, "Ts_star": None}
            return u_nom, info

        # Awake => run gatekeeper search
        Ts_star = self._find_Ts_star(x, goal_xy)
        if Ts_star > 0:
            info = {"mode": "nominal-awake", "gamma_min": gamma_min, "h_min": h_min, "Ts_star": Ts_star}
            return u_nom, info

        u_b = self.backup.solve(
            x0=x,
            goal_xy=goal_xy,
            obstacles=self.env.obstacles,
            robot_radius=self.env.robot_radius,
            safe_margin=self.cbf_margin
        )
        info = {"mode": "backup", "gamma_min": gamma_min, "h_min": h_min, "Ts_star": 0}
        return u_b, info