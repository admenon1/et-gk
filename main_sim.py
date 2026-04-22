import time
import numpy as np
import matplotlib.pyplot as plt

from models.unicycle import Unicycle
from controllers.pure_pursuit import PurePursuitGoal
from controllers.backup_mpc import BackupMPC
from controllers.et_gatekeeper import ETGatekeeper
from utils.environment import CircleObstacleField
from utils.plotting import draw_scene, draw_robot


def main():
    dt = 0.05
    T = 25.0
    steps = int(T / dt)

    x = np.array([0.0, 0.0, 0.0])
    goal = np.array([10.0, 10.0])

    obstacles = [
        {"cx": 3.0, "cy": 2.0, "r": 0.9},
        {"cx": 5.0, "cy": 4.0, "r": 1.1},
        {"cx": 6.8, "cy": 6.4, "r": 0.8},
        {"cx": 8.0, "cy": 5.0, "r": 0.9},
        {"cx": 4.0, "cy": 8.0, "r": 1.0},
    ]

    model = Unicycle(dt=dt, v_bounds=(0.2, 1.2), w_bounds=(-2.0, 2.0))
    env = CircleObstacleField(obstacles=obstacles, robot_radius=0.25)
    nominal = PurePursuitGoal(k_v=0.9, k_w=2.0, v_max=1.0, goal_tol=0.3)
    backup = BackupMPC(
        dt=dt,
        horizon=20,
        v_bounds=(0.2, 0.9),
        w_bounds=(-2.0, 2.0),
        obstacles=obstacles,
        robot_radius=env.robot_radius,
        safe_margin=0.15,  # match cbf_margin / your report
    )

    gk = ETGatekeeper(
        model=model,
        nominal_controller=nominal,
        backup_controller=backup,
        env=env,
        dt=dt,
        backup_horizon=1.5,
        nominal_horizon=1.5,
        horizon_discount=0.1,
        k_alpha=1.0,
        cbf_margin=0.15,
        gamma_off=0.2,
        min_awake_time=0.25,
    )

    traj = [x.copy()]
    modes = []
    cpu_ms = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    for k in range(steps):
        t0 = time.perf_counter()
        u, info = gk.compute_control(x, goal)
        cpu_ms.append((time.perf_counter() - t0) * 1e3)

        x = model.step(x, u)
        traj.append(x.copy())
        modes.append(info["mode"])

        if env.in_collision(x):
            print(f"[STOP] Collision at step {k}")
            break

        if np.hypot(x[0] - goal[0], x[1] - goal[1]) < 0.3:
            print(f"[DONE] Reached goal at step {k}")
            break

        draw_scene(ax, obstacles, goal)
        tr = np.array(traj)
        ax.plot(tr[:, 0], tr[:, 1], "k-", alpha=0.8, linewidth=1.5)
        draw_robot(ax, x, mode=info["mode"])
        ax.set_title(f"Mode={info['mode']} | gamma={info['gamma_min']:.3f} | CPU={cpu_ms[-1]:.2f} ms")
        plt.pause(0.001)

    plt.ioff()
    plt.show()

    cpu_ms = np.array(cpu_ms)
    if len(cpu_ms) > 0:
        print(f"Mean CPU per step: {cpu_ms.mean():.3f} ms")
        print(f"95th percentile CPU: {np.percentile(cpu_ms, 95):.3f} ms")
        print(f"Backup usage: {100.0 * np.mean(np.array(modes) == 'backup'):.1f}% steps")


if __name__ == "__main__":
    main()