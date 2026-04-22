import time
import numpy as np
import matplotlib.pyplot as plt

from models.unicycle import Unicycle
from controllers.pure_pursuit import PurePursuitGoal
from controllers.backup_mpc import BackupMPC
from controllers.et_gatekeeper import ETGatekeeper
from utils.environment import CircleObstacleField
from utils.plotting import draw_scene, draw_robot

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="et_gk",
                   choices=["backup", "tt_gk", "et_gk"])
    return p.parse_args()


def main():
    args = parse_args()

    dt = 0.05
    T = 50.0
    steps = int(T / dt)

    x = np.array([0.0, 0.0, 0.0])
    goal = np.array([14.0, 14.0])

    obstacles = [
        {"cx": 4.2, "cy": 3.1, "r": 0.65},
        {"cx": 7.8, "cy": 6.4, "r": 0.72},
        {"cx": 10.6, "cy": 8.9, "r": 0.58},
        {"cx": 13.4, "cy": 11.2, "r": 0.69},
        {"cx": 16.1, "cy": 14.8, "r": 0.63},
    ]

    model = Unicycle(dt=dt, v_bounds=(0.0, 1.2), w_bounds=(-2.0, 2.0))
    env = CircleObstacleField(obstacles=obstacles, robot_radius=0.25)
    nominal = PurePursuitGoal(k_v=0.9, k_w=2.0, v_max=1.0, goal_tol=0.3)
    backup = BackupMPC(
        dt=dt,
        horizon=40,
        v_bounds=(0.0, 0.9),
        w_bounds=(-2.0, 2.0),
        obstacles=obstacles,
        robot_radius=env.robot_radius,
        safe_margin=0.15,  # match cbf_margin / your report
    )

    if args.algo == "backup":
        controller = None  # handled directly in loop
    elif args.algo == "tt_gk":
        from controllers.tt_gatekeeper import TTGatekeeper
        controller = TTGatekeeper(
            model=model,
            nominal_controller=nominal,
            backup_controller=backup,
            env=env,
            dt=dt,
            backup_horizon=1.5,
            nominal_horizon=1.5,
            horizon_discount=0.1,
            cbf_margin=0.15,
        )
    else:
        controller = ETGatekeeper(
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
    success = False
    collision_count = 0
    min_clearance = float("inf")

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    for k in range(steps):
        t0 = time.perf_counter()
        
        if args.algo == "backup":
            u = backup.solve(x0=x, goal_xy=goal)
            info = {"mode": "backup", "Ts_star": None}
        else:
            u, info = controller.compute_control(x, goal)

        cpu_ms.append((time.perf_counter() - t0) * 1e3)

        x = model.step(x, u)
        traj.append(x.copy())
        modes.append(info["mode"])

        dmin, _ = env.min_signed_distance(x)
        min_clearance = min(min_clearance, dmin)

        if env.in_collision(x):
            collision_count += 1
            print(f"[STOP] Collision at step {k}")
            break

        if np.hypot(x[0] - goal[0], x[1] - goal[1]) < 0.3:
            success = True
            print(f"[DONE] Reached goal at step {k}")
            break

        draw_scene(ax, obstacles, goal)
        tr = np.array(traj)
        ax.plot(tr[:, 0], tr[:, 1], "k-", alpha=0.8, linewidth=1.5)
        draw_robot(ax, x, mode=info["mode"])
        gamma_txt = f"{info['gamma_min']:.3f}" if "gamma_min" in info else "NA"
        ax.set_title(f"Algo={args.algo} | Mode={info['mode']} | gamma={gamma_txt} | CPU={cpu_ms[-1]:.2f} ms")
        plt.pause(0.001)

    plt.ioff()
    plt.show()

    cpu_ms = np.array(cpu_ms)
    mean_cpu = float(cpu_ms.mean()) if len(cpu_ms) else float("nan")
    p95_cpu = float(np.percentile(cpu_ms, 95)) if len(cpu_ms) else float("nan")

    if args.algo in ["tt_gk", "et_gk"]:
        backup_usage = 100.0 * np.mean(np.array(modes) == "backup") if len(modes) else 0.0
    else:
        backup_usage = 100.0  # backup-only baseline

    print(f"Algorithm: {args.algo}")
    print(f"Success: {success}")
    print(f"Collision count: {collision_count}")
    print(f"Min clearance: {min_clearance:.3f} m")
    print(f"Mean CPU per step: {mean_cpu:.3f} ms")
    print(f"95th percentile CPU: {p95_cpu:.3f} ms")
    print(f"Backup usage: {backup_usage:.1f}%")


if __name__ == "__main__":
    main()