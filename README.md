# ET-GK: Event-Triggered Gatekeeper (Unicycle + Static Obstacles)

Simulation code for a **unicycle** navigating a **2D cluttered environment** with:

- **Nominal controller:** Pure Pursuit (goal tracking; not obstacle-aware by design).
- **Backup controller:** Nonlinear MPC in **CasADi**, solved with **Ipopt** (obstacle avoidance via stage constraints).
- **Safety logic:** Gatekeeper-style switching between nominal and backup:
  - **Event-triggered (ET-GK):** CBF-style trigger can skip heavy $T_s^*$ search when "dormant."
  - **Time-triggered (TT-GK):** runs the gatekeeper $T_s^*$ search **every control step** (fair comparison target for ET-GK).

Use this repo to compare **compute time** and behavior across three algorithms on the **same scenario**.

---

## Requirements

- **Python** 3.10+ recommended (CasADi availability depends on OS/Python build).
- Dependencies (`requirements.txt`):
  - `numpy`
  - `matplotlib`
  - `casadi` (uses **Ipopt**; first run may print an Ipopt banner — normal).

---

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Run everything below from the repo root so imports like `from models.unicycle import Unicycle` work.

---

## Usage

### Run the simulation (one algorithm at a time)

```bash
python main_sim.py --algo backup
python main_sim.py --algo tt_gk
python main_sim.py --algo et_gk
```

| `--algo` | What it does |
|----------|--------------|
| `backup` | Baseline: applies only backup MPC each step (`backup.solve`). |
| `tt_gk` | Time-triggered gatekeeper: every step, searches for safe nominal horizon $T_s^*$; if none, uses backup MPC. Nominal = Pure Pursuit. |
| `et_gk` | Event-triggered gatekeeper: skips $T_s^*$ search when trigger says dormant; otherwise same search as TT. |

> Default if you omit `--algo` is `et_gk`.

---

## Metrics printed at the end

After the run, `main_sim.py` prints:

- **Algorithm name**
- **Success** (goal within `0.3` m)
- **Collision count** (increments on collision, then stops)
- **Min clearance** (minimum signed distance along the trajectory via `CircleObstacleField.min_signed_distance`)
- **Mean CPU per step** and **95th percentile CPU** (per-step control computation)
- **Backup usage** (fraction of steps with `mode == "backup"`; for `backup` this is `100%` by definition)

---

## Scenario configuration

Defaults live in `main_sim.py`:

- Start `x`, goal `goal`, obstacle list `obstacles` (`cx`, `cy`, `r`).
- Simulator `Unicycle` with timestep `dt`.
- Environment `CircleObstacleField` with `robot_radius`.

> **Plot limits:** `utils/plotting.py` (`draw_scene`) sets fixed `xlim` / `ylim`. If you move the scenario, update those limits so the view matches.

---

## Implementation notes (for fair comparisons)

1. **Backup horizon alignment** — `ETGatekeeper` and `TTGatekeeper` set effective backup length to `backup.N * dt` (matches the MPC horizon used in `BackupMPC`). The `backup_horizon=...` argument passed from `main_sim.py` is overridden if it does not match.
2. **Different speed limits** — The plant `Unicycle` and `BackupMPC` may use different `v_bounds`. Fine if intentional; state it in your report when comparing methods.
3. **Plot colors** — `draw_robot` colors `nominal-dormant`, `nominal-awake`, `backup`. TT nominal uses `nominal-tt`, which currently maps to the default color unless you extend the map in `plotting.py`.
4. **Benchmarking** — First Ipopt solve can be slower. For serious timing, use multiple runs or a short warm-up; you already report mean and p95.

---

## Repository layout

```
et-gk/
├── main_sim.py
├── requirements.txt
├── models/unicycle.py
├── controllers/
│   ├── pure_pursuit.py
│   ├── backup_mpc.py
│   ├── et_gatekeeper.py
│   └── tt_gatekeeper.py
└── utils/
    ├── environment.py
    ├── cbf_triggers.py
    └── plotting.py
```

---

## Troubleshooting

- **`ModuleNotFoundError`:** run from repo root or set `PYTHONPATH` to the repo root.
- **CasADi / Ipopt install:** try another Python version or conda if pip wheels fail.
- **Slow runs:** interactive plotting uses `plt.pause`; disable plotting for cleaner CPU benchmarks if needed.