"""Single-agent PPO sweep: train every (config × seed) combination.

``run_one`` trains one PPO model for a given config + seed and writes
``logs/single_sweep/<config>/seed_<NN>/`` with:
  - monitor.csv  (per-episode reward, length, time)
  - final_model.zip
  - meta.json    (training time, steps, config)

``run_sweep`` fans these out across processes with ``multiprocessing.Pool``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time

import torch

from warehouse.envs import WarehouseRobotEnv, LAYOUT_POOL_5
from warehouse.training.configs import SINGLE_CONFIGS, DEFAULT_SWEEP_STEPS
from warehouse.paths import logs_dir

DEFAULT_SEEDS = list(range(10))


def run_one(config_name: str, seed: int,
            total_steps: int = DEFAULT_SWEEP_STEPS) -> str:
    """Train one PPO model. Returns the output directory."""
    # Keep torch single-threaded: we parallelise across processes, not within.
    torch.set_num_threads(1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    cfg = SINGLE_CONFIGS[config_name]
    out_dir = logs_dir("single_sweep", config_name, f"seed_{seed:02d}")

    raw_env = WarehouseRobotEnv(layout_pool=LAYOUT_POOL_5)
    env = Monitor(raw_env, filename=str(out_dir / "monitor.csv"))

    model = PPO("MlpPolicy", env, seed=seed, verbose=0, **cfg)

    start = time.perf_counter()
    model.learn(total_timesteps=total_steps)
    duration = time.perf_counter() - start

    model.save(str(out_dir / "final_model"))

    meta = {
        "config": config_name,
        "seed": seed,
        "total_timesteps": total_steps,
        "duration_s": duration,
        "steps_per_s": total_steps / duration if duration > 0 else None,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    env.close()
    return str(out_dir)


def _worker(args: tuple[str, int, int]) -> tuple[str, int, float]:
    config, seed, steps = args
    t0 = time.perf_counter()
    run_one(config, seed, steps)
    return config, seed, time.perf_counter() - t0


def run_sweep(
    configs: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int = DEFAULT_SWEEP_STEPS,
    parallel: int = 8,
) -> None:
    """Train every (config × seed) pair, ``parallel`` processes at a time."""
    configs = configs or list(SINGLE_CONFIGS)
    seeds = seeds or DEFAULT_SEEDS
    jobs = [(c, s, steps) for c in configs for s in seeds]
    print(f"Plan: {len(jobs)} runs × {steps:,} steps, "
          f"{parallel} processes in parallel.")

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=parallel) as pool:
        for i, (cfg, seed, dur) in enumerate(
            pool.imap_unordered(_worker, jobs), 1
        ):
            print(f"  [{i}/{len(jobs)}] {cfg} seed={seed:02d}  "
                  f"({dur / 60:.1f} min)")
    print(f"All done in {(time.perf_counter() - t0) / 60:.1f} min.")
