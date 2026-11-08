"""Longer single-agent training of one chosen config (e.g. the sweep winner).

Trains the given config for ``steps`` timesteps (default 1M) and saves the
model to ``models/single_best/best_long_<config>.zip``.
"""

from __future__ import annotations

import json
import time

import torch

from warehouse.envs import WarehouseRobotEnv, LAYOUT_POOL_5
from warehouse.training.configs import SINGLE_CONFIGS
from warehouse.paths import models_dir, logs_dir


def train_long(config: str, seed: int = 0, steps: int = 1_000_000) -> str:
    """Train one config for `steps` timesteps. Returns the saved model path."""
    torch.set_num_threads(4)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    out_root = models_dir("single_best")
    log_dir = logs_dir("single_best", config)

    raw_env = WarehouseRobotEnv(layout_pool=LAYOUT_POOL_5)
    env = Monitor(raw_env, filename=str(log_dir / "monitor.csv"))

    cfg = SINGLE_CONFIGS[config]
    model = PPO("MlpPolicy", env, seed=seed, verbose=1, **cfg)

    t0 = time.perf_counter()
    model.learn(total_timesteps=steps)
    dur = time.perf_counter() - t0

    out_path = out_root / f"best_long_{config}"
    model.save(str(out_path))

    with open(log_dir / "meta.json", "w") as f:
        json.dump({
            "config": config,
            "seed": seed,
            "total_timesteps": steps,
            "duration_s": dur,
            "model_path": str(out_path) + ".zip",
        }, f, indent=2)

    print(f"Saved {out_path}.zip  ({dur / 60:.1f} min)")
    return str(out_path) + ".zip"
