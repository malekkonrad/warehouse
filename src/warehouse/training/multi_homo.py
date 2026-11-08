"""Homogeneous (parameter-sharing) multi-agent training on ``WarehouseMultiEnv``.

One algorithm, one shared policy — all agents use the same network. Each run
saves its model + TensorBoard logs to a separate directory so nothing is
overwritten.
"""

from __future__ import annotations

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from warehouse.envs.wrappers import make_shared_vec_env
from warehouse.training.configs import MULTI_ALGO_CONFIGS
from warehouse.paths import models_dir, logs_dir


def build_model(algo: str, variant: str, vec_env, log_dir: str):
    cfg = MULTI_ALGO_CONFIGS[algo][variant]
    if algo == "ppo":
        return PPO("MlpPolicy", vec_env, verbose=1,
                   tensorboard_log=log_dir, **cfg)
    if algo == "sac":
        return SAC("MlpPolicy", vec_env, verbose=1,
                   tensorboard_log=log_dir, **cfg)
    raise ValueError(f"Unknown algo={algo}")


def train_multi_homo(
    algo: str,
    variant: str = "v1",
    timesteps: int = 1_000_000,
    n_agents: int = 3,
    n_packages: int = 6,
    n_envs: int = 4,
    random_layout: bool = False,
) -> str:
    """Train a shared multi-agent policy. Returns the model directory."""
    if variant not in MULTI_ALGO_CONFIGS[algo]:
        raise SystemExit(
            f"No variant {variant} for algo={algo}. "
            f"Available: {list(MULTI_ALGO_CONFIGS[algo])}"
        )

    run_name = f"homo_{algo}_{variant}"
    if random_layout:
        run_name += "_random"
    models = models_dir(run_name)
    logs = logs_dir(run_name)

    train_vec = make_shared_vec_env(
        n_agents=n_agents, n_packages=n_packages,
        n_envs=n_envs, random_layout=random_layout,
    )
    eval_vec = make_shared_vec_env(
        n_agents=n_agents, n_packages=n_packages,
        n_envs=1, random_layout=random_layout,
    )

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=str(models),
        log_path=str(logs / "eval"),
        eval_freq=max(1, 20_000 // train_vec.num_envs),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 100_000 // train_vec.num_envs),
        save_path=str(models / "checkpoints"),
        name_prefix=run_name,
    )

    model = build_model(algo, variant, train_vec, str(logs))
    print(f"=== {run_name} | timesteps={timesteps:,} | "
          f"vec_size={train_vec.num_envs} ===")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_cb, ckpt_cb],
        tb_log_name=run_name,
    )
    model.save(str(models / "final_model"))
    print(f"Saved: {models}/final_model.zip")
    return str(models)
