"""Train a single-agent PPO policy on ``WarehouseRobotEnv`` (spec section 17.7).

Each run writes its artifacts under ``models/<run_name>/`` and
``logs/<run_name>/`` so previous runs are never overwritten.
"""

from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from warehouse.envs import WarehouseRobotEnv
from warehouse.paths import models_dir, logs_dir


def train_single(
    run_name: str = "ppo_single",
    random_layout: bool = False,
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
) -> str:
    """Train PPO and return the directory holding the saved models."""
    models = models_dir(run_name)
    logs = logs_dir(run_name)

    def _make_env():
        return WarehouseRobotEnv(random_layout=random_layout)

    vec_env = make_vec_env(_make_env, n_envs=n_envs)
    eval_env = make_vec_env(_make_env, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models),
        log_path=str(logs / "eval"),
        eval_freq=10_000,
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(models / "checkpoints"),
        name_prefix="ppo_warehouse",
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log=str(logs),
    )

    print(
        f"Starting PPO run '{run_name}' "
        f"(random_layout={random_layout}) for {total_timesteps:,} timesteps …"
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="ppo_warehouse",
    )

    model.save(str(models / "final_model"))
    print(f"Training done. Best/final model saved under {models}")
    return str(models)
