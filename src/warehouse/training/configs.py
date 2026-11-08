"""All training hyperparameter presets in one place.

Previously these were scattered across three modules (``train_runs``,
``train_multi_homo``, ``train_multi_hetero``). Keeping them together makes the
CLI ``--config`` / ``--variant`` choices discoverable and avoids drift.
"""

from __future__ import annotations

# ─── Single-agent PPO presets (used by `sweep` and `train single`) ──────────
SINGLE_CONFIGS: dict[str, dict] = {
    "small_lr3e4": dict(
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    ),
    "big_lr3e4": dict(
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    ),
    "small_lr1e4": dict(
        learning_rate=1e-4,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    ),
}

DEFAULT_SWEEP_STEPS = 200_000

# ─── Multi-agent homogeneous presets (one shared policy) ────────────────────
MULTI_ALGO_CONFIGS: dict[str, dict[str, dict]] = {
    "ppo": {
        "v1": dict(learning_rate=3e-4, n_steps=2048, batch_size=64,
                   gamma=0.99, gae_lambda=0.95, ent_coef=0.0),
        "v2": dict(learning_rate=1e-4, n_steps=4096, batch_size=128,
                   gamma=0.995, gae_lambda=0.98, ent_coef=0.01),
    },
    "sac": {
        "v1": dict(learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                   tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                   learning_starts=5_000),
    },
}

# ─── Multi-agent heterogeneous line-up (one model + algo per agent) ─────────
# Used by round-robin self-play: three different algorithms cooperate inside a
# single episode.
HETERO_AGENTS: list[tuple[str, str, dict]] = [
    ("robot_0", "ppo",
     dict(learning_rate=3e-4, n_steps=2048, batch_size=64,
          gamma=0.99, gae_lambda=0.95, ent_coef=0.0)),
    ("robot_1", "ppo",
     dict(learning_rate=1e-4, n_steps=4096, batch_size=128,
          gamma=0.995, gae_lambda=0.98, ent_coef=0.01)),
    ("robot_2", "sac",
     dict(learning_rate=3e-4, buffer_size=200_000, batch_size=256,
          tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
          learning_starts=2_000)),
]
