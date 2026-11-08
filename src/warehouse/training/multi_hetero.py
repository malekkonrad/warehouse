"""Heterogeneous multi-agent training: 3 policies, 3 algorithms, one environment.

Strategy: round-robin self-play. Each of the 3 agents owns its own SB3 model
(PPO-v1, PPO-v2, SAC). In each round one agent learns while the others act as
*frozen* policies — so during every training episode three different algorithms
cooperate inside the environment. After a round we switch the learner and
repeat.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from warehouse.envs.multi import WarehouseMultiEnv
from warehouse.training.configs import HETERO_AGENTS
from warehouse.paths import models_dir, logs_dir


class FrozenOthersEnv(gym.Env):
    """Single-agent view of ``WarehouseMultiEnv``: one agent is the 'learner',
    the others are driven by frozen policies passed in the constructor.

    To SB3 this looks like a plain single-agent gym env, so ``model.learn()``
    works unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int,
        n_packages: int,
        learner_id: str,
        frozen_policies: dict[str, Any],
    ):
        super().__init__()
        self.ma_env = WarehouseMultiEnv(
            n_agents=n_agents, n_packages=n_packages
        )
        if learner_id not in self.ma_env.possible_agents:
            raise ValueError(
                f"learner_id={learner_id} not in "
                f"{self.ma_env.possible_agents}"
            )
        self.learner_id = learner_id
        self.frozen = frozen_policies  # {agent_id: SB3 model}
        self.observation_space = self.ma_env.observation_space(learner_id)
        self.action_space = self.ma_env.action_space(learner_id)
        self._last_obs_dict: dict | None = None

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self.ma_env.reset(seed=seed, options=options)
        self._last_obs_dict = obs_dict
        return obs_dict[self.learner_id], info_dict[self.learner_id]

    def step(self, action):
        actions = {self.learner_id: np.asarray(action, dtype=np.float32)}
        for aid in self.ma_env.possible_agents:
            if aid == self.learner_id:
                continue
            obs_i = self._last_obs_dict[aid]
            if aid in self.frozen:
                a_i, _ = self.frozen[aid].predict(obs_i, deterministic=False)
            else:
                a_i = self.ma_env.action_space(aid).sample()
            actions[aid] = np.asarray(a_i, dtype=np.float32)

        obs_d, rew_d, term_d, trunc_d, info_d = self.ma_env.step(actions)
        self._last_obs_dict = obs_d
        return (
            obs_d[self.learner_id],
            float(rew_d[self.learner_id]),
            bool(term_d[self.learner_id]),
            bool(trunc_d[self.learner_id]),
            info_d[self.learner_id],
        )

    def close(self):
        self.ma_env.close()


def build_model(algo: str, params: dict, env, log_dir: str):
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=0,
                   tensorboard_log=log_dir, **params)
    if algo == "sac":
        return SAC("MlpPolicy", env, verbose=0,
                   tensorboard_log=log_dir, **params)
    raise ValueError(f"Unknown algo={algo}")


def train_multi_hetero(
    rounds: int = 5,
    round_steps: int = 200_000,
    n_packages: int = 6,
) -> str:
    """Round-robin self-play training. Returns the model directory."""
    n_agents = len(HETERO_AGENTS)
    run_name = "hetero_ppo-v1_ppo-v2_sac"
    models = models_dir(run_name)
    logs = logs_dir(run_name)

    # Init one model per agent, each with a throwaway env to register spaces.
    # Models are later re-bound to different envs via set_env() per round.
    agent_models: dict[str, Any] = {}
    algos: dict[str, str] = {}
    for aid, algo, params in HETERO_AGENTS:
        boot_env = DummyVecEnv([
            lambda aid=aid: FrozenOthersEnv(
                n_agents=n_agents, n_packages=n_packages,
                learner_id=aid, frozen_policies={}
            )
        ])
        agent_models[aid] = build_model(algo, params, boot_env, str(logs))
        algos[aid] = algo

    # Round-robin: rounds × n_agents separate training sessions.
    for r in range(rounds):
        for learner_id, _algo, _ in HETERO_AGENTS:
            frozen = {
                aid: agent_models[aid] for aid in agent_models
                if aid != learner_id
            }
            # Monitor fills ep_info_buffer so SB3 logs rollout/ep_rew_mean
            # (reward curve in TensorBoard). Without it hetero logs only had
            # train/* tags and the plot was empty.
            train_env = DummyVecEnv([
                lambda lid=learner_id, fr=frozen: Monitor(FrozenOthersEnv(
                    n_agents=n_agents, n_packages=n_packages,
                    learner_id=lid, frozen_policies=fr,
                ))
            ])
            agent_models[learner_id].set_env(train_env)
            tb_name = f"{run_name}/round{r}_{learner_id}_{algos[learner_id]}"
            print(f"=== round {r+1}/{rounds} | learner={learner_id} "
                  f"({algos[learner_id]}) | steps={round_steps:,} ===")
            agent_models[learner_id].learn(
                total_timesteps=round_steps,
                reset_num_timesteps=False,
                tb_log_name=tb_name,
            )

    # Save final policies.
    for aid, model in agent_models.items():
        path = models / f"{aid}_{algos[aid]}_final"
        model.save(str(path))
        print(f"Saved: {path}.zip")
    return str(models)
