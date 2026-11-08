"""Wrappery łączące `WarehouseMultiEnv` (PettingZoo ParallelEnv) ze
Stable Baselines 3.

Dwa tryby:

1. ``make_shared_vec_env`` — parameter sharing (homogeneous training).
   Konwertuje ParallelEnv do gymnasium-podobnego VecEnv (każdy slot vec-a =
   jeden agent w środowisku). SB3 trenuje pojedynczą politykę, którą
   wszyscy agenci współdzielą. Najprostszy wariant CTDE z SB3.

2. ``SingleAgentSlotEnv`` — pojedynczy slot agenta w środowisku MA.
   Pozwala trenować różne polityki per-agent (heterogeneous): tworzymy N
   osobnych modeli SB3, ale dzielą ten sam egzemplarz `WarehouseMultiEnv`.
   Sterowanie pętlą treningu jest po stronie wywołującego (`train_multi_hetero.py`).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import supersuit as ss
from gymnasium import spaces

from .multi import WarehouseMultiEnv


def make_shared_vec_env(
    n_agents: int = 3,
    n_packages: int = 6,
    n_envs: int = 1,
    random_layout: bool = False,
    max_steps: int | None = None,
    base_class: str = "stable_baselines3",
):
    """Tworzy VecEnv dla SB3 z parameter-sharing (jedna polityka, wielu agentów).

    Po `pettingzoo_env_to_vec_env_v1` środowisko prezentuje się SB3 jako
    `n_agents` osobnych „środowisk" — każde dostarcza obserwację jednego
    agenta. `concat_vec_envs_v1` pozwala zrównoleglić to przez `n_envs`
    instancji PettingZoo (efektywny batch = n_envs * n_agents).
    """
    kwargs: dict[str, Any] = dict(
        n_agents=n_agents,
        n_packages=n_packages,
        random_layout=random_layout,
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps

    pz_env = WarehouseMultiEnv(**kwargs)
    pz_env = ss.black_death_v3(pz_env)  # padding gdy któryś agent skończy
    vec = ss.pettingzoo_env_to_vec_env_v1(pz_env)
    vec = ss.concat_vec_envs_v1(
        vec,
        num_vec_envs=n_envs,
        num_cpus=1,
        base_class=base_class,
    )
    return vec


class SingleAgentSlotEnv(gym.Env):
    """Widok pojedynczego agenta na współdzielony `WarehouseMultiEnv`.

    Używane przez heterogeniczny trening: tworzysz N takich wrapperów
    (po jednym na agenta), trzymasz wspólny `WarehouseMultiEnv`. Pętla
    treningu sama orkiestrowuje wywołania `step()` środowiska MA — ten
    wrapper jest tylko adapterem obserwacji / akcji do API gym.

    Uwaga: ten wrapper NIE wywołuje `reset()` / `step()` środowiska MA sam
    z siebie. Robi to nadrzędna pętla treningu.
    """

    metadata = {"render_modes": []}

    def __init__(self, ma_env: WarehouseMultiEnv, agent_id: str):
        super().__init__()
        if agent_id not in ma_env.possible_agents:
            raise ValueError(
                f"agent_id={agent_id} nie istnieje w {ma_env.possible_agents}"
            )
        self._ma = ma_env
        self.agent_id = agent_id
        self.observation_space = ma_env.observation_space(agent_id)
        self.action_space = ma_env.action_space(agent_id)
        # Bufor zapisywany przez pętlę nadrzędną
        self._last_obs: np.ndarray | None = None
        self._last_info: dict | None = None

    def set_step_result(self, obs: np.ndarray, info: dict):
        self._last_obs = obs
        self._last_info = info

    def reset(self, *, seed=None, options=None):
        if self._last_obs is None:
            raise RuntimeError(
                "SingleAgentSlotEnv.reset wywołany zanim pętla MA "
                "ustawiła pierwszą obserwację. Wywołaj ma_env.reset() i "
                "set_step_result() na każdym wrapperze najpierw."
            )
        return self._last_obs, (self._last_info or {})

    def step(self, action):  # noqa: D401
        raise RuntimeError(
            "SingleAgentSlotEnv.step nie jest wywoływany bezpośrednio — "
            "to pętla treningu MA wywołuje ma_env.step({agent: action}) "
            "i potem set_step_result() dla każdego slotu."
        )

    def render(self):
        return self._ma.render()

    def close(self):
        pass
