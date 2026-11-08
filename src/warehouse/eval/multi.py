"""Evaluate multi-agent policies: compute metrics and optionally render a GIF.

Modes:
  - homo:    one shared policy controls all agents
  - hetero:  N separate policies (one per agent), possibly different algorithms
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from warehouse.envs.multi import WarehouseMultiEnv


ALGO_CLS = {"ppo": PPO, "sac": SAC}


def load_model(algo: str, path: str):
    return ALGO_CLS[algo].load(path)


def run_episodes(
    ma_env: WarehouseMultiEnv,
    policies: dict[str, object],  # agent_id -> SB3 model
    n_episodes: int,
    deterministic: bool = True,
    record_first: bool = False,
):
    """Return a list of per-episode metric dicts plus optional RGB frames for
    the first episode (for a GIF)."""
    results = []
    frames: list[np.ndarray] = []
    for ep in range(n_episodes):
        obs, info = ma_env.reset(seed=ep)
        ep_reward_sum = 0.0
        ep_steps = 0
        ep_collisions = 0
        record_this = record_first and ep == 0
        if record_this and ma_env.render_mode is not None:
            f = ma_env.render()
            if f is not None:
                frames.append(f)
        done = False
        while not done:
            actions = {}
            for aid in ma_env.agents:
                a, _ = policies[aid].predict(obs[aid], deterministic=deterministic)
                actions[aid] = np.asarray(a, dtype=np.float32).reshape(2)
            obs, rew, term, trunc, info = ma_env.step(actions)
            ep_reward_sum += sum(rew.values()) / ma_env.n_agents  # mean team reward
            ep_steps += 1
            ep_collisions = next(iter(info.values()))["n_collisions_total"]
            if record_this and ma_env.render_mode is not None:
                f = ma_env.render()
                if f is not None:
                    frames.append(f)
            done = any(term.values()) or any(trunc.values())
        last_info = next(iter(info.values()))
        n_delivered = last_info["n_deliveries_total"]
        n_left = last_info["n_packages_left"]
        success = (n_left == 0)
        results.append({
            "episode": ep,
            "reward_mean": ep_reward_sum,
            "steps": ep_steps,
            "n_delivered": n_delivered,
            "n_collisions": ep_collisions,
            "success": bool(success),
        })
    return results, frames


def aggregate(results: list[dict]) -> dict:
    arr = lambda k: np.array([r[k] for r in results], dtype=np.float32)
    return {
        "n_episodes": len(results),
        "reward_mean": float(arr("reward_mean").mean()),
        "reward_std": float(arr("reward_mean").std()),
        "steps_mean": float(arr("steps").mean()),
        "delivered_mean": float(arr("n_delivered").mean()),
        "collisions_mean": float(arr("n_collisions").mean()),
        "success_rate": float(arr("success").mean()),
    }


def write_gif(frames: list[np.ndarray], path: str, fps: int = 20):
    if not frames:
        print("No frames to write — run with render mode rgb_array.")
        return
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
        iio.imwrite(path, frames, duration=int(1000 / fps), loop=0)
        print(f"GIF saved: {path}")
    except ImportError:
        # Fallback via PIL
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            path, save_all=True, append_images=imgs[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f"GIF saved (PIL): {path}")


def parse_hetero_spec(specs: list[str]) -> dict[str, tuple[str, str]]:
    """Parse a list of 'robot_X=algo:path' into {agent_id: (algo, path)}."""
    out = {}
    for s in specs:
        agent_part, rest = s.split("=", 1)
        algo, path = rest.split(":", 1)
        out[agent_part.strip()] = (algo.strip().lower(), path.strip())
    return out


def _report(agg: dict, results: list[dict], json_path: str | None):
    print(json.dumps(agg, indent=2))
    if json_path:
        Path(os.path.dirname(json_path) or ".").mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump({"agg": agg, "per_episode": results}, f, indent=2)
        print(f"Metrics saved: {json_path}")


def eval_homo(
    model: str,
    algo: str,
    episodes: int = 30,
    n_agents: int = 3,
    n_packages: int = 6,
    gif: str | None = None,
    json_path: str | None = None,
    fps: int = 20,
    stochastic: bool = False,
) -> dict:
    """Evaluate one shared policy across all agents."""
    policy = load_model(algo, model)
    ma_env = WarehouseMultiEnv(
        n_agents=n_agents, n_packages=n_packages,
        render_mode="rgb_array" if gif else None,
    )
    policies = {a: policy for a in ma_env.possible_agents}
    results, frames = run_episodes(
        ma_env, policies, episodes,
        deterministic=not stochastic, record_first=bool(gif),
    )
    agg = aggregate(results)
    _report(agg, results, json_path)
    if gif:
        write_gif(frames, gif, fps=fps)
    ma_env.close()
    return agg


def eval_hetero(
    specs: list[str],
    episodes: int = 30,
    n_agents: int = 3,
    n_packages: int = 6,
    gif: str | None = None,
    json_path: str | None = None,
    fps: int = 20,
    stochastic: bool = False,
) -> dict:
    """Evaluate N separate policies (a mixture of algorithms)."""
    parsed = parse_hetero_spec(specs)
    ma_env = WarehouseMultiEnv(
        n_agents=n_agents, n_packages=n_packages,
        render_mode="rgb_array" if gif else None,
    )
    missing = [a for a in ma_env.possible_agents if a not in parsed]
    if missing:
        raise SystemExit(f"Missing models for agents: {missing}")
    policies = {aid: load_model(algo, path) for aid, (algo, path) in parsed.items()}
    results, frames = run_episodes(
        ma_env, policies, episodes,
        deterministic=not stochastic, record_first=bool(gif),
    )
    agg = aggregate(results)
    _report(agg, results, json_path)
    if gif:
        write_gif(frames, gif, fps=fps)
    ma_env.close()
    return agg
