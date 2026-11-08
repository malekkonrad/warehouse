"""Command-line interface for the warehouse RL project.

    warehouse train single   --layout random --steps 2_500_000
    warehouse train multi homo   --algo ppo --variant v1
    warehouse train multi hetero --rounds 5 --round-steps 200_000
    warehouse train long     --config big_lr3e4
    warehouse sweep          --parallel 8
    warehouse play           --model models/random_v2/best_model.zip
    warehouse eval homo      --model models/homo_ppo_v1/final_model.zip --algo ppo
    warehouse eval hetero    --model robot_0=ppo:... --model robot_1=ppo:... --model robot_2=sac:...
    warehouse assets

Heavy imports (torch, SB3) are deferred into each command so that ``--help``
stays fast.
"""

from __future__ import annotations

from typing import List, Optional

import typer

app = typer.Typer(
    help="Warehouse robot RL: train, evaluate and visualise agents.",
    no_args_is_help=True,
    add_completion=False,
)
train_app = typer.Typer(help="Train agents.", no_args_is_help=True)
multi_app = typer.Typer(help="Multi-agent training.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate trained multi-agent policies.",
                       no_args_is_help=True)
app.add_typer(train_app, name="train")
train_app.add_typer(multi_app, name="multi")
app.add_typer(eval_app, name="eval")


# ─── train single ──────────────────────────────────────────────────────────
@train_app.command("single")
def train_single_cmd(
    run_name: str = typer.Option("ppo_single", help="Output dir name under models/ and logs/."),
    layout: str = typer.Option("fixed", help="'fixed' or 'random' map layout."),
    steps: int = typer.Option(1_000_000, help="Total training timesteps."),
    n_envs: int = typer.Option(4, help="Parallel environments."),
):
    """Train a single-agent PPO policy on WarehouseRobotEnv."""
    from warehouse.training import train_single
    train_single(
        run_name=run_name,
        random_layout=(layout == "random"),
        total_timesteps=steps,
        n_envs=n_envs,
    )


@train_app.command("long")
def train_long_cmd(
    config: str = typer.Option(..., help="Config name (see configs.SINGLE_CONFIGS)."),
    seed: int = typer.Option(0),
    steps: int = typer.Option(1_000_000),
):
    """Longer training of one chosen config (e.g. the sweep winner)."""
    from warehouse.training import train_long
    train_long(config=config, seed=seed, steps=steps)


# ─── train multi ───────────────────────────────────────────────────────────
@multi_app.command("homo")
def train_multi_homo_cmd(
    algo: str = typer.Option(..., help="'ppo' or 'sac'."),
    variant: str = typer.Option("v1"),
    timesteps: int = typer.Option(1_000_000),
    n_agents: int = typer.Option(3),
    n_packages: int = typer.Option(6),
    n_envs: int = typer.Option(4, help="Parallel PettingZoo instances."),
    random_layout: bool = typer.Option(False),
):
    """Homogeneous training: one shared policy for all agents."""
    from warehouse.training import train_multi_homo
    train_multi_homo(
        algo=algo, variant=variant, timesteps=timesteps,
        n_agents=n_agents, n_packages=n_packages, n_envs=n_envs,
        random_layout=random_layout,
    )


@multi_app.command("hetero")
def train_multi_hetero_cmd(
    rounds: int = typer.Option(5, help="Round-robin rounds (one per agent each)."),
    round_steps: int = typer.Option(200_000, help="Timesteps per learner per round."),
    n_packages: int = typer.Option(6),
):
    """Heterogeneous training: 3 policies / 3 algorithms via round-robin self-play."""
    from warehouse.training import train_multi_hetero
    train_multi_hetero(rounds=rounds, round_steps=round_steps, n_packages=n_packages)


# ─── sweep ─────────────────────────────────────────────────────────────────
@app.command("sweep")
def sweep_cmd(
    configs: Optional[List[str]] = typer.Option(None, help="Configs to run (default: all)."),
    seeds: Optional[List[int]] = typer.Option(None, help="Seeds to run (default: 0-9)."),
    steps: int = typer.Option(200_000, help="Timesteps per run."),
    parallel: int = typer.Option(8, help="Concurrent processes."),
):
    """Single-agent sweep over (config × seed)."""
    from warehouse.training import run_sweep
    run_sweep(configs=configs, seeds=seeds, steps=steps, parallel=parallel)


# ─── play ──────────────────────────────────────────────────────────────────
@app.command("play")
def play_cmd(
    model: str = typer.Option(..., help="Path to a trained single-agent model .zip."),
    episodes: int = typer.Option(5),
    random_layout: bool = typer.Option(True),
):
    """Visualise a trained single-agent policy in a pygame window."""
    from warehouse.render import play
    play(model_path=model, episodes=episodes, random_layout=random_layout)


# ─── eval ──────────────────────────────────────────────────────────────────
@eval_app.command("homo")
def eval_homo_cmd(
    model: str = typer.Option(..., help="Path to the shared policy .zip."),
    algo: str = typer.Option(..., help="'ppo' or 'sac'."),
    episodes: int = typer.Option(30),
    n_agents: int = typer.Option(3),
    n_packages: int = typer.Option(6),
    gif: Optional[str] = typer.Option(None, help="Write a GIF of the first episode."),
    json_path: Optional[str] = typer.Option(None, "--json", help="Write metrics JSON."),
    fps: int = typer.Option(20),
    stochastic: bool = typer.Option(False),
):
    """Evaluate a shared multi-agent policy."""
    from warehouse.eval import eval_homo
    eval_homo(
        model=model, algo=algo, episodes=episodes, n_agents=n_agents,
        n_packages=n_packages, gif=gif, json_path=json_path, fps=fps,
        stochastic=stochastic,
    )


@eval_app.command("hetero")
def eval_hetero_cmd(
    model: List[str] = typer.Option(
        ..., help="Repeat per agent: 'robot_0=ppo:path.zip'."),
    episodes: int = typer.Option(30),
    n_agents: int = typer.Option(3),
    n_packages: int = typer.Option(6),
    gif: Optional[str] = typer.Option(None),
    json_path: Optional[str] = typer.Option(None, "--json"),
    fps: int = typer.Option(20),
    stochastic: bool = typer.Option(False),
):
    """Evaluate a heterogeneous line-up (mixture of algorithms)."""
    from warehouse.eval import eval_hetero
    eval_hetero(
        specs=model, episodes=episodes, n_agents=n_agents,
        n_packages=n_packages, gif=gif, json_path=json_path, fps=fps,
        stochastic=stochastic,
    )


# ─── assets ────────────────────────────────────────────────────────────────
@app.command("assets")
def assets_cmd():
    """(Re)generate the PNG sprites used by the renderer."""
    from warehouse.render import make_assets
    make_assets.make_robot()
    make_assets.make_package()
    make_assets.make_shelf()
    print("Assets written to:", make_assets.ASSETS_DIR)


if __name__ == "__main__":
    app()
