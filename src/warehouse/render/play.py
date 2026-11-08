"""Visualise a trained PPO agent playing ``WarehouseRobotEnv`` in a pygame window."""

from __future__ import annotations

import time

from stable_baselines3 import PPO

from warehouse.envs import WarehouseRobotEnv


def play(model_path: str, episodes: int = 5, random_layout: bool = True) -> None:
    """Open a pygame window and run the policy for `episodes` episodes."""
    import pygame

    env = WarehouseRobotEnv(render_mode="human", random_layout=random_layout)
    model = PPO.load(model_path, env=env)

    for ep in range(episodes):
        obs, info = env.reset()
        env.render()  # initialise pygame before the first event.get()
        total_reward = 0.0
        step = 0
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            env.render()

        print(
            f"Episode {ep + 1}: steps={step}, reward={total_reward:.1f}, "
            f"delivered={info['n_deliveries_total']}, "
            f"collisions={info['n_collisions_total']}"
        )
        time.sleep(1.0)

    env.close()
