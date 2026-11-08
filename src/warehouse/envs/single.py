from __future__ import annotations

import math
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional

ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "render", "assets"
)


@dataclass
class Package:
    pos: np.ndarray
    active: bool = True


# ─── Constants (section 17 of spec) ────────────────────────────────────────
MAP_SIZE = 10.0
R_ROBOT = 0.3
R_PICKUP = 0.5
R_DELIVERY = 0.7
N_PACKAGES = 3
N_OBSTACLES = 2
DT = 0.1
V_MAX = 2.0
A_MAX = 1.0
V_PICKUP_MAX = 0.5
MAX_STEPS = 500
N_SENSORS = 8
SENSOR_RANGE = 3.0
SENSOR_STEP = 0.1

R_PICKUP_BONUS = 20.0
R_DELIVERY_BONUS = 100.0
R_COMPLETION_BONUS = 200.0
R_STEP = -0.1
R_COLLISION = -5.0
BETA_PROGRESS = 1.0

# Fixed layout (section 17.5)
FIXED_ROBOT_START = np.array([1.0, 1.0], dtype=np.float32)
FIXED_BASE_POS = np.array([9.0, 9.0], dtype=np.float32)
FIXED_PACKAGES = [
    np.array([2.0, 8.0], dtype=np.float32),
    np.array([8.0, 2.0], dtype=np.float32),
    np.array([5.0, 1.5], dtype=np.float32),
]
# AABB obstacles: (x_min, y_min, x_max, y_max)
FIXED_OBSTACLES = [
    (3.0, 3.0, 4.0, 7.0),
    (6.0, 3.0, 7.0, 7.0),
]

# Pula 5 deterministycznych layoutów (Projekt 5).
# Każdy layout to lista pozycji 3 paczek; spełniają ograniczenia
# _is_valid_package_pos (margines od półek, robota startowego i bazy).
LAYOUT_POOL_5: list[list[np.ndarray]] = [
    [np.array(p, dtype=np.float32) for p in pkgs]
    for pkgs in [
        [(2.0, 8.0), (8.0, 2.0), (5.0, 1.5)],
        [(1.5, 5.0), (5.0, 8.5), (7.5, 8.5)],
        [(2.0, 7.0), (5.0, 5.0), (8.0, 3.0)],
        [(2.0, 3.0), (5.0, 8.5), (8.5, 5.0)],
        [(2.5, 8.0), (5.0, 2.0), (8.0, 7.5)],
    ]
]

# Sensor angles: 8 directions starting from 0 (right), step 45°
SENSOR_ANGLES = [i * np.pi / 4 for i in range(8)]


def _circle_rect_collision(cx: float, cy: float, r: float,
                            x_min: float, y_min: float,
                            x_max: float, y_max: float) -> bool:
    closest_x = np.clip(cx, x_min, x_max)
    closest_y = np.clip(cy, y_min, y_max)
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) < r * r


class WarehouseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None,
                 random_layout: bool = False,
                 layout_pool: list[list[np.ndarray]] | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.random_layout = random_layout
        self.layout_pool = layout_pool

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # 17-dim observation vector (section 18.2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(17,), dtype=np.float32
        )

        # Static config
        self.map_size = MAP_SIZE
        self.r_robot = R_ROBOT
        self.r_pickup = R_PICKUP
        self.r_delivery = R_DELIVERY
        self.n_packages_initial = N_PACKAGES
        self.dt = DT
        self.v_max = V_MAX
        self.a_max = A_MAX
        self.v_pickup_max = V_PICKUP_MAX
        self.max_steps = MAX_STEPS
        self.sensor_range = SENSOR_RANGE
        self.sensor_step = SENSOR_STEP

        self.r_pickup_bonus = R_PICKUP_BONUS
        self.r_delivery_bonus = R_DELIVERY_BONUS
        self.r_completion_bonus = R_COMPLETION_BONUS
        self.r_step = R_STEP
        self.r_collision = R_COLLISION
        self.beta_progress = BETA_PROGRESS

        self.base_pos = FIXED_BASE_POS.copy()
        self.obstacles = list(FIXED_OBSTACLES)

        # Mutable state (initialised in reset)
        self.pos = FIXED_ROBOT_START.copy()
        self.v = np.zeros(2, dtype=np.float32)
        self.carrying = False
        self.step_count = 0
        self.n_delivered = 0
        self.n_collisions_total = 0
        self.packages: list[Package] = [
            Package(pos=p.copy()) for p in FIXED_PACKAGES
        ]

        # Pygame surface
        self._screen = None
        self._clock = None
        self._WINDOW = 900
        self._scale = self._WINDOW / self.map_size
        self._sprites = None
        self._last_angle = math.pi / 2  # robot facing +y (up) by default

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _check_collision(self, pos: np.ndarray) -> bool:
        cx, cy = float(pos[0]), float(pos[1])
        r = self.r_robot
        # Map boundaries
        if cx - r < 0 or cx + r > self.map_size or cy - r < 0 or cy + r > self.map_size:
            return True
        # Obstacles
        for (x_min, y_min, x_max, y_max) in self.obstacles:
            if _circle_rect_collision(cx, cy, r, x_min, y_min, x_max, y_max):
                return True
        return False

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _get_current_target_position(self) -> np.ndarray:
        if self.carrying:
            return self.base_pos
        # Nearest active package
        best_dist = float("inf")
        best_pos = self.base_pos
        for pkg in self.packages:
            if pkg.active:
                d = self._distance(self.pos, pkg.pos)
                if d < best_dist:
                    best_dist = d
                    best_pos = pkg.pos
        return best_pos

    def _sensor_reading(self, angle: float) -> float:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        for d in np.arange(self.sensor_step, self.sensor_range, self.sensor_step):
            probe = self.pos + direction * d
            if self._check_collision(probe):
                return float(d / self.sensor_range)
        return 1.0

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(17, dtype=np.float32)
        # 0-1: position normalised to [-1, 1]
        obs[0] = self.pos[0] / self.map_size * 2.0 - 1.0
        obs[1] = self.pos[1] / self.map_size * 2.0 - 1.0
        # 2-3: velocity normalised
        obs[2] = self.v[0] / self.v_max
        obs[3] = self.v[1] / self.v_max
        # 4: carrying flag
        obs[4] = 1.0 if self.carrying else 0.0
        # 5-7: vector + distance to current target
        target = self._get_current_target_position()
        delta = target - self.pos
        obs[5] = delta[0] / self.map_size
        obs[6] = delta[1] / self.map_size
        obs[7] = float(np.linalg.norm(delta)) / (self.map_size * np.sqrt(2))
        # 8-15: sensors
        for i, angle in enumerate(SENSOR_ANGLES):
            obs[8 + i] = self._sensor_reading(angle)
        # 16: remaining packages
        n_left = sum(1 for p in self.packages if p.active)
        obs[16] = float(n_left) / float(self.n_packages_initial)
        return np.clip(obs, -1.0, 1.0)

    def _is_valid_package_pos(self, pos: np.ndarray,
                              others: list[np.ndarray],
                              min_sep: float = 1.0) -> bool:
        x, y = float(pos[0]), float(pos[1])
        margin = self.r_pickup + 0.1
        if x < margin or x > self.map_size - margin:
            return False
        if y < margin or y > self.map_size - margin:
            return False
        # Outside obstacles (inflated by margin so pickup zone doesn't sit in a shelf)
        for (xmn, ymn, xmx, ymx) in self.obstacles:
            if (xmn - margin <= x <= xmx + margin
                    and ymn - margin <= y <= ymx + margin):
                return False
        # Far from robot start and base
        if np.linalg.norm(pos - FIXED_ROBOT_START) < 1.0:
            return False
        if np.linalg.norm(pos - self.base_pos) < self.r_delivery + 0.5:
            return False
        # Far from already-placed packages
        for o in others:
            if np.linalg.norm(pos - o) < min_sep:
                return False
        return True

    def _sample_package_positions(self) -> list[np.ndarray]:
        placed: list[np.ndarray] = []
        for _ in range(self.n_packages_initial):
            for _attempt in range(200):
                p = self.np_random.uniform(
                    low=0.5, high=self.map_size - 0.5, size=2
                ).astype(np.float32)
                if self._is_valid_package_pos(p, placed):
                    placed.append(p)
                    break
            else:
                # Fallback: use the corresponding fixed slot
                placed.append(FIXED_PACKAGES[len(placed)].copy())
        return placed

    def _build_info(self, collision: bool, pickup: bool, delivery: bool) -> dict:
        n_left = sum(1 for p in self.packages if p.active)
        return {
            "collision": collision,
            "pickup": pickup,
            "delivery": delivery,
            "n_packages_left": n_left,
            "carrying": self.carrying,
            "n_collisions_total": self.n_collisions_total,
            "n_deliveries_total": self.n_delivered,
        }

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.pos = FIXED_ROBOT_START.copy()
        self.v = np.zeros(2, dtype=np.float32)
        self.carrying = False
        self.step_count = 0
        self.n_delivered = 0
        self.n_collisions_total = 0

        if self.layout_pool is not None:
            idx = int(self.np_random.integers(0, len(self.layout_pool)))
            positions = [p.copy() for p in self.layout_pool[idx]]
        elif self.random_layout:
            positions = self._sample_package_positions()
        else:
            positions = [p.copy() for p in FIXED_PACKAGES]
        self.packages = [Package(pos=p) for p in positions]
        self._last_angle = math.pi / 2

        obs = self._build_observation()
        info = self._build_info(False, False, False)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        a = action * self.a_max

        # Fix target at start of step
        target_pos_start = self._get_current_target_position()
        d_prev = self._distance(self.pos, target_pos_start)

        # Candidate motion
        v_intended = np.clip(self.v + a * self.dt, -self.v_max, self.v_max)
        pos_new = self.pos + v_intended * self.dt
        speed_intended = float(np.linalg.norm(v_intended))

        # Collision check
        collision = self._check_collision(pos_new)

        if collision:
            self.v = np.zeros(2, dtype=np.float32)
            self.n_collisions_total += 1
        else:
            self.pos = pos_new
            self.v = v_intended

        # Pickup – only when no collision this step (section 18.6)
        pickup = False
        if not collision and not self.carrying and speed_intended < self.v_pickup_max:
            for pkg in self.packages:
                if pkg.active and self._distance(self.pos, pkg.pos) < self.r_pickup:
                    pkg.active = False
                    self.carrying = True
                    pickup = True
                    break

        # Delivery
        delivery = False
        if not collision and self.carrying and speed_intended < self.v_pickup_max:
            if self._distance(self.pos, self.base_pos) < self.r_delivery:
                self.carrying = False
                self.n_delivered += 1
                delivery = True

        # Reward
        reward = self.r_step
        if collision:
            reward += self.r_collision
        if pickup:
            reward += self.r_pickup_bonus
        if delivery:
            reward += self.r_delivery_bonus
        if not (pickup or delivery):
            d_curr = self._distance(self.pos, target_pos_start)
            reward += self.beta_progress * (d_prev - d_curr)

        self.step_count += 1
        all_delivered = (self.n_delivered == self.n_packages_initial)
        if all_delivered:
            reward += self.r_completion_bonus

        terminated = all_delivered
        truncated = self.step_count >= self.max_steps

        obs = self._build_observation()
        info = self._build_info(collision, pickup, delivery)
        return obs, float(reward), terminated, truncated, info

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _load_sprites(self, pygame) -> None:
        if self._sprites is not None:
            return
        S = self._scale
        tile_px = max(8, int(round(S)))                       # 1 m tile
        robot_px = max(12, int(round(2 * self.r_robot * S * 1.6)))
        pkg_px = max(8, int(round(0.45 * S)))
        carry_px = max(6, int(round(pkg_px * 0.7)))

        def load_scaled(name: str, w: int, h: int):
            raw = pygame.image.load(os.path.join(ASSETS_DIR, name))
            return pygame.transform.smoothscale(raw, (w, h))

        self._sprites = {
            "robot": load_scaled("robot.png", robot_px, robot_px),
            "package": load_scaled("package.png", pkg_px, pkg_px),
            "package_small": load_scaled("package.png", carry_px, carry_px),
            "shelf_tile": load_scaled("shelf.png", tile_px, tile_px),
        }

    def render(self):
        if self.render_mode is None:
            return None
        try:
            import pygame
        except ImportError:
            return None

        W = self._WINDOW
        S = self._scale

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((W, W))
                pygame.display.set_caption("WarehouseRobotEnv")
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((W, W))

        self._load_sprites(pygame)
        sprites = self._sprites

        surface = self._screen
        surface.fill((38, 40, 46))

        def to_px(x, y):
            return int(x * S), int((self.map_size - y) * S)

        # Floor grid (1 m)
        grid_color = (52, 54, 60)
        for i in range(int(self.map_size) + 1):
            x_px = int(i * S)
            pygame.draw.line(surface, grid_color, (x_px, 0), (x_px, W), 1)
            pygame.draw.line(surface, grid_color, (0, x_px), (W, x_px), 1)

        # Shelves: tile the 1×1 sprite over each obstacle's footprint
        shelf = sprites["shelf_tile"]
        tile_px = shelf.get_width()
        for (x_min, y_min, x_max, y_max) in self.obstacles:
            nx = max(1, int(round(x_max - x_min)))
            ny = max(1, int(round(y_max - y_min)))
            for ix in range(nx):
                for iy in range(ny):
                    wx = x_min + ix
                    wy_top = y_min + iy + 1  # top edge of tile in world
                    px, py = to_px(wx, wy_top)
                    surface.blit(shelf, (px, py))

        # Base / delivery zone
        bx, by = to_px(self.base_pos[0], self.base_pos[1])
        zone_r = int(self.r_delivery * S)
        zone_surf = pygame.Surface((zone_r * 2 + 4, zone_r * 2 + 4),
                                   pygame.SRCALPHA)
        pygame.draw.circle(zone_surf, (60, 200, 90, 70),
                           (zone_r + 2, zone_r + 2), zone_r)
        pygame.draw.circle(zone_surf, (60, 200, 90, 220),
                           (zone_r + 2, zone_r + 2), zone_r, 3)
        surface.blit(zone_surf, (bx - zone_r - 2, by - zone_r - 2))
        # Crosshair / drop marker
        for dx, dy in [(-zone_r * 0.5, 0), (zone_r * 0.5, 0),
                       (0, -zone_r * 0.5), (0, zone_r * 0.5)]:
            pygame.draw.line(surface, (60, 200, 90),
                             (bx, by), (bx + int(dx), by + int(dy)), 2)
        pygame.draw.circle(surface, (30, 110, 50), (bx, by), 4)

        # Packages
        pkg_sprite = sprites["package"]
        for pkg in self.packages:
            if pkg.active:
                px, py = to_px(pkg.pos[0], pkg.pos[1])
                rect = pkg_sprite.get_rect(center=(px, py))
                surface.blit(pkg_sprite, rect)

        # Robot — orient to current velocity (keep last when stationary)
        speed = float(np.linalg.norm(self.v))
        if speed > 0.05:
            self._last_angle = math.atan2(float(self.v[1]), float(self.v[0]))
        # Sprite's "front" points image-up = world +y, i.e. angle = pi/2.
        rot_deg = math.degrees(self._last_angle - math.pi / 2)
        robot_img = pygame.transform.rotate(sprites["robot"], rot_deg)
        rx, ry = to_px(self.pos[0], self.pos[1])
        rrect = robot_img.get_rect(center=(rx, ry))
        surface.blit(robot_img, rrect)

        # Carrying indicator: small package riding on the robot
        if self.carrying:
            ps = sprites["package_small"]
            prect = ps.get_rect(center=(rx, ry))
            surface.blit(ps, prect)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            arr = pygame.surfarray.array3d(surface)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._screen = None
            self._sprites = None


# Quick smoke-test
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = WarehouseRobotEnv()
    print("Running check_env …")
    check_env(env, warn=True)
    print("check_env passed.")

    obs, info = env.reset(seed=0)
    print(f"obs shape: {obs.shape}, obs: {obs}")
    total_reward = 0.0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"20 random steps, total reward: {total_reward:.2f}, info: {info}")
    env.close()
