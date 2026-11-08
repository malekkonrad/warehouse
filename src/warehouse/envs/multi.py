"""Multi-agent wersja środowiska magazynowego (Projekt 6).

Zbudowane na PettingZoo `ParallelEnv` — wszyscy agenci działają równolegle
w jednym kroku (każdy podaje akcję, środowisko zwraca słowniki obs / reward /
term / trunc / info).

Środowisko single-agent (`WarehouseRobotEnv`) zostaje nietknięte — ten plik
dodaje równoległy tryb wieloagentowy, dzieląc tylko sensowne stałe i funkcje
pomocnicze.

Cechy:
- 3 roboty (parametryzowalne) o niezależnych akcjach (Box 2D),
- powiększona mapa 14×14 m z 6 półkami w trzech kolumnach (korytarze 2 m),
- 6 paczek do zebrania, jedna baza dostaw,
- shared reward (kooperacja): każdy agent dostaje sumę nagród zespołu,
- kolizje robot↔robot (jako dynamiczne przeszkody) + widoczność w sensorach
  i w obserwacji (pozycje względne + flagi carrying pozostałych).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PettingZoo nie jest zainstalowane. Zainstaluj przez `uv add pettingzoo`."
    ) from e

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "render", "assets")


# ─── Stałe środowiska MA (większa mapa, więcej obiektów) ─────────────────
MAP_SIZE = 14.0
R_ROBOT = 0.3
R_PICKUP = 0.5
R_DELIVERY = 0.7
N_AGENTS_DEFAULT = 3
N_PACKAGES_DEFAULT = 6
DT = 0.1
V_MAX = 2.0
A_MAX = 1.0
V_PICKUP_MAX = 0.5
MAX_STEPS = 800
N_SENSORS = 8
SENSOR_RANGE = 3.0
SENSOR_STEP = 0.1

# Nagrody (shared — sumowane i rozdawane wszystkim)
R_PICKUP_BONUS = 20.0
R_DELIVERY_BONUS = 100.0
R_COMPLETION_BONUS = 200.0
R_STEP = -0.1                     # per-agent koszt kroku
R_COLLISION_STATIC = -5.0         # uderzenie w półkę / ścianę (per-agent)
R_COLLISION_AGENT = -3.0          # uderzenie w innego robota (do obu)
BETA_PROGRESS = 1.0

# Layout: 3 starty (rogi/krawędzie), baza po przeciwnej stronie
FIXED_ROBOT_STARTS = [
    np.array([1.0, 1.0], dtype=np.float32),
    np.array([13.0, 1.0], dtype=np.float32),
    np.array([1.0, 13.0], dtype=np.float32),
]
FIXED_BASE_POS = np.array([13.0, 13.0], dtype=np.float32)

# 6 półek w 3 kolumnach × 2 segmenty — zostawia korytarze 2 m między kolumnami
FIXED_OBSTACLES = [
    (3.0, 3.0, 4.0, 6.0),
    (3.0, 8.0, 4.0, 11.0),
    (6.5, 3.0, 7.5, 6.0),
    (6.5, 8.0, 7.5, 11.0),
    (10.0, 3.0, 11.0, 6.0),
    (10.0, 8.0, 11.0, 11.0),
]

# 6 paczek w przejściach między półkami
FIXED_PACKAGES = [
    np.array([2.0, 7.0], dtype=np.float32),
    np.array([5.0, 5.0], dtype=np.float32),
    np.array([5.0, 12.0], dtype=np.float32),
    np.array([9.0, 2.0], dtype=np.float32),
    np.array([8.5, 9.5], dtype=np.float32),
    np.array([12.5, 5.0], dtype=np.float32),
]

# Kolory robotów do renderingu (RGB)
AGENT_COLORS = [
    (231, 76, 60),    # czerwony
    (52, 152, 219),   # niebieski
    (46, 204, 113),   # zielony
    (241, 196, 15),   # żółty
    (155, 89, 182),   # fioletowy
]

SENSOR_ANGLES = [i * np.pi / 4 for i in range(N_SENSORS)]


@dataclass
class Package:
    pos: np.ndarray
    active: bool = True


def _circle_rect_collision(cx: float, cy: float, r: float,
                            x_min: float, y_min: float,
                            x_max: float, y_max: float) -> bool:
    closest_x = np.clip(cx, x_min, x_max)
    closest_y = np.clip(cy, y_min, y_max)
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) < r * r


class WarehouseMultiEnv(ParallelEnv):
    """PettingZoo ParallelEnv — N kooperujących robotów w magazynie."""

    metadata = {
        "name": "warehouse_multi_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        n_agents: int = N_AGENTS_DEFAULT,
        n_packages: int = N_PACKAGES_DEFAULT,
        render_mode: Optional[str] = None,
        random_layout: bool = False,
        max_steps: int = MAX_STEPS,
    ):
        super().__init__()
        if n_agents < 1 or n_agents > len(FIXED_ROBOT_STARTS):
            raise ValueError(
                f"n_agents musi być w [1, {len(FIXED_ROBOT_STARTS)}], dostałem {n_agents}."
            )
        if n_packages < 1:
            raise ValueError("n_packages musi być >= 1.")

        self.n_agents = n_agents
        self.n_packages_initial = n_packages
        self.render_mode = render_mode
        self.random_layout = random_layout
        self.max_steps = max_steps

        # PettingZoo API
        self.possible_agents = [f"robot_{i}" for i in range(n_agents)]
        self.agents: list[str] = []

        # Stałe
        self.map_size = MAP_SIZE
        self.r_robot = R_ROBOT
        self.r_pickup = R_PICKUP
        self.r_delivery = R_DELIVERY
        self.dt = DT
        self.v_max = V_MAX
        self.a_max = A_MAX
        self.v_pickup_max = V_PICKUP_MAX
        self.sensor_range = SENSOR_RANGE
        self.sensor_step = SENSOR_STEP
        self.beta_progress = BETA_PROGRESS

        self.r_step = R_STEP
        self.r_pickup_bonus = R_PICKUP_BONUS
        self.r_delivery_bonus = R_DELIVERY_BONUS
        self.r_completion_bonus = R_COMPLETION_BONUS
        self.r_collision_static = R_COLLISION_STATIC
        self.r_collision_agent = R_COLLISION_AGENT

        self.base_pos = FIXED_BASE_POS.copy()
        self.obstacles = list(FIXED_OBSTACLES)

        # Wymiar obserwacji per agent:
        # 0-1  pos (xy, norm)
        # 2-3  vel (xy, norm)
        # 4    carrying flag
        # 5-7  delta+dist do targetu
        # 8-15 sensory (8)
        # 16   % paczek pozostałych
        # 17+  per inny agent: rel_x, rel_y, carrying  (3 × (n_agents-1))
        self._obs_dim = 17 + 3 * (n_agents - 1)

        # Wspólny obs/action space dla wszystkich agentów (CTDE/parameter sharing)
        self._obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32
        )
        self._act_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Stan (inicjalizowany w reset)
        self.pos = np.zeros((n_agents, 2), dtype=np.float32)
        self.v = np.zeros((n_agents, 2), dtype=np.float32)
        self.carrying = np.zeros(n_agents, dtype=bool)
        self.step_count = 0
        self.n_delivered = 0
        self.n_collisions_total = 0
        self.packages: list[Package] = []
        self._last_angle = [math.pi / 2] * n_agents

        # Render
        self._screen = None
        self._clock = None
        self._WINDOW = 900
        self._scale = self._WINDOW / self.map_size
        self._sprites: dict | None = None
        self._robot_imgs: list | None = None

        # PRNG
        self._np_random: np.random.Generator | None = None

    # ── PettingZoo API ───────────────────────────────────────────────────

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        self.agents = list(self.possible_agents)

        # Pozycje startowe robotów
        for i in range(self.n_agents):
            self.pos[i] = FIXED_ROBOT_STARTS[i].copy()
        self.v[:] = 0.0
        self.carrying[:] = False
        self._last_angle = [math.pi / 2] * self.n_agents

        # Paczki
        if self.random_layout:
            positions = self._sample_package_positions()
        else:
            # Wybiera pierwsze N z listy fixed; jeśli > 6, dosyła randomowe
            base = [p.copy() for p in FIXED_PACKAGES[: self.n_packages_initial]]
            while len(base) < self.n_packages_initial:
                extra = self._sample_package_positions()
                base.extend(extra[: self.n_packages_initial - len(base)])
                break
            positions = base
        self.packages = [Package(pos=p) for p in positions]

        self.step_count = 0
        self.n_delivered = 0
        self.n_collisions_total = 0

        obs = {a: self._build_observation(i) for i, a in enumerate(self.agents)}
        infos = {a: self._build_info(False, False, False) for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, np.ndarray]):
        # Akcje → przyspieszenia, kandydatowe pozycje
        a_arr = np.zeros((self.n_agents, 2), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            raw = np.asarray(actions[agent], dtype=np.float32).reshape(2)
            a_arr[i] = np.clip(raw, -1.0, 1.0) * self.a_max

        # Stary dystans do celu (na potrzeby progress reward)
        targets_start = [self._get_current_target_position(i)
                         for i in range(self.n_agents)]
        d_prev = np.array(
            [self._distance(self.pos[i], targets_start[i])
             for i in range(self.n_agents)],
            dtype=np.float32,
        )

        v_intended = np.clip(self.v + a_arr * self.dt, -self.v_max, self.v_max)
        pos_new = self.pos + v_intended * self.dt
        speed_intended = np.linalg.norm(v_intended, axis=1)

        # Kolizje statyczne (mapa + półki)
        static_collision = np.array(
            [self._check_static_collision(pos_new[i])
             for i in range(self.n_agents)],
            dtype=bool,
        )
        # Kolizje robot↔robot: para (i,j) zderza się, gdy ich docelowe okręgi
        # mają < 2*r_robot odległości. Pozycje statyczne (jeśli ktoś już stoi
        # bo miał kolizję statyczną) też brane pod uwagę.
        agent_collision = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            ci = pos_new[i] if not static_collision[i] else self.pos[i]
            for j in range(self.n_agents):
                if i == j:
                    continue
                cj = pos_new[j] if not static_collision[j] else self.pos[j]
                if np.linalg.norm(ci - cj) < 2.0 * self.r_robot:
                    agent_collision[i] = True
                    break

        # Aplikacja ruchu
        any_collision = static_collision | agent_collision
        for i in range(self.n_agents):
            if any_collision[i]:
                self.v[i] = 0.0
                self.n_collisions_total += 1
            else:
                self.pos[i] = pos_new[i]
                self.v[i] = v_intended[i]

        # Pickupy — losowa kolejność dla fairness
        pickup_flags = np.zeros(self.n_agents, dtype=bool)
        order = self._np_random.permutation(self.n_agents)
        for i in order:
            if any_collision[i] or self.carrying[i]:
                continue
            if speed_intended[i] >= self.v_pickup_max:
                continue
            for pkg in self.packages:
                if pkg.active and self._distance(self.pos[i], pkg.pos) < self.r_pickup:
                    pkg.active = False
                    self.carrying[i] = True
                    pickup_flags[i] = True
                    break

        # Delivery
        delivery_flags = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            if any_collision[i] or not self.carrying[i]:
                continue
            if speed_intended[i] >= self.v_pickup_max:
                continue
            if self._distance(self.pos[i], self.base_pos) < self.r_delivery:
                self.carrying[i] = False
                self.n_delivered += 1
                delivery_flags[i] = True

        # Nagroda: shared (każdy agent dostaje sumę zespołową)
        team_reward = 0.0
        per_agent_extra = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            # Koszt kroku — per agent ale i tak sumujemy do team
            team_reward += self.r_step
            if static_collision[i]:
                team_reward += self.r_collision_static
            if agent_collision[i]:
                team_reward += self.r_collision_agent
            if pickup_flags[i]:
                team_reward += self.r_pickup_bonus
            if delivery_flags[i]:
                team_reward += self.r_delivery_bonus
            # Progress: liczony tylko gdy nie było pickup/delivery w tym kroku
            if not (pickup_flags[i] or delivery_flags[i]):
                target_now = self._get_current_target_position(i)
                d_curr = self._distance(self.pos[i], target_now)
                per_agent_extra[i] += self.beta_progress * (d_prev[i] - d_curr)
        team_reward += float(per_agent_extra.sum())

        self.step_count += 1
        all_delivered = (self.n_delivered == self.n_packages_initial)
        if all_delivered:
            team_reward += self.r_completion_bonus

        terminated = bool(all_delivered)
        truncated = self.step_count >= self.max_steps

        # Shared reward → wszyscy dostają tę samą wartość
        rewards = {a: float(team_reward) for a in self.agents}
        terms = {a: terminated for a in self.agents}
        truncs = {a: truncated for a in self.agents}
        obs = {a: self._build_observation(i) for i, a in enumerate(self.agents)}
        infos = {
            a: self._build_info(
                bool(any_collision[i]),
                bool(pickup_flags[i]),
                bool(delivery_flags[i]),
            )
            for i, a in enumerate(self.agents)
        }

        if terminated or truncated:
            self.agents = []

        return obs, rewards, terms, truncs, infos

    # ── Pomocnicze ───────────────────────────────────────────────────────

    def _check_static_collision(self, pos: np.ndarray) -> bool:
        cx, cy = float(pos[0]), float(pos[1])
        r = self.r_robot
        if cx - r < 0 or cx + r > self.map_size:
            return True
        if cy - r < 0 or cy + r > self.map_size:
            return True
        for (x_min, y_min, x_max, y_max) in self.obstacles:
            if _circle_rect_collision(cx, cy, r, x_min, y_min, x_max, y_max):
                return True
        return False

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _get_current_target_position(self, idx: int) -> np.ndarray:
        if self.carrying[idx]:
            return self.base_pos
        best_dist = float("inf")
        best_pos = self.base_pos
        for pkg in self.packages:
            if pkg.active:
                d = self._distance(self.pos[idx], pkg.pos)
                if d < best_dist:
                    best_dist = d
                    best_pos = pkg.pos
        return best_pos

    def _sensor_reading(self, idx: int, angle: float) -> float:
        """Ray-march: sensor wykrywa półki, ścianę mapy oraz innych robotów."""
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        for d in np.arange(self.sensor_step, self.sensor_range, self.sensor_step):
            probe = self.pos[idx] + direction * d
            # Statyczne
            if self._check_static_collision(probe):
                return float(d / self.sensor_range)
            # Inni agenci
            for j in range(self.n_agents):
                if j == idx:
                    continue
                if np.linalg.norm(probe - self.pos[j]) < self.r_robot:
                    return float(d / self.sensor_range)
        return 1.0

    def _build_observation(self, idx: int) -> np.ndarray:
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[0] = self.pos[idx, 0] / self.map_size * 2.0 - 1.0
        obs[1] = self.pos[idx, 1] / self.map_size * 2.0 - 1.0
        obs[2] = self.v[idx, 0] / self.v_max
        obs[3] = self.v[idx, 1] / self.v_max
        obs[4] = 1.0 if self.carrying[idx] else 0.0
        target = self._get_current_target_position(idx)
        delta = target - self.pos[idx]
        obs[5] = delta[0] / self.map_size
        obs[6] = delta[1] / self.map_size
        obs[7] = float(np.linalg.norm(delta)) / (self.map_size * np.sqrt(2))
        for k, angle in enumerate(SENSOR_ANGLES):
            obs[8 + k] = self._sensor_reading(idx, angle)
        n_left = sum(1 for p in self.packages if p.active)
        obs[16] = float(n_left) / float(self.n_packages_initial)
        # Inni agenci
        offset = 17
        for j in range(self.n_agents):
            if j == idx:
                continue
            rel = self.pos[j] - self.pos[idx]
            obs[offset] = rel[0] / self.map_size
            obs[offset + 1] = rel[1] / self.map_size
            obs[offset + 2] = 1.0 if self.carrying[j] else 0.0
            offset += 3
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
        for (xmn, ymn, xmx, ymx) in self.obstacles:
            if (xmn - margin <= x <= xmx + margin
                    and ymn - margin <= y <= ymx + margin):
                return False
        for start in FIXED_ROBOT_STARTS[: self.n_agents]:
            if np.linalg.norm(pos - start) < 1.0:
                return False
        if np.linalg.norm(pos - self.base_pos) < self.r_delivery + 0.5:
            return False
        for o in others:
            if np.linalg.norm(pos - o) < min_sep:
                return False
        return True

    def _sample_package_positions(self) -> list[np.ndarray]:
        placed: list[np.ndarray] = []
        for k in range(self.n_packages_initial):
            for _attempt in range(300):
                p = self._np_random.uniform(
                    low=0.5, high=self.map_size - 0.5, size=2
                ).astype(np.float32)
                if self._is_valid_package_pos(p, placed):
                    placed.append(p)
                    break
            else:
                fallback = FIXED_PACKAGES[k % len(FIXED_PACKAGES)].copy()
                placed.append(fallback)
        return placed

    def _build_info(self, collision: bool, pickup: bool, delivery: bool) -> dict:
        n_left = sum(1 for p in self.packages if p.active)
        return {
            "collision": collision,
            "pickup": pickup,
            "delivery": delivery,
            "n_packages_left": n_left,
            "n_deliveries_total": self.n_delivered,
            "n_collisions_total": self.n_collisions_total,
        }

    # ── Render ───────────────────────────────────────────────────────────

    def _load_sprites(self, pygame) -> None:
        if self._sprites is not None:
            return
        S = self._scale
        tile_px = max(8, int(round(S)))
        robot_px = max(12, int(round(2 * self.r_robot * S * 1.6)))
        pkg_px = max(8, int(round(0.45 * S)))
        carry_px = max(6, int(round(pkg_px * 0.7)))

        def load_scaled(name: str, w: int, h: int):
            raw = pygame.image.load(os.path.join(ASSETS_DIR, name))
            return pygame.transform.smoothscale(raw, (w, h))

        base_robot = load_scaled("robot.png", robot_px, robot_px)
        self._sprites = {
            "robot": base_robot,
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
                pygame.display.set_caption("WarehouseMultiEnv")
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((W, W))

        self._load_sprites(pygame)
        sprites = self._sprites
        surface = self._screen
        surface.fill((38, 40, 46))

        def to_px(x, y):
            return int(x * S), int((self.map_size - y) * S)

        # Siatka
        grid_color = (52, 54, 60)
        for i in range(int(self.map_size) + 1):
            xp = int(i * S)
            pygame.draw.line(surface, grid_color, (xp, 0), (xp, W), 1)
            pygame.draw.line(surface, grid_color, (0, xp), (W, xp), 1)

        # Półki
        shelf = sprites["shelf_tile"]
        for (x_min, y_min, x_max, y_max) in self.obstacles:
            nx = max(1, int(round(x_max - x_min)))
            ny = max(1, int(round(y_max - y_min)))
            for ix in range(nx):
                for iy in range(ny):
                    wx = x_min + ix
                    wy_top = y_min + iy + 1
                    px, py = to_px(wx, wy_top)
                    surface.blit(shelf, (px, py))

        # Baza dostaw
        bx, by = to_px(self.base_pos[0], self.base_pos[1])
        zone_r = int(self.r_delivery * S)
        zone_surf = pygame.Surface(
            (zone_r * 2 + 4, zone_r * 2 + 4), pygame.SRCALPHA
        )
        pygame.draw.circle(zone_surf, (60, 200, 90, 70),
                           (zone_r + 2, zone_r + 2), zone_r)
        pygame.draw.circle(zone_surf, (60, 200, 90, 220),
                           (zone_r + 2, zone_r + 2), zone_r, 3)
        surface.blit(zone_surf, (bx - zone_r - 2, by - zone_r - 2))
        pygame.draw.circle(surface, (30, 110, 50), (bx, by), 4)

        # Paczki
        pkg_sprite = sprites["package"]
        for pkg in self.packages:
            if pkg.active:
                px, py = to_px(pkg.pos[0], pkg.pos[1])
                rect = pkg_sprite.get_rect(center=(px, py))
                surface.blit(pkg_sprite, rect)

        # Roboty (każdy z kolorowym ringiem)
        for i in range(self.n_agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            speed = float(np.linalg.norm(self.v[i]))
            if speed > 0.05:
                self._last_angle[i] = math.atan2(
                    float(self.v[i, 1]), float(self.v[i, 0])
                )
            rot_deg = math.degrees(self._last_angle[i] - math.pi / 2)
            robot_img = pygame.transform.rotate(sprites["robot"], rot_deg)
            rx, ry = to_px(self.pos[i, 0], self.pos[i, 1])
            rrect = robot_img.get_rect(center=(rx, ry))
            # Kolorowy ring oznaczający agenta
            ring_r = int(self.r_robot * S * 1.4)
            pygame.draw.circle(surface, color, (rx, ry), ring_r, 3)
            surface.blit(robot_img, rrect)
            if self.carrying[i]:
                ps = sprites["package_small"]
                prect = ps.get_rect(center=(rx, ry))
                surface.blit(ps, prect)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
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


def env(**kwargs) -> WarehouseMultiEnv:
    """Factory zgodny ze stylem PettingZoo."""
    return WarehouseMultiEnv(**kwargs)


# ── Smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    e = WarehouseMultiEnv(n_agents=3, n_packages=6)
    print(f"possible_agents: {e.possible_agents}")
    print(f"obs dim: {e._obs_dim}")
    obs, infos = e.reset(seed=0)
    print(f"reset obs keys: {list(obs.keys())}, shape: {obs['robot_0'].shape}")
    total = 0.0
    for t in range(50):
        actions = {a: e.action_space(a).sample() for a in e.agents}
        obs, rew, term, trunc, infos = e.step(actions)
        total += sum(rew.values()) / e.n_agents
        if all(term.values()) or all(trunc.values()):
            print(f"stopped at t={t}, terminated={term}, truncated={trunc}")
            break
    print(f"50 random steps, mean team reward (per-agent avg): {total:.2f}")
    print(f"info robot_0: {infos['robot_0']}")
    e.close()
