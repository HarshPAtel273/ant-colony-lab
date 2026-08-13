"""The environment: pheromone map, food sources, obstacles, and the colony."""

import numpy as np

from .pathfield import distance_field

# While walls are being painted, refresh the distance fields at most once
# every N ticks so the frame rate stays smooth.
FIELD_REFRESH_TICKS = 10


class FoodSource:
    __slots__ = ("x", "y", "amount", "initial")

    def __init__(self, x, y, amount):
        self.x = float(x)
        self.y = float(y)
        self.amount = int(amount)
        self.initial = int(amount)


class World:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.pheromone = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        self.obstacles = np.zeros((cfg.height, cfg.width), dtype=bool)
        self.visited = np.zeros((cfg.height, cfg.width), dtype=bool)
        self.colony = np.array([cfg.width / 2.0, cfg.height / 2.0], dtype=np.float32)
        self.food_sources = []
        self._place_obstacles()
        self._place_food()

        # Pathfinding distance fields, computed lazily and invalidated when
        # walls or food change.
        self.tick = 0
        self._home_field = None
        self._food_field = None
        self._home_dirty = True
        self._food_dirty = True
        self._home_tick = -FIELD_REFRESH_TICKS
        self._food_tick = -FIELD_REFRESH_TICKS

    # ------------------------------------------------------------------ setup

    def _place_obstacles(self):
        """A few random walls, kept away from the colony."""
        cfg, rng = self.cfg, self.rng
        cx, cy = self.colony
        for _ in range(cfg.n_obstacles):
            for _attempt in range(50):
                horizontal = rng.random() < 0.5
                length = int(rng.integers(25, 60))
                thickness = int(rng.integers(3, 6))
                w, h = (length, thickness) if horizontal else (thickness, length)
                x = int(rng.integers(5, cfg.width - w - 5))
                y = int(rng.integers(5, cfg.height - h - 5))
                # keep a clear bubble around the colony
                nearest_x = min(max(cx, x), x + w)
                nearest_y = min(max(cy, y), y + h)
                if np.hypot(nearest_x - cx, nearest_y - cy) < cfg.colony_radius + 14:
                    continue
                self.obstacles[y:y + h, x:x + w] = True
                break

    def _place_food(self):
        cfg, rng = self.cfg, self.rng
        cx, cy = self.colony
        placed = 0
        attempts = 0
        while placed < cfg.n_food_sources and attempts < 500:
            attempts += 1
            x = rng.uniform(10, cfg.width - 10)
            y = rng.uniform(10, cfg.height - 10)
            if np.hypot(x - cx, y - cy) < 62:
                continue
            if self.obstacles[int(y), int(x)]:
                continue
            if any(np.hypot(x - f.x, y - f.y) < 30 for f in self.food_sources):
                continue
            self.food_sources.append(FoodSource(x, y, cfg.food_amount))
            placed += 1

    # ------------------------------------------------------- dynamic editing

    def add_food(self, x, y, amount=None):
        if amount is None:
            amount = self.cfg.food_amount
        if not self.obstacles[int(np.clip(y, 0, self.cfg.height - 1)),
                              int(np.clip(x, 0, self.cfg.width - 1))]:
            self.food_sources.append(FoodSource(x, y, amount))
            self._food_dirty = True

    def remove_food_near(self, x, y):
        if not self.food_sources:
            return
        dists = [np.hypot(f.x - x, f.y - y) for f in self.food_sources]
        self.food_sources.pop(int(np.argmin(dists)))
        self._food_dirty = True

    def paint_obstacle(self, x, y, radius=4, erase=False):
        cfg = self.cfg
        yy, xx = np.ogrid[:cfg.height, :cfg.width]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        # never wall in the colony itself
        cx, cy = self.colony
        if not erase and np.hypot(x - cx, y - cy) < cfg.colony_radius + 8:
            return
        self.obstacles[mask] = not erase
        if not erase:
            self.pheromone[mask] = 0.0
        self._home_dirty = True
        self._food_dirty = True

    # ----------------------------------------------------------------- queries

    def _cell_indices(self, xs, ys):
        xi = np.clip(xs, 0, self.cfg.width - 1).astype(np.int32)
        yi = np.clip(ys, 0, self.cfg.height - 1).astype(np.int32)
        return xi, yi

    def sample_pheromone(self, xs, ys):
        xi, yi = self._cell_indices(xs, ys)
        return self.pheromone[yi, xi]

    def sample_field(self, field, xs, ys):
        xi, yi = self._cell_indices(xs, ys)
        return field[yi, xi]

    def _circle_mask(self, centers, radius):
        yy, xx = np.ogrid[:self.cfg.height, :self.cfg.width]
        mask = np.zeros((self.cfg.height, self.cfg.width), dtype=bool)
        for x, y in centers:
            mask |= (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        return mask

    def _field_stale(self, dirty, field, last_tick):
        if field is None:
            return True
        return dirty and (self.tick - last_tick) >= FIELD_REFRESH_TICKS

    @property
    def home_field(self):
        """Geodesic distance (around walls) from every cell to the colony."""
        if self._field_stale(self._home_dirty, self._home_field, self._home_tick):
            seeds = self._circle_mask([self.colony], self.cfg.colony_radius)
            self._home_field = distance_field(self.obstacles, seeds)
            self._home_dirty = False
            self._home_tick = self.tick
        return self._home_field

    @property
    def food_field(self):
        """Geodesic distance from every cell to the nearest stocked food."""
        if self._field_stale(self._food_dirty, self._food_field, self._food_tick):
            centers = [(f.x, f.y) for f in self.food_sources if f.amount > 0]
            seeds = self._circle_mask(centers, self.cfg.food_radius)
            self._food_field = distance_field(self.obstacles, seeds)
            self._food_dirty = False
            self._food_tick = self.tick
        return self._food_field

    def is_blocked(self, pos):
        x, y = pos[..., 0], pos[..., 1]
        out = (x < 1) | (x >= self.cfg.width - 1) | (y < 1) | (y >= self.cfg.height - 1)
        xi, yi = self._cell_indices(x, y)
        return out | self.obstacles[yi, xi]

    def area_richness(self, source):
        """How much food sits around `source` (clustered piles count together),
        as a multiple of one standard pile. Rich patches recruit harder."""
        total = sum(f.amount for f in self.food_sources
                    if f.amount > 0
                    and np.hypot(f.x - source.x, f.y - source.y) <= self.cfg.richness_radius)
        return float(np.clip(total / self.cfg.food_amount, 0.3, self.cfg.richness_max))

    def try_pickup(self, pos, searching, radius):
        """Ants within `radius` of a stocked source each take one unit.
        Returns (picked mask, trail-strength multiplier per picked ant)."""
        picked = np.zeros(len(pos), dtype=bool)
        strength = np.zeros(len(pos), dtype=np.float32)
        for f in self.food_sources:
            if f.amount <= 0:
                continue
            d = np.linalg.norm(pos - np.array([f.x, f.y], dtype=np.float32), axis=1)
            candidates = np.flatnonzero(searching & ~picked & (d < radius))
            if len(candidates) == 0:
                continue
            take = candidates[:min(len(candidates), f.amount)]
            picked[take] = True
            strength[take] = self.area_richness(f)
            f.amount -= len(take)
            if f.amount <= 0:
                self._food_dirty = True  # depleted source stops attracting
        return picked, strength

    # ----------------------------------------------------------------- updates

    def deposit_pheromone(self, pos, strength):
        if len(pos) == 0:
            return
        xi, yi = self._cell_indices(pos[:, 0], pos[:, 1])
        np.add.at(self.pheromone, (yi, xi), strength.astype(np.float32))
        np.clip(self.pheromone, 0.0, self.cfg.pheromone_max, out=self.pheromone)
        self.pheromone[self.obstacles] = 0.0

    def mark_visited(self, pos):
        xi, yi = self._cell_indices(pos[:, 0], pos[:, 1])
        self.visited[yi, xi] = True

    def evaporate_and_diffuse(self):
        cfg = self.cfg
        p = self.pheromone
        if cfg.diffusion > 0:
            blur = (np.roll(p, 1, 0) + np.roll(p, -1, 0)
                    + np.roll(p, 1, 1) + np.roll(p, -1, 1)) * 0.25
            p = p * (1.0 - cfg.diffusion) + blur * cfg.diffusion
        p *= (1.0 - cfg.evaporation)
        p[p < 0.01] = 0.0
        p[self.obstacles] = 0.0
        self.pheromone = p
        self.tick += 1

    @property
    def explored_fraction(self):
        open_cells = ~self.obstacles
        return float(self.visited[open_cells].sum()) / float(open_cells.sum())

    @property
    def food_remaining(self):
        return sum(f.amount for f in self.food_sources)
