"""Vectorized ant behavior. All N ants are updated at once with NumPy.

Behavior (per the classic ant-colony rules):

    if carrying food:
        pathfind toward the colony (descend the colony distance field, which
        routes around walls), deposit pheromone (stronger for fresh trips)
    else:
        if food is within smell radius: pathfind toward it (wall-aware)
        elif a pheromone trail is detected: probabilistically follow the
             stronger of three whisker samples (sometimes ignore it and explore)
        else: random walk

Pheromone deposits fade with trip length, so short paths end up with stronger
trails - that asymmetry is what lets the colony converge on efficient routes.
"""

import numpy as np

from .pathfield import UNREACHABLE


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Ants:
    def __init__(self, cfg, world, rng):
        self.cfg = cfg
        self.rng = rng
        n = cfg.n_ants
        self.n = n
        spread = rng.normal(0.0, 1.5, (n, 2)).astype(np.float32)
        self.pos = (world.colony[None, :] + spread).astype(np.float32)
        self.angle = rng.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
        self.carrying = np.zeros(n, dtype=bool)
        self.trip_steps = np.zeros(n, dtype=np.int32)
        # Trail-strength multiplier set at pickup: richer food patches make
        # returning ants lay stronger pheromone, recruiting more ants there.
        self.trail_strength = np.ones(n, dtype=np.float32)
        # Site fidelity: remembered location of the last food pickup.
        self.memory = np.zeros((n, 2), dtype=np.float32)
        self.has_memory = np.zeros(n, dtype=bool)
        self.memory_ttl = np.zeros(n, dtype=np.int32)

        self._sensor_offsets = np.array(
            [-cfg.sensor_angle, 0.0, cfg.sensor_angle], dtype=np.float32
        )

        # per-step outputs for metrics
        self.delivered_this_step = 0
        self.picked_this_step = 0
        self.completed_trip_lengths = []

    def add_ants(self, x, y, count):
        """Spawn `count` new searching ants around (x, y) at runtime."""
        rng = self.rng
        new_pos = (np.array([x, y], dtype=np.float32)[None, :]
                   + rng.normal(0.0, 1.5, (count, 2)).astype(np.float32))
        self.pos = np.concatenate([self.pos, new_pos])
        self.angle = np.concatenate(
            [self.angle, rng.uniform(0.0, 2.0 * np.pi, count).astype(np.float32)])
        self.carrying = np.concatenate([self.carrying, np.zeros(count, dtype=bool)])
        self.trip_steps = np.concatenate(
            [self.trip_steps, np.zeros(count, dtype=np.int32)])
        self.trail_strength = np.concatenate(
            [self.trail_strength, np.ones(count, dtype=np.float32)])
        self.memory = np.concatenate(
            [self.memory, np.zeros((count, 2), dtype=np.float32)])
        self.has_memory = np.concatenate(
            [self.has_memory, np.zeros(count, dtype=bool)])
        self.memory_ttl = np.concatenate(
            [self.memory_ttl, np.zeros(count, dtype=np.int32)])
        self.n += count

    def step(self, world):
        cfg, rng = self.cfg, self.rng
        n = self.n
        pos, angle, carrying = self.pos, self.angle, self.carrying
        searching = ~carrying

        # Baseline: random wander
        turn = rng.normal(0.0, cfg.wander, n).astype(np.float32)

        # --- Trail following (searching ants) -----------------------------
        sensor_angles = angle[:, None] + self._sensor_offsets[None, :]     # (n, 3)
        sx = pos[:, 0, None] + np.cos(sensor_angles) * cfg.sensor_dist
        sy = pos[:, 1, None] + np.sin(sensor_angles) * cfg.sensor_dist
        s = world.sample_pheromone(sx, sy)                                 # (n, 3)

        weights = (s + 0.1) ** cfg.trail_power
        weights /= weights.sum(axis=1, keepdims=True)
        r = rng.random(n)
        choice = np.where(r < weights[:, 0], 0,
                          np.where(r < weights[:, 0] + weights[:, 1], 1, 2))
        trail_turn = (choice - 1).astype(np.float32) * cfg.turn_rate

        # Follow probability: any established trail gets a solid base rate
        # (so young trails can bootstrap), plus a bonus that rises with trail
        # strength (saturating response). Trails from rich food patches are
        # laid stronger, get followed more reliably, and therefore recruit
        # more ants - this is what pulls the colony toward food clusters.
        s_max = s.max(axis=1)
        strength_bonus = s_max ** 2 / (s_max ** 2 + cfg.trail_half_sat ** 2)
        p_follow = np.where(s_max > 0.5,
                            cfg.trail_base_follow
                            + (1.0 - cfg.trail_base_follow) * strength_bonus,
                            0.0)
        keep_exploring = rng.random(n) < cfg.exploration_prob
        follow = searching & (rng.random(n) < p_follow) & ~keep_exploring
        turn = np.where(follow, trail_turn, turn)

        # --- Site fidelity: head back to a remembered food site ------------
        # (overridden below by direct smell once the ant gets close)
        mem_active = searching & self.has_memory
        if mem_active.any():
            self.memory_ttl[mem_active] -= 1
            toward_mem = wrap_angle(
                np.arctan2(self.memory[:, 1] - pos[:, 1],
                           self.memory[:, 0] - pos[:, 0]) - angle)
            mem_turn = (np.clip(toward_mem, -2 * cfg.turn_rate, 2 * cfg.turn_rate)
                        + rng.normal(0.0, cfg.wander * 0.3, n).astype(np.float32))
            turn = np.where(mem_active, mem_turn, turn)
            # Forget: memory expired, or we arrived and the patch is gone.
            mem_dist = np.linalg.norm(pos - self.memory, axis=1)
            arrived = mem_active & (mem_dist < 5.0)
            expired = mem_active & (self.memory_ttl <= 0)
            self.has_memory[arrived | expired] = False

        # --- Direct food smell overrides the trail ------------------------
        # The food field is a wall-aware distance-to-food map, so ants smell
        # (and pathfind) around obstacles, never through them.
        food_field = world.food_field
        food_dist_here = world.sample_field(food_field, pos[:, 0], pos[:, 1])
        near_food = food_dist_here < cfg.sense_food_radius
        if near_food.any():
            fs = world.sample_field(food_field, sx, sy)             # (n, 3)
            food_turn = (fs.argmin(axis=1) - 1).astype(np.float32) * cfg.turn_rate * 1.5
            turn = np.where(searching & near_food, food_turn, turn)

        # --- Homing (carrying ants): pathfind via the colony field ---------
        home_field = world.home_field
        hs = world.sample_field(home_field, sx, sy)                 # (n, 3)
        field_turn = ((hs.argmin(axis=1) - 1).astype(np.float32) * cfg.turn_rate * 1.5
                      + rng.normal(0.0, cfg.wander * 0.3, n).astype(np.float32))
        # Straight-line fallback for the rare case of a walled-in pocket.
        toward_home = wrap_angle(
            np.arctan2(world.colony[1] - pos[:, 1], world.colony[0] - pos[:, 0]) - angle
        )
        direct_turn = np.clip(toward_home, -2 * cfg.turn_rate, 2 * cfg.turn_rate)
        home_turn = np.where(hs.min(axis=1) < UNREACHABLE, field_turn, direct_turn)
        turn = np.where(carrying, home_turn, turn)

        # --- Move, colliding with walls and obstacles ----------------------
        angle += turn
        new_pos = pos + np.stack(
            [np.cos(angle), np.sin(angle)], axis=1
        ).astype(np.float32) * cfg.speed
        blocked = world.is_blocked(new_pos)
        free = ~blocked
        pos[free] = new_pos[free]
        n_blocked = int(blocked.sum())
        if n_blocked:
            angle[blocked] += np.pi + rng.normal(0.0, 0.6, n_blocked).astype(np.float32)
        np.clip(pos[:, 0], 1.0, cfg.width - 2.0, out=pos[:, 0])
        np.clip(pos[:, 1], 1.0, cfg.height - 2.0, out=pos[:, 1])
        self.angle = wrap_angle(angle).astype(np.float32)

        # --- Deposit pheromone (carrying ants; fades with trip length) -----
        self.trip_steps[carrying] += 1
        if carrying.any():
            freshness = np.maximum(
                0.0, 1.0 - self.trip_steps[carrying] * cfg.deposit_decay_per_step
            )
            world.deposit_pheromone(
                pos[carrying],
                cfg.pheromone_deposit * freshness * self.trail_strength[carrying])

        # --- Pick up food ----------------------------------------------------
        picked, strength = world.try_pickup(pos, searching, cfg.food_radius)
        self.picked_this_step = int(picked.sum())
        if self.picked_this_step:
            carrying[picked] = True
            self.trip_steps[picked] = 0
            self.trail_strength[picked] = strength[picked]
            self.memory[picked] = pos[picked]

        # --- Deliver to colony ----------------------------------------------
        home_dist = np.linalg.norm(pos - world.colony[None, :], axis=1)
        delivered = carrying & (home_dist < cfg.colony_radius)
        self.delivered_this_step = int(delivered.sum())
        if self.delivered_this_step:
            self.completed_trip_lengths.extend(self.trip_steps[delivered].tolist())
            carrying[delivered] = False
            # Site fidelity roll: richer patches keep more of their foragers.
            idx = np.flatnonzero(delivered)
            p_return = np.clip(cfg.site_fidelity_per_richness
                               * self.trail_strength[idx],
                               0.0, cfg.site_fidelity_max)
            loyal = rng.random(len(idx)) < p_return
            self.has_memory[idx] = loyal
            self.memory_ttl[idx[loyal]] = cfg.memory_ttl

        world.mark_visited(pos)
