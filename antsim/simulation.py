"""Ties the world and the ants together and records research metrics."""

import numpy as np

from .ants import Ants
from .world import World


class Simulation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.world = World(cfg, self.rng)
        self.ants = Ants(cfg, self.world, self.rng)

        self.step_count = 0
        self.total_delivered = 0
        self.deliveries_per_step = []   # for convergence / rate analysis
        self.first_pickup_step = None   # "time to find food"
        self.first_delivery_step = None

    def step(self):
        self.ants.step(self.world)
        self.world.evaporate_and_diffuse()
        self.step_count += 1

        d = self.ants.delivered_this_step
        self.total_delivered += d
        self.deliveries_per_step.append(d)
        if self.first_pickup_step is None and self.ants.picked_this_step > 0:
            self.first_pickup_step = self.step_count
        if self.first_delivery_step is None and d > 0:
            self.first_delivery_step = self.step_count

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()
        return self

    # -------------------------------------------------------------- metrics

    @property
    def avg_trip_length(self):
        trips = self.ants.completed_trip_lengths
        return float(np.mean(trips)) if trips else float("nan")

    def convergence_step(self, window=250, frac=0.8):
        """First step at which the rolling delivery rate reaches `frac` of its
        eventual peak - i.e. when the colony 'locked in' its foraging routes."""
        arr = np.asarray(self.deliveries_per_step, dtype=np.float64)
        if arr.sum() == 0 or len(arr) < window:
            return float("nan")
        rate = np.convolve(arr, np.ones(window) / window, mode="valid")
        peak = rate.max()
        if peak <= 0:
            return float("nan")
        return float(np.argmax(rate >= frac * peak) + window)

    def metrics(self):
        return {
            "steps": self.step_count,
            "food_collected": self.total_delivered,
            "time_to_find_food": (float("nan") if self.first_pickup_step is None
                                  else self.first_pickup_step),
            "time_to_first_delivery": (float("nan") if self.first_delivery_step is None
                                       else self.first_delivery_step),
            "avg_trip_length": self.avg_trip_length,
            "convergence_step": self.convergence_step(),
            "explored_fraction": self.world.explored_fraction,
        }
