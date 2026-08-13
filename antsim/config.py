from dataclasses import dataclass
from typing import Optional


@dataclass
class SimConfig:
    """All tunable parameters of the simulation.

    The experiment runner sweeps `evaporation`; everything else is a knob you
    can play with (several map directly onto the 'genetic evolution' roadmap:
    exploration_prob, trail_power, speed, ...).
    """

    # World (grid cells)
    width: int = 240
    height: int = 160
    n_food_sources: int = 3
    food_amount: int = 1200         # units of food per source
    food_radius: float = 4.0        # pickup radius
    colony_radius: float = 5.0
    n_obstacles: int = 5            # random walls placed at reset
    seed: Optional[int] = None

    # Ants
    n_ants: int = 300
    speed: float = 1.0              # cells per step
    turn_rate: float = 0.35         # radians per steering decision
    wander: float = 0.25            # std-dev of random steering noise
    exploration_prob: float = 0.05  # chance to ignore a trail and keep exploring

    # Sensing
    sensor_angle: float = 0.45      # whisker spread (radians)
    sensor_dist: float = 6.0        # whisker reach (cells)
    sense_food_radius: float = 6.0  # direct food smell radius
    trail_power: float = 2.0        # weight = (pheromone + eps) ** power
    trail_base_follow: float = 0.5  # follow chance for any detected trail
    trail_half_sat: float = 30.0    # pheromone level at which the strength
                                    # bonus reaches half; stronger trails
                                    # hold ants more reliably

    # Pheromone
    pheromone_deposit: float = 12.0        # deposit per step right after pickup
    richness_radius: float = 25.0          # area considered "one food patch"
    richness_max: float = 3.0              # max trail boost from a rich patch

    # Site fidelity: after a delivery, foragers return to their last food
    # site with probability scaled by that area's richness, so rich patches
    # accumulate a loyal workforce.
    site_fidelity_per_richness: float = 0.3
    site_fidelity_max: float = 0.9
    memory_ttl: int = 600                  # steps before a food memory fades
    deposit_decay_per_step: float = 0.005  # deposit fades as the trip drags on
    evaporation: float = 0.015             # fraction lost per step
    diffusion: float = 0.06                # neighbor mixing per step
    pheromone_max: float = 600.0  # high ceiling so strong trails can outshine weak ones
