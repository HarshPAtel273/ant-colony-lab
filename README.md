# Ant Colony Lab

A research-grade digital ant colony: hundreds to thousands of virtual ants
following simple rules, communicating indirectly through an evaporating
pheromone map, avoiding obstacles, and collectively converging on efficient
foraging routes. No ant is given a map — efficient paths *emerge*.

Built with NumPy (vectorized simulation), Pygame (real-time visualization),
Matplotlib and Pandas (research experiments).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Watch the colony

```bash
.venv/bin/python run_sim.py
```

| Control | Action |
| --- | --- |
| SPACE | pause / resume |
| UP / DOWN | simulation speed (1–16 steps per frame) |
| LEFT CLICK | drop a new food source (test dynamic adaptation) |
| X | remove the food source nearest the mouse |
| A | add 20 ants at the mouse position |
| HOLD W | draw walls at the mouse position |
| HOLD E | erase walls at the mouse position |
| RIGHT DRAG | draw obstacles (hold SHIFT to erase) |
| R | reset with a new random world |
| ESC / Q | quit |

Flags: `--ants 1000`, `--evaporation 0.05`, `--seed 42`.

## The rules each ant follows

```
if carrying food:
    pathfind toward the colony (wall-aware)
    deposit pheromone (deposit fades as the trip drags on)
else:
    if food is within smell radius:  pathfind toward it (wall-aware)
    elif pheromone trail detected:   probabilistically follow the stronger
                                     whisker sample (sometimes ignore it
                                     and keep exploring)
    else:                            random walk

pheromone: evaporates and diffuses every step
```

### Rich areas recruit harder

Two mechanisms make food-dense areas pull in more of the colony:

1. **Richness-scaled trails** — at pickup, an ant checks how much food sits
   within `richness_radius` of the pile (clustered piles count together) and
   lays proportionally stronger pheromone on the way home. Trail following is
   probabilistic with a strength-dependent bonus, so stronger trails hold
   ants more reliably.
2. **Site fidelity** — after delivering, a forager returns straight to its
   remembered food site with probability scaled by the area's richness
   (rich patch: up to 90%, poor pile: ~30%). Rich areas accumulate a loyal
   workforce; memories fade after `memory_ttl` steps or when the patch is
   mined out.

### Pathfinding

Navigation uses **distance fields** (`antsim/pathfield.py`): a wavefront
flood-fill computes the geodesic distance from every cell to the colony
(and to the nearest food), routing around walls. An ant pathfinds by simply
stepping toward whichever of its three whisker points has the lowest field
value — one shared field serves all ants at once, so this stays fast with
thousands of ants. Fields recompute automatically when you draw walls or
food changes. Finding food in the first place is still driven by exploration
and pheromone recruitment — pathfinding only handles *navigation*, so the
emergent trail behavior is preserved.

Because deposits fade with trip length, **short paths accumulate stronger
trails** (fresher deposits + more round-trips per ant), which recruits more
ants, which reinforces the trail further — the classic emergent
shortest-path optimization.

## Run the research experiment

> How does pheromone evaporation rate affect the colony's ability to
> discover and exploit food?

```bash
.venv/bin/python run_experiment.py
```

Sweeps evaporation ∈ {0.01, 0.05, 0.10, 0.20, 0.50} across 5 seeds each
(25 headless runs) and writes to `results/`:

- `experiment_results.csv` — raw per-run metrics
- `summary.csv` — mean per evaporation rate
- `food_over_time.png` — cumulative food curves per rate
- `metrics_by_rate.png` — all metrics as bar charts
- `findings.md` — auto-generated write-up

Measured per run: time to find food, time to first delivery, total food
collected, average return-trip length, convergence time (when the delivery
rate reaches 80% of its peak), and exploration efficiency (fraction of the
map visited).

Customize: `--rates 0.005 0.02 0.08 --seeds 10 --steps 8000 --ants 500`.

## Project layout

- `antsim/config.py` — every tunable parameter in one dataclass
- `antsim/world.py` — pheromone grid (evaporation + diffusion), food, obstacles, colony
- `antsim/ants.py` — vectorized ant behavior (all N ants updated at once)
- `antsim/simulation.py` — glue + metrics recording
- `run_sim.py` — interactive Pygame visualization
- `run_experiment.py` — headless experiment sweep + graphs

## Roadmap (not yet implemented)

- Ant roles (workers / scouts / soldiers / explorers)
- Genetic evolution of per-ant parameters (`exploration_prob`,
  `trail_power`, `speed` are already in the config, ready to vary per ant)
- Multiple competing colonies
- Pheromone ants vs. reinforcement-learned ants comparison
- FastAPI + React/Three.js interactive web front-end
