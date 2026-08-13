# Findings: pheromone evaporation rate vs. colony performance

Setup: 300 ants, 5000 steps per run, 5 seeds per rate, random obstacles and 3 food sources per world.

## Summary (mean across seeds)

|   evaporation |   steps |   food_collected |   time_to_find_food |   time_to_first_delivery |   avg_trip_length |   convergence_step |   explored_fraction |
|--------------:|--------:|-----------------:|--------------------:|-------------------------:|------------------:|-------------------:|--------------------:|
|          0.01 |    5000 |           3547.2 |                68.4 |                    134   |              80.7 |              742.4 |                   1 |
|          0.05 |    5000 |           3444.2 |                68.4 |                    134   |              80.4 |              688.2 |                   1 |
|          0.1  |    5000 |           3324   |                68.4 |                    133.8 |              78.9 |              796.8 |                   1 |
|          0.2  |    5000 |           1836.8 |                68.4 |                    133.6 |              77.8 |              716.8 |                   1 |
|          0.5  |    5000 |           1465.6 |                68.4 |                    134   |              79   |              454.6 |                   1 |

## Observations

- Highest total food collected: evaporation = 0.01.
- Fastest route convergence: evaporation = 0.5.
- Low evaporation rates preserve trails long enough for other ants to reinforce them, but stale trails to depleted food linger and mislead.
- High evaporation rates erase trails before they can be reinforced, so the colony degenerates toward independent random walkers.
- The sweet spot balances trail persistence against adaptability - strong enough memory to converge, fast enough forgetting to adapt.
