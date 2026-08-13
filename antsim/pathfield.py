"""Grid pathfinding via distance fields (a.k.a. flow fields).

`distance_field` flood-fills geodesic distance from a set of seed cells,
treating obstacles as impassable. An ant can then path toward the seeds
from anywhere simply by stepping toward the neighboring cell with the
lowest field value - one shared field serves all ants at once, which is
why this stays fast with thousands of ants.
"""

import numpy as np

INF = np.float32(1.0e9)
UNREACHABLE = 1.0e8  # threshold for "no path exists"


def distance_field(obstacles, seed_mask, max_iters=None):
    """Vectorized 8-neighbor wavefront expansion (diagonal cost sqrt(2))."""
    h, w = obstacles.shape
    dist = np.full((h, w), INF, dtype=np.float32)
    dist[seed_mask & ~obstacles] = 0.0
    if max_iters is None:
        max_iters = h + w

    straight = np.float32(1.0)
    diagonal = np.float32(np.sqrt(2.0))

    for it in range(max_iters):
        padded = np.pad(dist, 1, constant_values=INF)
        new = dist.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                cost = diagonal if (dy and dx) else straight
                np.minimum(new, padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] + cost,
                           out=new)
        new[obstacles] = INF
        if it % 8 == 7 and np.array_equal(new, dist):
            return new
        dist = new
    return dist
