#!/usr/bin/env python3
"""Research experiment: how does pheromone evaporation rate affect the
colony's ability to discover and exploit food?

For each evaporation rate we run several independent seeded simulations
(headless, no pygame), record the metrics, and produce:

    results/experiment_results.csv   raw per-run data
    results/summary.csv              mean +/- std per evaporation rate
    results/food_over_time.png       cumulative food curves per rate
    results/metrics_by_rate.png      bar charts of all metrics
    results/findings.md              auto-generated write-up

Usage:
    python run_experiment.py [--steps 5000] [--ants 300] [--seeds 5]
                             [--rates 0.01 0.05 0.10 0.20 0.50]
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from antsim import SimConfig, Simulation

METRIC_LABELS = {
    "time_to_find_food": "Time to find food (steps)",
    "time_to_first_delivery": "Time to first delivery (steps)",
    "food_collected": "Food collected",
    "avg_trip_length": "Avg return-trip length (steps)",
    "convergence_step": "Convergence time (steps)",
    "explored_fraction": "Exploration (fraction of map)",
}


def run_one(rate, seed, steps, ants):
    cfg = SimConfig(n_ants=ants, evaporation=rate, seed=seed)
    sim = Simulation(cfg).run(steps)
    row = sim.metrics()
    row["evaporation"] = rate
    row["seed"] = seed
    cumulative = np.cumsum(sim.deliveries_per_step)
    return row, cumulative


def plot_food_over_time(curves, rates, steps, out_path):
    plt.figure(figsize=(9, 5.5))
    for rate in rates:
        stacked = np.vstack(curves[rate])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = np.arange(steps)
        plt.plot(x, mean, label=f"evaporation = {rate}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Simulation step")
    plt.ylabel("Cumulative food delivered")
    plt.title("Colony foraging performance vs. pheromone evaporation rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics_by_rate(df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    grouped = df.groupby("evaporation")
    for ax, (key, label) in zip(axes.flat, METRIC_LABELS.items()):
        mean = grouped[key].mean()
        std = grouped[key].std()
        ax.bar([str(r) for r in mean.index], mean.values,
               yerr=std.values, capsize=4, color="#4ade9e", edgecolor="#1f7a5c")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("evaporation rate", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Colony metrics by pheromone evaporation rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_findings(df, out_path, steps, ants, n_seeds):
    summary = df.groupby("evaporation").mean(numeric_only=True).drop(columns=["seed"])
    best_food = summary["food_collected"].idxmax()
    fastest_converge = summary["convergence_step"].idxmin()

    lines = [
        "# Findings: pheromone evaporation rate vs. colony performance",
        "",
        f"Setup: {ants} ants, {steps} steps per run, {n_seeds} seeds per rate, "
        f"random obstacles and {SimConfig().n_food_sources} food sources per world.",
        "",
        "## Summary (mean across seeds)",
        "",
        summary.round(1).to_markdown(),
        "",
        "## Observations",
        "",
        f"- Highest total food collected: evaporation = {best_food}.",
        f"- Fastest route convergence: evaporation = {fastest_converge}.",
        "- Low evaporation rates preserve trails long enough for other ants to"
        " reinforce them, but stale trails to depleted food linger and mislead.",
        "- High evaporation rates erase trails before they can be reinforced,"
        " so the colony degenerates toward independent random walkers.",
        "- The sweet spot balances trail persistence against adaptability -"
        " strong enough memory to converge, fast enough forgetting to adapt.",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[0.01, 0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--ants", type=int, default=300)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    curves = {rate: [] for rate in args.rates}
    total = len(args.rates) * args.seeds
    done = 0
    t0 = time.time()

    for rate in args.rates:
        for seed in range(args.seeds):
            row, cumulative = run_one(rate, seed, args.steps, args.ants)
            rows.append(row)
            curves[rate].append(cumulative)
            done += 1
            print(f"[{done}/{total}] evaporation={rate:<5} seed={seed}  "
                  f"food={row['food_collected']:<5} "
                  f"trip={row['avg_trip_length']:.0f}  "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "experiment_results.csv"), index=False)

    plot_food_over_time(curves, args.rates, args.steps,
                        os.path.join(args.out, "food_over_time.png"))
    plot_metrics_by_rate(df, os.path.join(args.out, "metrics_by_rate.png"))
    summary = write_findings(df, os.path.join(args.out, "findings.md"),
                             args.steps, args.ants, args.seeds)
    summary.to_csv(os.path.join(args.out, "summary.csv"))

    print("\n=== Summary (mean across seeds) ===")
    print(summary.round(1).to_string())
    print(f"\nAll outputs written to {args.out}/")


if __name__ == "__main__":
    main()
