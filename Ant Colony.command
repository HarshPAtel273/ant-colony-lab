#!/bin/zsh
# Double-click this file in Finder to launch the ant colony simulation.
cd "$(dirname "$0")"
exec .venv/bin/python run_sim.py
