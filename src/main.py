"""
File: main.py
Author: Nathan Tandory <nathtand@gmail.com>
Date: 2026-04-09
Description: Main entry point for the 3D mass-spring-damper cloth simulation.
"""

from vpython import rate
from simulation.simulation import simulation
from rendering.renderer import renderer
from config import cloth_config, sim_params

def main() -> None:
    sim = simulation(sim_params, cloth_config)
    display = renderer(sim)

    sim.resume()

    while True:
        rate(60)
        if not sim.paused:
            sim.step()
        display.update()

if __name__ == "__main__":
    main()