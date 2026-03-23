from vpython import rate
import numpy as np
from models.cloth import cloth_config
from simulation.simulation import simulation
from rendering.renderer import renderer
from simulation.integrators import integrator

def main() -> None:
    config = cloth_config(
        rows = 18,
        columns = 12,
        spacing = 0.2,
        pin_top_row = True,
        pin_top_corners = False,
        pin_left_column = False,
        pin_right_column = False,
        pin_all_edges = False,
    )
    sim_params = {
        "particle_mass": 0.05,
        "k_structural": 200.0,
        "k_shear": 200.0,
        "k_bend": 200.0,
        "damping": 0.8,
        "gravity": np.array([0.0, -9.8, 0.0]),
        "wind_strength": 1.0,
        "wind_angle": 90.0,
        "dt": 0.0075,
        "integrator": integrator.RK4,
        "title": "Cloth Simulation"
    }

    sim = simulation(sim_params, config)
    display = renderer(sim)

    sim.resume()

    while True:
        rate(60)
        if not sim.paused:
            sim.step()
        display.update()

if __name__ == "__main__":
    main()