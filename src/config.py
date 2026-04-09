"""
File: config.py
Author: Nathan Tandory <nathtand@gmail.com>
Date: 2026-04-09
Description: Default configuration for the 3D mass-spring-damper cloth simulation.
"""

import numpy as np
from simulation.integrators import integrator

# default cloth configuration
cloth_config = {
    "rows": 18,
    "columns": 12,
    "spacing": 0.2,
    # only one of the following should be true
    "pin_top_row": True,
    "pin_top_corners": False,
    "pin_left_column": False,
    "pin_right_column": False,
    "pin_all_edges": False
}
# default simulation parameters
sim_params = {
        "particle_mass": 0.05,
        "k_structural": 600.0,
        "k_shear": 600.0,
        "k_bend": 600.0,
        "damping": 0.8,
        "gravity": np.array([0.0, -9.8, 0.0]),
        "wind_strength": 1.0,
        "wind_angle": 90.0,
        "dt": 0.0075,
        "integrator": integrator.RK4,
        "title": "Cloth Simulation"
}