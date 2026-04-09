"""
File: simulation/derivatives.py
Author: Nathan Tandory <nathtand@gmail.com>
Date: 2026-04-09
Description: Computation of the derivative values of the state vector for the 3D mass-spring-damper cloth simulation.
"""

import numpy as np
from models.cloth import cloth

def state_derivatives(state: np.ndarray, cloth: cloth, sim_params: dict) -> np.ndarray:
    """
    computes and returns the derivative values of the cloth state vector
    parameters:
        - state: flat array of particle positions and velocities
        - cloth: cloth object
        - sim_params: dictionary of simulation parameters
    returns:
        - flat array of particle velocities and accelerations
            formatted: [v_x_1, v_y_1, v_z_1, a_x_1, a_y_1, a_z_1, ...]
    """
    num_particles = cloth.num_particles
    state_2d = state.reshape((num_particles, 6)) # reshape state vector into 2D (6 state variables per particle)
    positions = state_2d[:, :3] # all particle positions (num_particles, 3)
    velocities = state_2d[:, 3:] # all particle velocities (num_particles, 3)
    forces = np.zeros((num_particles, 3), 'float64') # initialize new 2D forces array (3D vector for each particle)

    #
    # force: gravity
    #
    forces += sim_params["particle_mass"] * sim_params["gravity"]

    #
    # force: wind
    #
    angle_rad = np.radians(sim_params["wind_angle"])
    wind_dir = np.array([np.cos(angle_rad), 0.0, np.sin(angle_rad)])
    forces += wind_dir * sim_params["wind_strength"]

    #
    # force: springs + damping
    #
    displacements = positions[cloth._spring_b] - positions[cloth._spring_a] # displacement for each spring
    distances = np.linalg.norm(displacements, axis = 1) # distance vector for each spring
    safe = distances > 1e-9 # flag distances too small to avoid 0 division operations
    # compute unit vectors for each spring (a to b)
    unit_vectors = np.zeros_like(displacements, 'float64')
    unit_vectors[safe] = displacements[safe] / distances[safe, np.newaxis]
    # hooke's law: f_spring = k * (current_length - rest_length)
    spring_magnitudes = cloth._spring_k * (distances - cloth._spring_rest)
    spring_magnitudes[~safe] = 0.0
    f_spring = spring_magnitudes[:, np.newaxis] * unit_vectors # spring force (magnitude * direction)
    # damping: f_damping = c * (v_relative * unit_vector) * unit_vector
    velocities_relative = velocities[cloth._spring_b] - velocities[cloth._spring_a] # compute relative velocities
    velocities_proj = np.sum(velocities_relative * unit_vectors, axis = 1) # project relative velocities onto unit vectors
    f_damping = (cloth._spring_damping * velocities_proj)[:, np.newaxis] * unit_vectors # compute damping forces
    f_total = f_spring + f_damping # finally: total spring + damping force
    # add spring + damper forces to particle forces
    np.add.at(forces, cloth._spring_a, f_total)
    np.add.at(forces, cloth._spring_b, -f_total)

    #
    # construct derivatives
    #
    derivatives = np.zeros_like(state_2d)
    derivatives[:, 0:3] = velocities # velocity
    derivatives[:, 3:6] = forces / sim_params["particle_mass"] # acceleration = force / mass
    derivatives[cloth._pinned_mask] = 0.0 # cancel derivatives for pinned particles
    
    return derivatives.ravel() # flatten derivates from 2D back to 1D and return