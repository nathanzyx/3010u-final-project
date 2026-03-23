import numpy as np
from models.cloth import cloth

def compute_derivatives(state: np.ndarray, cloth: cloth, sim_params: dict) -> np.ndarray:
    """
    computes rate of change of cloth state
    parameters:
        - state: flat array of particle positions and velocities
        - cloth: cloth object
        - sim_params: dictionary of simulation parameters
    returns:
        - flat array of particle velocities and accelerations
            formatted: [v_x_1, v_y_1, v_z_1, a_x_1, a_y_1, a_z_1, ...]
    """
    num_particles = cloth.num_particles

    # reshape state array where each row is one particle (x_i, y_i, z_i, v_x_i, v_y_i, v_z_i)
    state_2d = state.reshape((num_particles, 6))
    positions = state_2d[:, :3] # all particle positions (num_particles, 3)
    velocities = state_2d[:, 3:] # all particle velocities (num_particles, 3)

    # initialize new forces array (3D force vector for each particle)
    forces = np.zeros((num_particles, 3), 'float64')


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
    # get particle positions for each spring (_spring_a, _spring_b are arrays of particle indices)
    pos_a = positions[cloth._spring_a]
    pos_b = positions[cloth._spring_b]
    
    # compute displacement and distance vectors for each spring
    displacements = pos_b - pos_a
    distances = np.linalg.norm(displacements, axis = 1)

    # flag distances too small to avoid 0 division operations
    safe = distances > 1e-9

    # compute unit vectors for each spring (a to b)
    unit_vectors = np.zeros_like(displacements, 'float64')
    unit_vectors[safe] = displacements[safe] / distances[safe, np.newaxis]

    # hooke's law: f_spring = k * (current_length - rest_length)
    spring_magnitudes = cloth._spring_k * (distances - cloth._spring_rest)
    spring_magnitudes[~safe] = 0.0

    # spring force (magnitude * direction)
    f_spring = spring_magnitudes[:, np.newaxis] * unit_vectors
    
    # damping: f_damping = c * (v_relative * unit_vector) * unit_vector
    # compute relative velocities
    velocities_a = velocities[cloth._spring_a]
    velocities_b = velocities[cloth._spring_b]
    velocities_relative = velocities_b - velocities_a
    # project relative velocities onto unit vectors
    velocities_proj = np.sum(velocities_relative * unit_vectors, axis = 1)
    # compute damping forces
    f_damping = (cloth._spring_damping * velocities_proj)[:, np.newaxis] * unit_vectors

    # finally: total spring + damping force
    f_total = f_spring + f_damping
    
    # add forces to particles
    np.add.at(forces, cloth._spring_a, f_total)
    np.add.at(forces, cloth._spring_b, -f_total)


    #
    # compute derivatives
    #
    derivatives = np.zeros_like(state_2d)
    derivatives[:, 0:3] = velocities # velocity
    derivatives[:, 3:6] = forces / sim_params["particle_mass"] # acceleration = force / mass
    
    # cancel derivatives for pinned particles
    derivatives[cloth._pinned_mask] = 0.0
    
    # flatten derivates from state_2d back to 1D and return
    return derivatives.ravel()