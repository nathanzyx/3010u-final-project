"""
File: simulation/simulation.py
Author: Nathan Tandory <nathtand@gmail.com>
Date: 2026-04-09
Description: Simulation class for the 3D mass-spring-damper cloth simulation.
"""

import threading
import numpy as np
from simulation.integrators import integrator
from simulation.derivatives import state_derivatives
from models.cloth import cloth

class simulation:
    """
    attributes:
        - params: dictionary of simulation parameters
        - cloth: cloth object
        - state: flat array of particle positions and velocities
        - time: float of simulation time
        - paused: boolean of simulation state
        - _lock: threading lock to prevent race conditions
        - title: string of simulation title
    """
    def __init__(self, simulation_params: dict, cloth_config: dict) -> None:
        self.params = simulation_params
        self.cloth = cloth(cloth_config, self.params)
        self.state = self.cloth.build_initial_state()
        self.time = 0.0
        self.paused = True
        self._lock = threading.Lock() # thread lock to prevent race conditions while resetting/rebuilding cloth
        self.title = self.params["title"]
    
    # simulation core

    def derivatives(self, state: np.ndarray) -> np.ndarray: return state_derivatives(state, self.cloth, self.params)

    def step(self) -> None:
        with self._lock:
            self.state = self.params["integrator"](self.state, self.params["dt"], self.derivatives)
            self.time += self.params["dt"]
    
    def pause(self) -> None: self.paused = True
    
    def resume(self) -> None: self.paused = False

    # cloth management
    
    def reset(self) -> None:
        """
        resets the simulation (resets cloth to initial state and sets time to 0)
        """
        with self._lock:
            self.cloth = cloth(self.cloth.config, self.params)
            self.state = self.cloth.build_initial_state()
            self.time = 0.0
    
    def rebuild_cloth(self, new_config: dict) -> cloth:
        """
        rebuilds the cloth with new parameters (resets cloth + state + time)
        parameters:
            - new_config: dictionary of new cloth parameters
        returns:
            - cloth: the new cloth object
        """
        with self._lock:
            self.cloth = cloth(new_config, self.params)
            self.state = self.cloth.build_initial_state()
            self.time = 0.0
            return self.cloth
