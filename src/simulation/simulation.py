import threading
import numpy as np
from simulation.integrators import integrator
from simulation.forces import compute_derivatives
from models.cloth import cloth, cloth_config

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
    def __init__(self, simulation_params: dict, cloth_config: cloth_config) -> None:
        self.params = simulation_params
        self.cloth = cloth(cloth_config, self.params)
        self.state = self.cloth.build_initial_state()
        self.time = 0.0
        self.paused = True
        # thread lock to ensure no race conditions occur while resetting or rebuilding the cloth
        self._lock = threading.Lock()
        self.title = self.params["title"]


    
    # simulation core

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        return compute_derivatives(state, self.cloth, self.params)

    def step(self) -> None:
        with self._lock:
            self.state = self.params["integrator"](self.state, self.params["dt"], self.derivatives)
            self.time += self.params["dt"]
    
    def pause(self) -> None:
        self.paused = True
    
    def resume(self) -> None:
        self.paused = False



    # cloth management
    
    def reset(self) -> None:
        with self._lock:
            self.cloth = cloth(self.cloth.config, self.params)
            self.state = self.cloth.build_initial_state()
            self.time = 0.0
    
    def rebuild_cloth(self, new_config: cloth_config) -> cloth:
        with self._lock:
            self.cloth = cloth(new_config, self.params)
            self.state = self.cloth.build_initial_state()
            self.time = 0.0
            return self.cloth
