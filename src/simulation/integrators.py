"""
File: simulation/integrators.py
Author: Nathan Tandory <nathtand@gmail.com>
Date: 2026-04-09
Description: Numerical integrators for the 3D mass-spring-damper cloth simulation.
"""

import numpy as np
from enum import Enum

def euler(state: np.ndarray, dt: float, function) -> np.ndarray:
    return state + (dt * function(state))

def rk4(state: np.ndarray, dt: float, function) -> np.ndarray:
    k_1 = function(state)
    k_2 = function(state + 0.5 * dt * k_1)
    k_3 = function(state + 0.5 * dt * k_2)
    k_4 = function(state + dt * k_3)
    return state + (dt / 6.0) * (k_1 + (2 * k_2) + (2 * k_3) + k_4)
    
class integrator(Enum):
    EULER = 1
    RK4 = 2

    def __call__(self, state: np.ndarray, dt: float, function) -> np.ndarray:
        match self:
            case integrator.EULER: return euler(state, dt, function)
            case integrator.RK4: return rk4(state, dt, function)