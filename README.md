# 3D Mass-Spring Cloth Simulation
**Course**: CSCI 3010U Simulation & Modeling
**Author**: Nathan Tandory

## Overview
A real-time 3D cloth simulation using a mass-spring-damper model. The cloth is modeled as a grid of particles connected by structural, shear, and bend springs. Two numerical solvers, Euler's method and Fourth-order Runge-kutta (RK4) are implemented. Built with Python, VPython, and NumPy.

Detailed analysis, results, and discussion are provided in the course report PDF.

## Requirements
- Python 3.10+
- pip

## Setup

Create and activate a virtual environment:
```bash
python -m venv .venv

# for Windows:
.venv\Scripts\activate

# for macOS / Linux:
source .venv/bin/activate
```

Install dependencies:
```bash
pip install setuptools numpy vpython
```

## Running
From the project root, with the virtual environment activated:
```bash
python src/main.py
```

A browser window will open with the 3D visualization and control panel.

## Controls
- `Pause / Resume`: pause or resume the simulation
- `Reset Cloth`: reset the cloth to its initial state
- `Sliders`: adjust gravity, particle mass, wind strength, wind angle, and time-step in real time
- `Integrator dropdown`: switch between RK4 and Euler
- `Cloth Generation panel`: configure grid size, spacing, spring constants, damping, and pinning, then click Generate Cloth to rebuild

## Project Structure
```
src/
├── main.py                     # entry point and configuration
├── models/
│   └── cloth.py                # cloth grid structure, springs, and state
├── rendering/
|   └── renderer.py             # vpython visualization and UI
└── simulation/
    ├── forces.py               # vectorized force computation
    ├── integrators.py          # euler and rk4 integrators
    └── simulation.py           # simulation loop and cloth management
```