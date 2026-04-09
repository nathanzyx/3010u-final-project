"""
File: models/cloth.py
Author: Nathan Tandory <nathtand@gmail.com>
Date: 2026-04-09
Description: Cloth class for the 3D mass-spring-damper cloth simulation.
"""

import numpy as np

class cloth:
    def __init__(self, config: dict, sim_params: dict) -> None:
        self.config = config
        self.num_particles = self.config["rows"] * self.config["columns"]

        self.pinned = set() # indices of immovable particles
        self.faces = [] # tuples of 3 particle indices, triangles for rendering

        # pin particles
        for row in range(self.config["rows"]):
            for column in range(self.config["columns"]):
                index = self.get_1d_index(row, column)
                # pin top row
                if self.config["pin_top_row"] and row == 0: self.pinned.add(index)
                # pin top corners
                if (self.config["pin_top_corners"] and row == 0 and column == 0) or \
                   (self.config["pin_top_corners"] and row == 0 and column == self.config["columns"] - 1): self.pinned.add(index)
                # pin left row
                if self.config["pin_left_column"] and column == 0: self.pinned.add(index)
                if self.config["pin_right_column"] and column == self.config["columns"] - 1: self.pinned.add(index)
                # pin all edges
                if self.config["pin_all_edges"] and (row == 0 or row == self.config["rows"] - 1 or column == 0 or column == self.config["columns"] - 1): self.pinned.add(index)

        self.build_springs(sim_params)
        self.build_faces()



    # index helpers

    def get_position(self, state: np.ndarray, idx: int) -> np.ndarray:
        """
        returns the position (3D vector) of a particle given its index in the state vector
        """
        offset = idx * 6
        return state[offset : offset + 3]
        
    def get_velocity(self, state: np.ndarray, idx: int) -> np.ndarray:
        """
        returns the velocity (3D vector) of a particle given its index in the state vector
        """
        offset = idx * 6
        return state[offset + 3 : offset + 6]
        
    def get_1d_index(self, row: int, column: int) -> int:
        """
        returns the flattened index location given 2D input row and column
        """
        return row * self.config["columns"] + column



    # state builders
    
    def build_initial_state(self) -> np.ndarray:
        """
        builds a flat state vector to initialize the cloth state.
        6 state variables per particle: x, y, z, v_x, v_y, v_z.
        all velocities are initialized to 0.
        """
        # initialize 6 state variables set to 0.0 per particle
        state = np.zeros(self.num_particles * 6, 'float64')

        for row in range(self.config["rows"]):
            for column in range(self.config["columns"]):
                index = row * self.config["columns"] + column
                offset = index * 6
                # initialize particle position (velocities already 0.0)
                state[offset] = column * self.config["spacing"] # x
                state[offset + 1] = -(row * self.config["spacing"]) # y
                state[offset + 2] = 0.0 # z
        return state
        
    def build_springs(self, sim_params: dict) -> None:
        """
        builds all the springs for the cloth.
        """
        spacing = self.config["spacing"]
        diag_rest = spacing * np.sqrt(2.0)
        bend_rest = 2.0 * spacing
        
        k_structural = sim_params["k_structural"]
        k_shear = sim_params["k_shear"]
        k_bend = sim_params["k_bend"]
        damping = sim_params["damping"]

        a_list = [] # particle index for spring endpoint a
        b_list = [] # particle index for spring endpoint b
        k_list = [] # spring constants
        rest_list = [] # rest lengths
        damping_list = [] # damping coefficients

        def _add_spring(a: int, b: int, rest_length: float, k: float, damping: float) -> None:
            """
            helper function to add a spring between particles a and b.
            """
            a_list.append(a)
            b_list.append(b)
            rest_list.append(rest_length)
            k_list.append(k)
            damping_list.append(damping)
        
        for row in range(self.config["rows"]):
            for column in range(self.config["columns"]):
                index = self.get_1d_index(row, column)
                
                # structural springs
                if column + 1 < self.config["columns"]:
                    other = self.get_1d_index(row, column + 1)
                    _add_spring(index, other, spacing, k_structural, damping)
                if row + 1 < self.config["rows"]:
                    other = self.get_1d_index(row + 1, column)
                    _add_spring(index, other, spacing, k_structural, damping)
                
                # shear springs
                if row + 1 < self.config["rows"] and column + 1 < self.config["columns"]:
                    other = self.get_1d_index(row + 1, column + 1)
                    _add_spring(index, other, diag_rest, k_shear, damping)
                if row + 1 < self.config["rows"] and column - 1 >= 0:
                    other = self.get_1d_index(row + 1, column - 1)
                    _add_spring(index, other, diag_rest, k_shear, damping)

                # bending springs
                if column + 2 < self.config["columns"]:
                    other = self.get_1d_index(row, column + 2)
                    _add_spring(index, other, bend_rest, k_bend, damping)
                if row + 2 < self.config["rows"]:
                    other = self.get_1d_index(row + 2, column)
                    _add_spring(index, other, bend_rest, k_bend, damping)

        self._spring_a = np.array(a_list, np.intp)
        self._spring_b = np.array(b_list, np.intp)
        self._spring_k = np.array(k_list, 'float64')
        self._spring_rest = np.array(rest_list, 'float64')
        self._spring_damping = np.array(damping_list, 'float64')
        self._pinned_mask = np.array([i in self.pinned for i in range(self.num_particles)], bool)
                
    def build_faces(self) -> None:
        """
        build 2 triangle faces for each grid cell of 4 particles.
        this is used only for rendering.
        """
        for row in range(self.config["rows"] - 1):
            for column in range(self.config["columns"] - 1):
                top_left = self.get_1d_index(row, column)
                top_right = self.get_1d_index(row, column + 1)
                bottom_left = self.get_1d_index(row + 1, column)
                bottom_right = self.get_1d_index(row + 1, column + 1)
                self.faces.append((top_left, top_right, bottom_left))
                self.faces.append((top_right, bottom_right, bottom_left))
    