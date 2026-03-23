import threading
import numpy as np
from vpython import canvas, vector, triangle, vertex, color, sphere, slider, wtext, button, menu, winput
from simulation.integrators import integrator as integ_enum
from simulation.simulation import simulation
import os, signal

DEFAULTS = {
    # cloth_config defaults
    "rows": 18,
    "columns": 12,
    "spacing": 0.2,
    "pin_top_row": True,
    "pin_top_corners": False,
    "pin_left_column": False,
    "pin_right_column": False,
    "pin_all_edges": False,

    # simulation parameter defaults
    "particle_mass": 0.05,
    "k_structural": 200.0,
    "k_shear": 200.0,
    "k_bend": 75.0,
    "damping": 0.8,
    "gravity": np.array([0.0, -9.8, 0.0]),
    "wind_strength": 0.0,
    "wind_angle": 90.0,
    "dt": 0.0075,
    "integrator": "RK4",
    "title": "Cloth Simulation"
}

class renderer:
    def __init__(self, simulation: simulation) -> None:
        self.simulation = simulation
        self.cloth = self.simulation.cloth

        # vpython panel for rendering scene
        self.scene = canvas(
            title=(
                # css to display ui elements on the same line
                f'<b></b><br>'
                '<style>'
                '  #glowscript { display: flex !important; flex-wrap: wrap; }'
                '  canvas { flex: 0 0 auto; }'
                '  .print_output { display: none; }'
                '  div.glowscript { flex-direction: row !important; }'
                '</style>'
            ),
            width = 1080,
            height = 1080,
            center = vector(
                self.cloth.config.columns * self.cloth.config.spacing / 2,
                -self.cloth.config.rows * self.cloth.config.spacing / 2,
                0
            ),
            background = vector(0.2, 0.2, 0.2)
        )

        # rendering objects
        self.vertices = [] # vertex for each particle
        self.triangles = [] # triangular cloth mesh (2 triangles per grid cell)
        self.pin_markers = [] # spheres to visualize pinned particles

        # ui elements
        self.sliders = {}
        self.unpause_next_reset_or_generate = False # flag to auto-resume simulation if paused due to divergence

        # thread lock to prevent race conditions while resetting or rebuilding the cloth
        self._lock = threading.RLock()

        self._build_controls()
        self._build_mesh()



    # mesh handlers

    def _build_mesh(self) -> None:
        """
        builds the scene mesh:
            - builds vertices
            - builds triangles
            - builds pins indicators
        """
        state = self.simulation.state

        # build vertices
        for i in range(self.cloth.num_particles):
            position = self.cloth.get_position(state, i)
            self.vertices.append(
                vertex(
                    pos = vector(position[0], position[1], position[2]),
                    color = color.cyan,
                    normal = vector(0, 0, 1)
                )
            )
        
        # build triangles
        for face in self.cloth.faces:
            i, j, k = face
            self.triangles.append(
                triangle(
                    vs = [self.vertices[i], self.vertices[j], self.vertices[k]]
                )
            )
        
        # build pins
        for i in self.cloth.pinned:
            position = self.cloth.get_position(state, i)
            self.pin_markers.append(
                sphere(
                    pos = vector(position[0], position[1], position[2]),
                    radius = self.cloth.config.spacing * 0.05,
                    color = color.white
                )
            )

    def _destroy_mesh(self) -> None:
        """
        moves old vertices to (0, 0, 0) to create invisible triangles
            since vpython vertices cannot be deleted or hidden.
        makes pin markers invisible
        clears the mesh lists
        """
        for v in self.vertices:
            v.pos = vector(0, 0, 0)
        for pin in self.pin_markers:
            pin.visible = False
        
        self.vertices = []
        self.triangles = []
        self.pin_markers = []

    def _rebuild_mesh(self) -> None:
        """
        rebuilds the scene mesh:
            - destroys the current mesh
            - builds a new mesh
            - re-centers the camera around the new mesh
        """
        with self._lock:
            self._destroy_mesh()
            self._build_mesh()
            self.scene.center = vector(
                self.cloth.config.columns * self.cloth.config.spacing / 2,
                -self.cloth.config.rows * self.cloth.config.spacing / 2,
                0
            )



    # event handlers / parameter updates

    def _on_pause(self, evt) -> None:
        if self.simulation.paused:
            self.simulation.resume()
            self.pause_btn.text = "Pause"
        else:
            self.simulation.pause()
            self.pause_btn.text = "Resume"
 
    def _on_reset(self, evt) -> None:
        with self._lock:
            self.simulation.reset()
            self.cloth = self.simulation.cloth
            self.update()
        # auto unpause simulation if it was paused due to integrator instability
        if self.unpause_next_reset_or_generate:
            self.simulation.resume()
            self.unpause_next_reset_or_generate = False
 
    def _on_quit(self, evt) -> None:
        # delete the scene
        self.scene.delete()
        # kill the process (fully exits python)
        os.kill(os.getpid(), signal.SIGTERM)
 
    def _update_param(self, name: str, value = None, label = None, label_text: str = None) -> None:
        """
        updates any simulation parameter along with its label if provided.
        parameters:
            - 'name : str': string of parameter name
            - 'value': value to update parameter with
            - 'label': optional label to update
            - 'label_text : str': optional text to update label with
        """
        self.simulation.params[name] = value
        if label and label_text:
            label.text = label_text

    def _reset_param(self, name: str, label = None, format_string: str = "{}") -> None:
        """
        resets a simulation parameter to its default value along with its label if provided.
        parameters:
            - 'name : str': string of parameter name
            - 'label': optional label to update
            - 'format_string : str': format string for label
        """
        val = DEFAULTS[name]
        self.simulation.params[name] = val
 
        if name in self.sliders:
            if name == "gravity":
                self.sliders[name].value = val[1]
                label.text = format_string.format(val[1])
            else:
                self.sliders[name].value = val
                label.text = format_string.format(val)
        else:
            display_val = val[1] if isinstance(val, np.ndarray) and name == "gravity" else val
            label.text = format_string.format(display_val)

    def _on_generate(self, evt) -> None:
        """
        generates a new cloth with the given parameters.
        parameters:
            - 'evt : any': event object
        """
        from models.cloth import cloth_config
 
        def read_input(inp, default, min_val = None, max_val = None, as_int = False) -> float | int:
            """
            reads input from a widget.
            parameters:
                - 'inp': widget to read input from
                - 'default': default value to fall back on
                - 'min_val': minimum value
                - 'max_val': maximum value
                - 'as_int': whether to convert to int
            returns:
                - float or int: value from widget
            """
            try:
                # Try reading .text first, because if we programmatically changed .text during reset, 
                # .number is stale.
                val = float(inp.text) if getattr(inp, 'text', None) else float(inp.number)
                if val <= 0:
                    val = default
            except (TypeError, ValueError, AttributeError):
                val = default

            # clamp value to min/max
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)

            # return int if requested
            return int(val) if as_int else val

        # read inputs
        rows = read_input(self.rows_input, DEFAULTS["rows"], 2, 50, as_int = True)
        columns = read_input(self.cols_input, DEFAULTS["columns"], 2, 50, as_int = True)
        spacing = read_input(self.spacing_input, DEFAULTS["spacing"], 0.05, 1.0)
        k_struct = read_input(self.k_struct_input, DEFAULTS["k_structural"], 1.0)
        k_shear = read_input(self.k_shear_input, DEFAULTS["k_shear"], 1.0)
        k_bend = read_input(self.k_bend_input, DEFAULTS["k_bend"], 1.0)
        damping = read_input(self.damping_input, DEFAULTS["damping"], 0.0)
        selected_pin = self.pinned_edges_menu.selected
        pin_top_row = (selected_pin == "Top Row")
        pin_top_corners = (selected_pin == "Top Corners")
        pin_left_column = (selected_pin == "Left Column")
        pin_right_column = (selected_pin == "Right Column")
        pin_all_edges = (selected_pin == "All Edges")

        # update simulation parameters
        self.simulation.params["k_structural"] = k_struct
        self.simulation.params["k_shear"] = k_shear
        self.simulation.params["k_bend"] = k_bend
        self.simulation.params["damping"] = damping

        with self._lock:
            # rebuild cloth
            new_config = cloth_config(
                rows=rows, 
                columns=columns, 
                spacing=spacing,
                pin_top_row=pin_top_row,
                pin_top_corners=pin_top_corners,
                pin_left_column=pin_left_column,
                pin_right_column=pin_right_column,
                pin_all_edges=pin_all_edges,
            )
            self.simulation.rebuild_cloth(new_config)
            self.cloth = self.simulation.cloth
    
            # rebuild cloth mesh
            self._rebuild_mesh()
            self.update()

        # auto unpause simulation if it was paused due to integrator instability
        if self.unpause_next_reset_or_generate:
            self.simulation.resume()
            self.unpause_next_reset_or_generate = False

    def _reset_generation_defaults(self, evt) -> None:
        """
        resets all generation parameters to their default values.
        parameters:
            - 'evt : any': event object
        """
        self.rows_input.text = str(DEFAULTS["rows"])
        self.cols_input.text = str(DEFAULTS["columns"])
        self.spacing_input.text = str(DEFAULTS["spacing"])
        self.k_struct_input.text = str(DEFAULTS["k_structural"])
        self.k_shear_input.text = str(DEFAULTS["k_shear"])
        self.k_bend_input.text = str(DEFAULTS["k_bend"])
        self.damping_input.text = str(DEFAULTS["damping"])
        if DEFAULTS.get("pin_top_row"):
            self.pinned_edges_menu.index = 0
        elif DEFAULTS.get("pin_top_corners"):
            self.pinned_edges_menu.index = 1
        elif DEFAULTS.get("pin_left_column"):
            self.pinned_edges_menu.index = 2
        elif DEFAULTS.get("pin_right_column"):
            self.pinned_edges_menu.index = 3
        elif DEFAULTS.get("pin_all_edges"):
            self.pinned_edges_menu.index = 4
        else:
            self.pinned_edges_menu.index = 0



    # canvas updates
    
    def update(self) -> None:
        """
        updates the cloth mesh to match the simulation state.
        """
        with self._lock:
            state = self.simulation.state

            # pause simulation if state array contains NaN, infinity, or extremely large values
            if not np.isfinite(state).all() or np.max(np.abs(state)) > 10000.0:
                if not self.simulation.paused:
                    self.simulation.pause()
                    # set flag to auto-resume simulation after next reset/generate
                    self.unpause_next_reset_or_generate = True
                return

            # update vertices
            for i in range(self.cloth.num_particles):
                position = self.cloth.get_position(state, i)
                self.vertices[i].pos = vector(position[0], position[1], position[2])

            # update normals for faces
            for face, triangle in zip(self.cloth.faces, self.triangles):
                i, j, k = face
                p0 = self.vertices[i].pos
                p1 = self.vertices[j].pos
                p2 = self.vertices[k].pos
                edge1 = p1 - p0
                edge2 = p2 - p0
                normal = edge1.cross(edge2)
                if normal.mag > 0.000001:
                    normal = normal.hat
                self.vertices[i].normal = normal
                self.vertices[j].normal = normal
                self.vertices[k].normal = normal
    
    

    # controls

    def _build_controls(self) -> None:
        """
        builds the control panel for the simulation.
        """
        params = self.simulation.params
        s = self.scene
 
        # simulation controls
        s.append_to_caption("<b>Simulation Controls</b>\n\n")
        self.pause_btn = button(text = "Pause", bind = self._on_pause)
        s.append_to_caption(" ")
        button(text = "Reset Cloth", bind = self._on_reset)
        s.append_to_caption("  ")
        button(text = "Quit", bind = self._on_quit)
        s.append_to_caption("\n\n")
 
        # gravity
        s.append_to_caption("Gravity: ")
        self.gravity_label = wtext(text = f"{params['gravity'][1]:.1f}")
        s.append_to_caption("  ")
        button(text = "Reset", bind = lambda e: 
            self._reset_param("gravity", self.gravity_label, "{:.1f}")
        )
        s.append_to_caption("\n")
        self.sliders["gravity"] = slider(
            min = -100.0, max = 0.0, value = params["gravity"][1],
            length = 300, bind = lambda sl: 
                (self._update_param("gravity", np.array([0.0, sl.value, 0.0]), self.gravity_label, f"{sl.value:.1f}"))
        )
        s.append_to_caption("\n\n")
 
        # particle mass
        s.append_to_caption("Particle Mass: ")
        self.mass_label = wtext(text = f"{params['particle_mass']:.3f}")
        s.append_to_caption("  ")
        button(text = "Reset", bind = lambda e: 
            self._reset_param("particle_mass", self.mass_label, "{:.3f}"))
        s.append_to_caption("\n")
        self.sliders["particle_mass"] = slider(
            min = 0.05, max = 0.5, value = params["particle_mass"], step = 0.001, length = 300,
            bind = lambda sl, n = "particle_mass", l = None: 
                self._update_param(n, sl.value, self.mass_label, f"{sl.value:.3f}")
        )
        s.append_to_caption("\n\n")
 
        # wind strength
        s.append_to_caption("Wind Strength: ")
        self.wind_strength_label = wtext(text = f"{params['wind_strength']:.1f}")
        s.append_to_caption("  ")
        button(text = "Reset", bind = lambda e: self._reset_param(
            "wind_strength", self.wind_strength_label, "{:.1f}"
        ))
        s.append_to_caption("\n")
        self.sliders["wind_strength"] = slider(
            min = 0.0, max = 10.0, value = params["wind_strength"],
            step = 0.1, length = 300,
            bind = lambda sl, n = "wind_strength", l = None: 
                self._update_param(n, sl.value, self.wind_strength_label, f"{sl.value:.1f}")
        )
        s.append_to_caption("\n\n")
 
        # wind angle (X, Z plane)
        s.append_to_caption("Wind Angle: ")
        self.wind_angle_label = wtext(text = f"{params['wind_angle']:.0f}°")
        s.append_to_caption("(X & Z plane)")
        s.append_to_caption("  ")
        button(text = "Reset", bind = lambda e: self._reset_param(
            "wind_angle", self.wind_angle_label, "{:.0f}°"
        ))
        s.append_to_caption("\n")
        self.sliders["wind_angle"] = slider(
            min = 0, max = 360, value = params["wind_angle"],
            step = 5, length = 300,
            bind = lambda sl, n = "wind_angle", l = None: 
                self._update_param(n, sl.value, self.wind_angle_label, f"{sl.value:.0f}°")
        )
        s.append_to_caption("\n\n")
 
        # integrator
        s.append_to_caption("Integrator: (Euler requires very small time-steps for stability)")
        s.append_to_caption("\n")
        menu(
            choices = ["RK4", "EULER"],
            index = 0 if params["integrator"] == integ_enum.RK4 else 1,
            bind = lambda m, n = "integrator", l = None: 
                self._update_param(
                    n,
                    integ_enum.RK4 if m.selected == "RK4" else integ_enum.EULER,
                    l,
                    m.selected
                )
        )
        s.append_to_caption("\n\n\n")

        # time step
        s.append_to_caption("Time Step (dt): ")
        self.dt_label = wtext(text = f"{params['dt']:.4f}")
        s.append_to_caption("  ")
        button(text = "Reset", bind = lambda e: 
            self._reset_param("dt", self.dt_label, "{:.4f}")
        )
        s.append_to_caption("\n")
        self.sliders["dt"] = slider(
            min = 0.0001, max = 0.02, value = params["dt"],
            step = 0.001, length = 300,
            bind = lambda sl, n = "dt", l = None: 
                self._update_param(n, sl.value, self.dt_label, f"{sl.value:.4f}")
        )
        s.append_to_caption("\n\n")
 

        # cloth generation section
        s.append_to_caption("─" * 40 + "\n")
        s.append_to_caption("<b>Cloth Generation</b>\n")
        s.append_to_caption("Design and generate a custom cloth\n\n")

        button(text = "Generate Cloth", bind = self._on_generate)
        s.append_to_caption("  ")
        button(text = "Reset Parameters", bind = self._reset_generation_defaults)
        s.append_to_caption("\n\n")
 
        s.append_to_caption("Rows: ")
        self.rows_input = winput(type = "numeric", bind = lambda e: None, width = 50, text = str(self.cloth.config.rows))
        s.append_to_caption("  ")
 
        s.append_to_caption("Columns: ")
        self.cols_input = winput(type = "numeric", bind = lambda e: None, width = 50, text = str(self.cloth.config.columns))
        s.append_to_caption("  ")
 
        s.append_to_caption("Particle Spacing: ")
        self.spacing_input = winput(type = "numeric", bind = lambda e: None, width = 50, text = str(self.cloth.config.spacing))
        s.append_to_caption("\n\n")
 
        s.append_to_caption("Structural Spring Constant: ")
        self.k_struct_input = winput(type = "numeric", bind = lambda e: None, width = 50, text = str(params["k_structural"]))
        s.append_to_caption("  ")
 
        s.append_to_caption("Shear Spring Constant: ")
        self.k_shear_input = winput(type = "numeric", bind = lambda e: None, width = 50, text = str(params["k_shear"]))
        s.append_to_caption("  ")
 
        s.append_to_caption("Bending Spring Constant: ")
        self.k_bend_input = winput(type = "numeric", bind = lambda e: None, width = 50, text = str(params["k_bend"]))
        s.append_to_caption("\n\n")
 
        s.append_to_caption("Damping Constant: ")
        self.damping_input = winput(type = "numeric", bind = lambda e: None, width = 50, text = str(params["damping"]))
        s.append_to_caption("\n")

        # pinned edges dropdown options
        s.append_to_caption("Pinned Edges: ")
        s.append_to_caption("\n")
        initial_index = 0
        if getattr(self.cloth.config, "pin_top_corners", False):
            initial_index = 1
        elif getattr(self.cloth.config, "pin_left_column", False):
            initial_index = 2
        elif getattr(self.cloth.config, "pin_right_column", False):
            initial_index = 3
        elif getattr(self.cloth.config, "pin_all_edges", False):
            initial_index = 4
        self.pinned_edges_menu = menu(
            choices=["Top Row", "Top Corners", "Left Column", "Right Column", "All Edges"],
            index=initial_index,
            bind=lambda m: None
        )
