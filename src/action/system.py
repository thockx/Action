"""System orchestration: topology, configuration, energies, trajectory, visuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from .components import AttachmentPoint, Mass, Rod, Spring, Wall
from .configuration import Configuration
from .connections import Connection, Gravity, Hinge
from .context import pop_system, push_system
from .coordinates import Coordinate, CoordinateRate

if TYPE_CHECKING:
    from .visualizations import MassVector

try:
    from manim import (
        Animation,
        Arrow,
        BLACK,
        Circle,
        Line,
        MathTex,
        VMobject,
        VGroup,
        WHITE,
        config,
    )

    MANIM_AVAILABLE = True
    config.background_color = WHITE

except ImportError:  # pragma: no cover
    MANIM_AVAILABLE = False

    class VGroup:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass


@dataclass(frozen=True)
class Trajectory:
    """Cached numerical generalized-coordinate state history."""

    time: np.ndarray
    state: np.ndarray

    def state_at(self, time: float) -> np.ndarray:
        return np.array(
            [np.interp(time, self.time, row) for row in self.state]
        )


class System(VGroup):
    """A Manim-compatible 2D mechanism with one shared Lagrangian pipeline."""

    def __init__(
        self,
        objects: list[object] | None = None,
        initial: Mapping[Coordinate | CoordinateRate, float] | None = None,
        fields: list[Gravity] | None = None,
        show_equations: bool = False,
    ) -> None:
        super().__init__()

        self.objects = list(objects or [])
        self.initial = dict(initial or {})
        self.fields = list(fields) if fields is not None else []
        self.show_equations = show_equations

        self._uses_active_gravity = fields is None
        self._context_connections: list[Connection] = []

        # Vector visualizations.
        self._visualizations: list["MassVector"] = []
        self._vector_mobjects: dict["MassVector", object] = {}
        self._vector_scales: dict["MassVector", float] = {}
        self._vector_max_magnitudes: dict["MassVector", float] = {}
        self._vector_samples: dict["MassVector", np.ndarray] = {}

        self._context_token = None
        self._compiled = False

        if objects is not None:
            self._compile()

    # ------------------------------------------------------------------
    # Context-manager API
    # ------------------------------------------------------------------

    def __enter__(self) -> "System":
        if self._compiled:
            raise RuntimeError(
                "A compiled System cannot be used as a definition context."
            )

        self._context_token = push_system(self)
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:  # type: ignore[no-untyped-def]

        if self._context_token is not None:
            pop_system(self._context_token)
            self._context_token = None

        if exc_type is None:
            self._compile()

        return False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_object(self, component: object) -> None:
        if self._compiled:
            raise RuntimeError(
                "Cannot add components after a System has been compiled."
            )

        self.objects.append(component)

    def _register_connection(self, connection: Connection) -> None:
        if self._compiled:
            raise RuntimeError(
                "Cannot add connections after a System has been compiled."
            )

        self._context_connections.append(connection)

    def _register_visualization(
        self,
        visualization: "MassVector",
    ) -> None:
        if self._compiled:
            raise RuntimeError(
                "Cannot add visualizations after a System has been compiled."
            )

        self._visualizations.append(visualization)

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        if self._compiled:
            return

        if self._uses_active_gravity:
            self.fields = [Gravity.active()]

        self.masses = [
            item for item in self.objects
            if isinstance(item, Mass)
        ]

        for visualization in self._visualizations:
            if visualization.mass not in self.masses:
                raise ValueError(
                    "Mass vector visualization targets a Mass "
                    "outside this System."
                )

        self.rods = [
            item for item in self.objects
            if isinstance(item, Rod)
        ]

        self.springs = [
            item for item in self.objects
            if isinstance(item, Spring)
        ]

        self.walls = [
            item for item in self.objects
            if isinstance(item, Wall)
        ]

        self.connections = self._collect_connections()
        self._node_by_point = self._make_nodes()

        self.configuration = Configuration(self)

        self._trajectory: Trajectory | None = None

        self._scale = 1.0
        self._origin = np.zeros(2)

        self._validate_initial_conditions()

        (
            self._q,
            self._dq,
            self._kinetic,
            self._potential,
        ) = self._energies()

        (
            self._mass_matrix,
            self._bias,
        ) = self._equation_terms()

        self._mass_matrix_fn = sp.lambdify(
            [*self._q, *self._dq],
            self._mass_matrix,
            "numpy",
        )

        self._bias_fn = sp.lambdify(
            [*self._q, *self._dq],
            self._bias,
            "numpy",
        )

        if MANIM_AVAILABLE:
            self._build_visuals()
            self._update_visuals(0.0)

        self._compiled = True

    # ------------------------------------------------------------------
    # Public physics interface
    # ------------------------------------------------------------------

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        """All exposed intrinsic coordinates, including spring extensions."""

        return (
            *self.configuration.intrinsic_coordinates,
            *(spring.extension for spring in self.springs),
        )

    @property
    def degrees_of_freedom(self) -> int:
        return len(self._q)

    def equations_of_motion(self) -> tuple[sp.Equality, ...]:
        time = sp.symbols("t", real=True)

        functions = [
            sp.Function(str(value))(time)
            for value in self._q
        ]

        mapping = (
            dict(zip(self._q, functions))
            | dict(
                zip(
                    self._dq,
                    [sp.diff(value, time) for value in functions],
                )
            )
        )

        lagrangian = (
            self._kinetic - self._potential
        ).subs(mapping)

        return tuple(
            sp.Eq(
                sp.simplify(
                    sp.diff(
                        sp.diff(
                            lagrangian,
                            sp.diff(value, time),
                        ),
                        time,
                    )
                    - sp.diff(lagrangian, value)
                ),
                0,
            )
            for value in functions
        )

    # ------------------------------------------------------------------
    # Numerical simulation
    # ------------------------------------------------------------------

    def solve(
        self,
        duration: float,
        samples_per_second: int = 120,
    ) -> Trajectory:

        if duration <= 0:
            raise ValueError(
                "Simulation duration must be positive."
            )

        time = np.linspace(
            0.0,
            duration,
            max(
                2,
                int(duration * samples_per_second) + 1,
            ),
        )

        solution = solve_ivp(
            self._derivative,
            (0.0, duration),
            self._initial_state(),
            t_eval=time,
            rtol=1e-8,
            atol=1e-10,
        )

        if not solution.success:
            raise RuntimeError(
                f"Unable to solve system: {solution.message}"
            )

        self._trajectory = Trajectory(
            solution.t,
            solution.y,
        )

        self._fit_view()
        self._fit_vector_scales()

        return self._trajectory

    def simulate(self, duration: float):  # type: ignore[no-untyped-def]
        self.solve(duration)

        if not MANIM_AVAILABLE:
            raise RuntimeError(
                "simulate() requires the optional "
                "'action[manim]' dependency."
            )

        return _SimulationAnimation(
            self,
            duration,
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def geometry_at(
        self,
        time: float,
    ) -> dict[object, tuple[float, float]]:

        state = (
            self._trajectory.state_at(time)
            if self._trajectory is not None
            else self._initial_state()
        )

        return self.configuration.endpoint_positions(
            state[: len(self._q)]
        )

    def _points(self) -> list[AttachmentPoint]:
        points: list[AttachmentPoint] = []

        for item in self.objects:
            if isinstance(item, (Mass, Wall)):
                points.append(item.attachment)

            elif isinstance(item, (Rod, Spring)):
                points.extend(
                    (
                        item.start,
                        item.end,
                    )
                )

        return points

    def _collect_connections(self) -> list[Connection]:
        found: list[Connection] = []

        for point in self._points():
            for connection in point.connections:
                if connection not in found:
                    found.append(connection)

        return found

    def _make_nodes(self) -> dict[int, int]:
        points = self._points()

        parent = {
            id(point): id(point)
            for point in points
        }

        def find(point: AttachmentPoint) -> int:
            identifier = id(point)

            while parent[identifier] != identifier:
                parent[identifier] = parent[
                    parent[identifier]
                ]

                identifier = parent[identifier]

            return identifier

        for connection in self.connections:
            parent[
                find(connection.first)
            ] = find(connection.second)

        return {
            id(point): find(point)
            for point in points
        }

    def _node(self, point: AttachmentPoint) -> int:
        return self._node_by_point[id(point)]

    # ------------------------------------------------------------------
    # Wall layout
    # ------------------------------------------------------------------

    def _layout_walls(self) -> dict[Wall, np.ndarray]:
        average_length = float(
            np.mean(
                [
                    spring.rest_length
                    for spring in self.springs
                ]
                or [1.0]
            )
        )

        vertical_layout = bool(
            self.walls
            and self.walls[0].orientation == "horizontal"
        )

        return {
            wall: np.array(
                wall.position
                if wall.position is not None
                else (
                    (0.0, -2.0 * average_length * index)
                    if vertical_layout
                    else (2.0 * average_length * index, 0.0)
                ),
                dtype=float,
            )
            for index, wall in enumerate(self.walls)
        }

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def _validate_initial_conditions(self) -> None:
        if not self.masses and not self.rods:
            raise ValueError(
                "System needs at least one Mass or Rod."
            )

        allowed = set(self.coordinates)

        allowed |= {
            coordinate.rate
            for coordinate in self.coordinates
        }

        invalid = set(self.initial).difference(allowed)

        if invalid:
            names = ", ".join(
                getattr(value, "name", repr(value))
                for value in invalid
            )

            raise ValueError(
                "Initial conditions are not exposed "
                f"intrinsic coordinates: {names}."
            )

    def _initial_state(self) -> np.ndarray:
        q = np.zeros(len(self._q))

        for index, spec in enumerate(
            self.configuration.specs
        ):
            if spec.intrinsic is not None:
                q[index] = float(
                    self.initial.get(
                        spec.intrinsic,
                        0.0,
                    )
                )

        self._seed_free_mass_positions(q)
        self._check_spring_extensions(q)

        dq = self._initial_rates(q)

        return np.concatenate(
            (
                q,
                dq,
            )
        )

    # ------------------------------------------------------------------
    # Initial wall positions
    # ------------------------------------------------------------------

    def _resolve_initial_wall_positions(self) -> None:
        """Fit wall anchors to requested initial spring lengths."""

        q = np.zeros(len(self._q))

        for index, spec in enumerate(
            self.configuration.specs
        ):
            if spec.intrinsic is not None:
                q[index] = float(
                    self.initial.get(
                        spec.intrinsic,
                        0.0,
                    )
                )

        self._seed_free_mass_positions(q)

        geometry = self.configuration.endpoint_positions(q)

        for spring in self.springs:
            if spring.extension not in self.initial:
                continue

            start_node = self._node(spring.start)
            end_node = self._node(spring.end)

            wall = next(
                (
                    candidate
                    for candidate in self.walls
                    if self._node(candidate.attachment)
                    in (start_node, end_node)
                ),
                None,
            )

            if wall is None:
                continue

            wall_point = (
                spring.start
                if self._node(wall.attachment) == start_node
                else spring.end
            )

            other_point = (
                spring.end
                if wall_point is spring.start
                else spring.start
            )

            wall_position = np.array(
                geometry[wall_point]
            )

            other_position = np.array(
                geometry[other_point]
            )

            requested_length = (
                spring.rest_length
                + float(
                    self.initial[spring.extension]
                )
            )

            separation = (
                wall_position - other_position
            )

            distance = np.linalg.norm(
                separation
            )

            if np.isclose(
                distance - spring.rest_length,
                self.initial[spring.extension],
                atol=1e-7,
            ):
                continue

            direction = (
                separation / distance
                if distance
                else np.array((1.0, 0.0))
            )

            adjusted_position = (
                other_position
                + requested_length * direction
            )

            self.configuration.set_wall_position(
                wall,
                adjusted_position,
            )

            print(
                "Action: adjusted Wall position to "
                f"({adjusted_position[0]:.6g}, "
                f"{adjusted_position[1]:.6g}) "
                "to satisfy initial "
                f"{spring.extension.name}="
                f"{self.initial[spring.extension]:.6g}."
            )

    # ------------------------------------------------------------------
    # Free-mass initialization
    # ------------------------------------------------------------------

    def _seed_free_mass_positions(
        self,
        q: np.ndarray,
    ) -> None:

        free = self.configuration.free_masses

        if not free:
            return

        symbolic = (
            self.configuration.symbolic_positions()
        )

        known: dict[int, np.ndarray] = {}

        hinge_substitutions = dict(
            zip(
                self._q,
                q,
            )
        )

        for node, position in symbolic.items():
            if not position.free_symbols.intersection(
                self._q
            ):
                known[node] = np.array(
                    position,
                    dtype=float,
                ).reshape(2)

            elif not position.free_symbols.intersection(
                set(
                    self._q[
                        len(
                            self.configuration.hinges
                        ):
                    ]
                )
            ):
                known[node] = np.array(
                    [
                        float(
                            value.subs(
                                hinge_substitutions
                            )
                        )
                        for value in position
                    ]
                )

        for mass in free:
            node = self._node(
                mass.attachment
            )

            anchors: list[
                tuple[
                    np.ndarray,
                    float,
                    np.ndarray,
                ]
            ] = []

            for spring in self.springs:
                start = self._node(
                    spring.start
                )

                end = self._node(
                    spring.end
                )

                if node not in (start, end):
                    continue

                other = (
                    end
                    if node == start
                    else start
                )

                if other not in known:
                    continue

                wall = next(
                    (
                        candidate
                        for candidate in self.walls
                        if self._node(
                            candidate.attachment
                        ) == other
                    ),
                    None,
                )

                axis = (
                    np.array((0.0, -1.0))
                    if wall is not None
                    and wall.orientation == "horizontal"
                    else np.array((1.0, 0.0))
                )

                direction = (
                    -axis
                    if node == start
                    else axis
                )

                anchors.append(
                    (
                        known[other],
                        spring.rest_length
                        + float(
                            self.initial.get(
                                spring.extension,
                                0.0,
                            )
                        ),
                        direction,
                    )
                )

            if len(anchors) >= 2:
                position = self._circle_intersection(
                    (
                        anchors[0][0],
                        anchors[0][1],
                    ),
                    (
                        anchors[1][0],
                        anchors[1][1],
                    ),
                )

            elif anchors:
                anchor, radius, direction = anchors[0]

                position = (
                    anchor
                    + direction * radius
                )

            else:
                position = np.zeros(2)

            start = self.configuration.symbols.index(
                self.configuration._mass_symbols[mass][0]
            )

            q[start:start + 2] = position
            known[node] = position

    @staticmethod
    def _circle_intersection(
        first: tuple[np.ndarray, float],
        second: tuple[np.ndarray, float],
    ) -> np.ndarray:

        center_one, radius_one = first
        center_two, radius_two = second

        delta = (
            center_two - center_one
        )

        distance = np.linalg.norm(delta)

        if (
            distance == 0
            or distance
            > radius_one + radius_two + 1e-8
            or distance
            < abs(radius_one - radius_two) - 1e-8
        ):
            raise ValueError(
                "Spring extension initial conditions "
                "are geometrically inconsistent."
            )

        along = (
            radius_one**2
            - radius_two**2
            + distance**2
        ) / (2.0 * distance)

        height = max(
            radius_one**2 - along**2,
            0.0,
        ) ** 0.5

        direction = (
            delta / distance
        )

        return (
            center_one
            + along * direction
            + height
            * np.array(
                (
                    -direction[1],
                    direction[0],
                )
            )
        )

    # ------------------------------------------------------------------
    # Springs
    # ------------------------------------------------------------------

    def _spring_extensions(
        self,
        q: np.ndarray,
    ) -> np.ndarray:

        geometry = (
            self.configuration.endpoint_positions(q)
        )

        return np.array(
            [
                np.linalg.norm(
                    np.array(
                        geometry[spring.end]
                    )
                    - np.array(
                        geometry[spring.start]
                    )
                )
                - spring.rest_length
                for spring in self.springs
            ]
        )

    def _check_spring_extensions(
        self,
        q: np.ndarray,
    ) -> None:

        extensions = self._spring_extensions(q)

        for spring, extension in zip(
            self.springs,
            extensions,
        ):
            if (
                spring.extension in self.initial
                and not np.isclose(
                    extension,
                    self.initial[
                        spring.extension
                    ],
                    atol=1e-7,
                )
            ):
                raise ValueError(
                    "Initial extension for "
                    f"{spring.extension.name} "
                    "conflicts with topology."
                )

    def _initial_rates(
        self,
        q: np.ndarray,
    ) -> np.ndarray:

        dq = np.zeros(len(self._q))

        for index, spec in enumerate(
            self.configuration.specs
        ):
            if spec.intrinsic is not None:
                dq[index] = float(
                    self.initial.get(
                        spec.intrinsic.rate,
                        0.0,
                    )
                )

        constrained = [
            spring
            for spring in self.springs
            if spring.extension.rate
            in self.initial
        ]

        if not constrained:
            return dq

        epsilon = 1e-6

        jacobian = np.column_stack(
            [
                (
                    self._spring_extensions(
                        q
                        + epsilon
                        * np.eye(len(q))[index]
                    )
                    - self._spring_extensions(
                        q
                        - epsilon
                        * np.eye(len(q))[index]
                    )
                )
                / (2 * epsilon)
                for index in range(len(q))
            ]
        )

        rows = np.array(
            [
                jacobian[
                    self.springs.index(spring)
                ]
                for spring in constrained
            ]
        )

        targets = np.array(
            [
                float(
                    self.initial[
                        spring.extension.rate
                    ]
                )
                for spring in constrained
            ]
        )

        fixed = len(
            self.configuration.hinges
        )

        residual = (
            targets
            - rows[:, :fixed]
            @ dq[:fixed]
        )

        if fixed == len(q):
            if not np.allclose(
                residual,
                0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "Spring extension rates are "
                    "inconsistent with the hinge rates."
                )

            return dq

        solution, _, _, _ = np.linalg.lstsq(
            rows[:, fixed:],
            residual,
            rcond=None,
        )

        if not np.allclose(
            rows[:, fixed:] @ solution,
            residual,
            atol=1e-6,
        ):
            raise ValueError(
                "Spring extension rates are "
                "geometrically inconsistent."
            )

        dq[fixed:] = solution

        return dq

    # ------------------------------------------------------------------
    # Lagrangian
    # ------------------------------------------------------------------

    def _energies(
        self,
    ) -> tuple[
        list[sp.Symbol],
        list[sp.Symbol],
        sp.Expr,
        sp.Expr,
    ]:

        q = self.configuration.symbols

        dq = [
            sp.Symbol(
                f"d_{coordinate}",
                real=True,
            )
            for coordinate in q
        ]

        positions = (
            self.configuration.symbolic_positions()
        )

        kinetic = sp.Integer(0)
        potential = sp.Integer(0)

        for mass in self.masses:
            position = positions[
                self._node(mass.attachment)
            ]

            velocity = sum(
                (
                    position.diff(coordinate)
                    * rate
                    for coordinate, rate
                    in zip(q, dq)
                ),
                sp.zeros(2, 1),
            )

            kinetic += (
                mass.m
                * velocity.dot(velocity)
                / 2
            )

            potential += (
                mass.m
                * self.fields[0].g
                * position[1]
            )

        for spring in self.springs:
            delta = (
                positions[
                    self._node(spring.end)
                ]
                - positions[
                    self._node(spring.start)
                ]
            )

            potential += (
                spring.k
                * (
                    sp.sqrt(
                        delta.dot(delta)
                    )
                    - spring.rest_length
                ) ** 2
                / 2
            )

        return (
            q,
            dq,
            kinetic,
            potential,
        )

    def _equation_terms(
        self,
    ) -> tuple[sp.Matrix, sp.Matrix]:

        matrix = sp.hessian(
            self._kinetic,
            self._dq,
        )

        bias = sp.Matrix(
            [
                sum(
                    sp.diff(
                        sp.diff(
                            self._kinetic,
                            speed,
                        ),
                        coordinate,
                    )
                    * rate
                    for coordinate, rate
                    in zip(
                        self._q,
                        self._dq,
                    )
                )
                - sp.diff(
                    self._kinetic,
                    coordinate,
                )
                + sp.diff(
                    self._potential,
                    coordinate,
                )
                for speed, coordinate
                in zip(
                    self._dq,
                    self._q,
                )
            ]
        )

        return matrix, bias

    def _derivative(
        self,
        _: float,
        state: np.ndarray,
    ) -> np.ndarray:

        count = len(self._q)

        q = state[:count]
        dq = state[count:]

        matrix = np.array(
            self._mass_matrix_fn(
                *q,
                *dq,
            ),
            dtype=float,
        )

        bias = np.array(
            self._bias_fn(
                *q,
                *dq,
            ),
            dtype=float,
        ).reshape(-1)

        ddq = np.linalg.solve(
            matrix,
            -bias,
        )

        return np.concatenate(
            (
                dq,
                ddq,
            )
        )

    # ------------------------------------------------------------------
    # Camera / fitting
    # ------------------------------------------------------------------

    def _fit_view(self) -> None:
        if self._trajectory is None:
            return

        points = np.array(
            [
                point
                for time in self._trajectory.time
                for point in self.geometry_at(
                    float(time)
                ).values()
            ]
        )

        low = points.min(axis=0)
        high = points.max(axis=0)

        self._origin = (
            low + high
        ) / 2

        self._scale = (
            5.5
            / max(
                *(high - low),
                1.0,
            )
        )

    # ------------------------------------------------------------------
    # Vector calculations
    # ------------------------------------------------------------------

    def _mass_position(
        self,
        mass: Mass,
        time: float,
    ) -> np.ndarray:

        return np.array(
            self.geometry_at(time)[mass],
            dtype=float,
        )

    def _motion_vector(
        self,
        visualization: "MassVector",
        time: float,
    ) -> np.ndarray:

        if self._trajectory is None or visualization not in self._vector_samples:
            return np.zeros(2)

        samples = self._vector_samples[visualization]
        return np.array(
            [
                np.interp(time, self._trajectory.time, samples[:, axis])
                for axis in range(2)
            ]
        )

    def _fit_vector_scales(self) -> None:
        if self._trajectory is None:
            return

        for visualization in self._visualizations:
            positions = np.array(
                [
                    self.geometry_at(float(time))[visualization.mass]
                    for time in self._trajectory.time
                ],
                dtype=float,
            )
            velocities = np.gradient(
                positions,
                self._trajectory.time,
                axis=0,
                edge_order=2,
            )
            accelerations = np.gradient(
                velocities,
                self._trajectory.time,
                axis=0,
                edge_order=2,
            )
            samples = (
                velocities
                if visualization.quantity == "velocity"
                else accelerations
                if visualization.quantity == "acceleration"
                else visualization.mass.m * accelerations
            )
            self._vector_samples[visualization] = samples
            magnitudes = np.linalg.norm(samples, axis=1)

            maximum = max(
                magnitudes,
                default=0.0,
            )

            self._vector_max_magnitudes[visualization] = maximum

            if maximum > 1e-8:
                self._vector_scales[visualization] = 2.0 / maximum
            else:
                self._vector_scales[visualization] = 0.0

    # ------------------------------------------------------------------
    # Physical -> Manim coordinate transform
    # ------------------------------------------------------------------

    def _visual_point(
        self,
        point: tuple[float, float],
    ) -> np.ndarray:

        transformed = (
            self._scale
            * (
                np.array(point)
                - self._origin
            )
        )

        return np.array(
            (
                transformed[0],
                transformed[1],
                0.0,
            )
        )

    # ------------------------------------------------------------------
    # Visual creation
    # ------------------------------------------------------------------

    def _build_visuals(self) -> None:

        # --------------------------------------------------------------
        # Masses
        # --------------------------------------------------------------

        self._mass_mobjects = {
            mass: VGroup(
                Circle(
                    radius=0.25,
                    color=BLACK,
                    fill_color=WHITE,
                    fill_opacity=1,
                ),
                MathTex(
                    mass.label,
                    color=BLACK,
                ).scale(0.55),
            )
            for mass in self.masses
        }

        # --------------------------------------------------------------
        # Rods
        # --------------------------------------------------------------

        self._rod_mobjects = {
            rod: VGroup(
                Line(
                    (0, 0, 0),
                    (1, 0, 0),
                    color=BLACK,
                ),
                self._attachment_dot(),
            )
            for rod in self.rods
        }

        # --------------------------------------------------------------
        # Springs
        # --------------------------------------------------------------

        self._spring_mobjects = {
            spring: VGroup(
                VMobject(color=BLACK),
                self._attachment_dot(),
            )
            for spring in self.springs
        }

        # --------------------------------------------------------------
        # Walls
        # --------------------------------------------------------------

        self._wall_mobjects = {
            wall: self._wall_mobject()
            for wall in self.walls
        }

        # --------------------------------------------------------------
        # Vectors
        #
        # IMPORTANT:
        # Create the arrow at a reasonable length rather than at 0.1.
        # This gives Manim a proper arrowhead to work with.
        # --------------------------------------------------------------

        self._vector_mobjects = {}

        for visualization in self._visualizations:

            arrow = Arrow(
                start=np.array((0.0, 0.0, 0.0)),
                end=np.array((1.0, 0.0, 0.0)),
                color=visualization.color,
                buff=0,
                stroke_width=7,
                max_tip_length_to_length_ratio=1,
                max_stroke_width_to_length_ratio=100,
            )

            label = MathTex(
                self._vector_label(visualization),
                color=visualization.color,
            ).scale(0.6)

            vector_group = VGroup(
                arrow,
                label,
            )

            vector_group.set_opacity(0)

            self._vector_mobjects[visualization] = vector_group

        self._equation_mobject = (
            self._equation_display()
            if self.show_equations
            else None
        )

        # --------------------------------------------------------------
        # Add everything
        # --------------------------------------------------------------

        self.add(
            *self._wall_mobjects.values(),
            *self._spring_mobjects.values(),
            *self._rod_mobjects.values(),
            *self._mass_mobjects.values(),
            *self._vector_mobjects.values(),
            *([self._equation_mobject] if self._equation_mobject is not None else []),
        )

    def _equation_display(self):  # type: ignore[no-untyped-def]
        lagrangian = sp.simplify(
            self._kinetic - self._potential
        )
        equations = self.equations_of_motion()
        display = VGroup(
            MathTex(
                r"\mathcal{L} = " + sp.latex(lagrangian),
                color=BLACK,
            ),
            *[
                MathTex(
                    sp.latex(equation),
                    color=BLACK,
                )
                for equation in equations
            ],
        ).arrange(
            direction=np.array((0.0, -1.0, 0.0)),
            aligned_edge=np.array((-1.0, 0.0, 0.0)),
            buff=0.08,
        )
        display.scale(0.28)
        if display.width > config.frame_width - 0.5:
            display.scale_to_fit_width(config.frame_width - 0.5)
        display.to_edge(np.array((0.0, -1.0, 0.0)), buff=0.2)
        return display

    # ------------------------------------------------------------------
    # Wall visual
    # ------------------------------------------------------------------

    @staticmethod
    def _wall_mobject():  # type: ignore[no-untyped-def]

        return VGroup(
            Line(
                (-0.2, -0.45, 0),
                (-0.2, 0.45, 0),
                color=BLACK,
            ),
            *[
                Line(
                    (-0.2, value, 0),
                    (
                        -0.35,
                        value - 0.12,
                        0,
                    ),
                    color=BLACK,
                    stroke_width=1.5,
                )
                for value in np.linspace(
                    -0.35,
                    0.35,
                    5,
                )
            ],
        )

    @staticmethod
    def _attachment_dot():  # type: ignore[no-untyped-def]

        return Circle(
            radius=0.055,
            color=BLACK,
            fill_color=BLACK,
            fill_opacity=1,
        ).set_opacity(0)

    @staticmethod
    def _vector_label(
        visualization: "MassVector",
    ) -> str:

        return {
            "velocity": r"\vec{v}",
            "acceleration": r"\vec{a}",
            "force": r"\vec{F}",
        }[
            visualization.quantity
        ]

    # ------------------------------------------------------------------
    # Attachment dots
    # ------------------------------------------------------------------

    def _update_attachment_dot(
        self,
        mobject,
        start: AttachmentPoint,
        end: AttachmentPoint,
        geometry: dict[object, tuple[float, float]],
    ) -> None:  # type: ignore[no-untyped-def]

        wall_nodes = {
            self._node(wall.attachment)
            for wall in self.walls
        }

        point = (
            start
            if self._node(start) in wall_nodes
            else end
            if self._node(end) in wall_nodes
            else None
        )

        dot = mobject[1]

        if point is None:
            dot.set_opacity(0)

        else:
            dot.set_opacity(1)

            dot.move_to(
                self._visual_point(
                    geometry[point]
                )
            )

    # ------------------------------------------------------------------
    # Wall hatching
    # ------------------------------------------------------------------

    def _wall_hatch_side(
        self,
        wall: Wall,
        geometry: dict[object, tuple[float, float]],
    ) -> float:

        attachment = np.array(
            geometry[wall]
        )

        directions: list[np.ndarray] = []

        for connection in wall.attachment.connections:

            point = (
                connection.second
                if connection.first
                is wall.attachment
                else connection.first
            )

            owner = point.owner

            if isinstance(
                owner,
                (Rod, Spring),
            ):
                other = (
                    owner.end
                    if point is owner.start
                    else owner.start
                )

                target = np.array(
                    geometry[other]
                )

            elif isinstance(owner, Mass):
                target = np.array(
                    geometry[owner]
                )

            else:
                continue

            directions.append(
                target - attachment
            )

        direction = (
            np.mean(
                directions,
                axis=0,
            )
            if directions
            else np.array((1.0, 0.0))
        )

        normal_axis = (
            0
            if wall.orientation == "vertical"
            else 1
        )

        return (
            -1.0
            if direction[normal_axis] >= 0
            else 1.0
        )

    # ------------------------------------------------------------------
    # Visual updates
    # ------------------------------------------------------------------

    def _update_visuals(
        self,
        time: float,
    ) -> None:

        if not MANIM_AVAILABLE:
            return

        geometry = self.geometry_at(time)

        # --------------------------------------------------------------
        # Walls
        # --------------------------------------------------------------

        for wall, mobject in self._wall_mobjects.items():

            attachment = self._visual_point(
                geometry[wall]
            )

            hatch_side = self._wall_hatch_side(
                wall,
                geometry,
            )

            if wall.orientation == "vertical":

                support_start = (
                    attachment
                    + np.array(
                        (0.0, -0.45, 0.0)
                    )
                )

                support_end = (
                    attachment
                    + np.array(
                        (0.0, 0.45, 0.0)
                    )
                )

                hatch_starts = [
                    attachment
                    + np.array(
                        (0.0, offset, 0.0)
                    )
                    for offset in np.linspace(
                        -0.35,
                        0.35,
                        5,
                    )
                ]

                hatch_ends = [
                    start
                    + np.array(
                        (
                            0.15 * hatch_side,
                            -0.12,
                            0.0,
                        )
                    )
                    for start in hatch_starts
                ]

            else:

                support_start = (
                    attachment
                    + np.array(
                        (-0.45, 0.0, 0.0)
                    )
                )

                support_end = (
                    attachment
                    + np.array(
                        (0.45, 0.0, 0.0)
                    )
                )

                hatch_starts = [
                    attachment
                    + np.array(
                        (offset, 0.0, 0.0)
                    )
                    for offset in np.linspace(
                        -0.35,
                        0.35,
                        5,
                    )
                ]

                hatch_ends = [
                    start
                    + np.array(
                        (
                            -0.12,
                            0.15 * hatch_side,
                            0.0,
                        )
                    )
                    for start in hatch_starts
                ]

            mobject[0].put_start_and_end_on(
                support_start,
                support_end,
            )

            for hatch, start, end in zip(
                mobject[1:],
                hatch_starts,
                hatch_ends,
            ):
                hatch.put_start_and_end_on(
                    start,
                    end,
                )

        # --------------------------------------------------------------
        # Rods
        # --------------------------------------------------------------

        for rod, mobject in self._rod_mobjects.items():

            mobject[0].put_start_and_end_on(
                self._visual_point(
                    geometry[rod.start]
                ),
                self._visual_point(
                    geometry[rod.end]
                ),
            )

            self._update_attachment_dot(
                mobject,
                rod.start,
                rod.end,
                geometry,
            )

        # --------------------------------------------------------------
        # Springs
        # --------------------------------------------------------------

        for spring, mobject in self._spring_mobjects.items():

            self._spring_points(
                mobject[0],
                self._visual_point(
                    geometry[spring.start]
                ),
                self._visual_point(
                    geometry[spring.end]
                ),
            )

            self._update_attachment_dot(
                mobject,
                spring.start,
                spring.end,
                geometry,
            )

        # --------------------------------------------------------------
        # Masses
        # --------------------------------------------------------------

        for mass, mobject in self._mass_mobjects.items():

            mobject.move_to(
                self._visual_point(
                    geometry[mass]
                )
            )

        # --------------------------------------------------------------
        # Motion vectors
        # --------------------------------------------------------------

        for visualization, group in self._vector_mobjects.items():

            if self._trajectory is None:
                group.set_opacity(0)
                continue

            vector = self._motion_vector(
                visualization,
                time,
            )

            magnitude = np.linalg.norm(vector)

            if magnitude < 1e-8:
                group.set_opacity(0)
                continue

            # ----------------------------------------------------------
            # Convert physical vector -> visually scaled vector.
            # ----------------------------------------------------------

            maximum = self._vector_max_magnitudes.get(
                visualization,
                0.0,
            )

            normalized_magnitude = min(1.0, magnitude / maximum)
            visual_length = max(0.0, 2.0 * normalized_magnitude)
            scaled_vector = vector / magnitude * visual_length

            start = self._visual_point(
                geometry[
                    visualization.mass
                ]
            )

            end = (
                start
                + np.array(
                    (
                        scaled_vector[0],
                        scaled_vector[1],
                        0.0,
                    )
                )
            )

            # ----------------------------------------------------------
            # Update Arrow.
            #
            # Crucially, the direction and length below are based on
            # the VISUAL vector, not the physical vector.
            # ----------------------------------------------------------

            arrow = group[0]

            arrow.put_start_and_end_on(
                start,
                end,
            )

            arrow.set_color(
                visualization.color
            )

            # Scale all arrow dimensions from the same trajectory-normalized
            # magnitude. The peak vector receives the current maximum styling.
            arrow.set_stroke(
                width=2.5 + 4.5 * normalized_magnitude
            )
            arrow.tip.scale_to_fit_width(
                0.08 + 0.28 * normalized_magnitude
            )

            # ----------------------------------------------------------
            # Label placement.
            #
            # Put it slightly beyond the arrowhead and offset it
            # perpendicular to the vector so it doesn't overlap.
            # ----------------------------------------------------------

            direction_2d = (
                scaled_vector
                / visual_length
            )

            direction = np.array(
                (
                    direction_2d[0],
                    direction_2d[1],
                    0.0,
                )
            )

            perpendicular = np.array(
                (
                    -direction_2d[1],
                    direction_2d[0],
                    0.0,
                )
            )

            label_position = (
                end
                + direction * (0.1 + 0.12 * normalized_magnitude)
                + perpendicular * (0.08 + 0.08 * normalized_magnitude)
            )

            label = group[1]

            label.move_to(
                label_position
            )

            label.set_color(
                visualization.color
            )

            group.set_opacity(1)

    # ------------------------------------------------------------------
    # Spring geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _spring_points(
        mobject,
        start: np.ndarray,
        end: np.ndarray,
    ) -> None:  # type: ignore[no-untyped-def]

        vector = end - start
        length = np.linalg.norm(vector)

        if length == 0:
            mobject.set_points_as_corners(
                [
                    start,
                    end,
                ]
            )
            return

        unit = vector / length

        normal = np.array(
            (
                -unit[1],
                unit[0],
                0.0,
            )
        )

        lead = min(
            0.25,
            length / 5,
        )

        points = [
            start,
            start + unit * lead,
        ]

        points.extend(
            start
            + vector * fraction
            + normal
            * (
                0.12
                if index % 2
                else -0.12
            )
            for index, fraction
            in enumerate(
                np.linspace(
                    0.15,
                    0.85,
                    9,
                )
            )
        )

        points.extend(
            (
                end - unit * lead,
                end,
            )
        )

        mobject.set_points_as_corners(
            points
        )


# ----------------------------------------------------------------------
# Manim simulation animation
# ----------------------------------------------------------------------

if MANIM_AVAILABLE:

    class _SimulationAnimation(Animation):

        def __init__(
            self,
            system: System,
            duration: float,
        ) -> None:

            super().__init__(
                system,
                run_time=duration,
            )

            self.system = system
            self.duration = duration

        def interpolate_mobject(
            self,
            alpha: float,
        ) -> None:

            self.system._update_visuals(
                alpha * self.duration
            )