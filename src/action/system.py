"""System orchestration: topology, configuration, energies, trajectory, visuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from .components import AttachmentPoint, Mass, Rod, Spring, Wall
from .configuration import Configuration
from .connections import Connection, Gravity, Hinge
from .context import pop_system, push_system
from .coordinates import Coordinate, CoordinateRate
from .style import VisualStyle

if TYPE_CHECKING:
    from .visualizations import MassVector

try:
    from manim import (
        Animation,
        Arc,
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


@dataclass(frozen=True)
class _HingeCoordinateVisual:
    coordinate: Coordinate
    hinge: Hinge
    child_rod: Rod
    parent_rod: Rod | None
    wall: Wall | None
    pivot: AttachmentPoint
    child_endpoint: AttachmentPoint
    label_latex: str


class CoordinateCollection(tuple):
    def __new__(
        cls,
        coordinates: tuple[Coordinate, ...],
        system: "System",
    ) -> "CoordinateCollection":
        instance = super().__new__(cls, coordinates)
        instance._system = system
        return instance

    def show(self) -> "CoordinateCollection":
        self._system._show_all_hinge_coordinates()
        return self


class System(VGroup):
    """A Manim-compatible 2D mechanism with one shared Lagrangian pipeline."""

    def __init__(
        self,
        objects: list[object] | None = None,
        initial: Mapping[Coordinate | CoordinateRate, float] | None = None,
        fields: list[Gravity] | None = None,
        show_equations: bool | str | Sequence[str] = False,
        equation_dot_notation: bool = False,
        show_gravity: bool = False,
        wall_angle_reference: str = "wall",
        bar_angle_reference: str = "relative",
        visual_style: VisualStyle | None = None,
    ) -> None:
        super().__init__()

        if wall_angle_reference not in {
            "wall",
            "wall_normal",
            "gravity",
            "global_x",
            "global_y",
        }:
            raise ValueError(
                "wall_angle_reference must be one of: wall, wall_normal, "
                "gravity, global_x, global_y."
            )

        if bar_angle_reference not in {"relative", "geometric"}:
            raise ValueError(
                "bar_angle_reference must be either 'relative' or 'geometric'."
            )

        self.objects = list(objects or [])
        self.initial = dict(initial or {})
        self.fields = list(fields) if fields is not None else []
        self.show_equations = show_equations
        self.equation_dot_notation = equation_dot_notation
        self.show_gravity = show_gravity
        self.wall_angle_reference = wall_angle_reference
        self.bar_angle_reference = bar_angle_reference
        self.visual_style = visual_style or VisualStyle()

        self._uses_active_gravity = fields is None
        self._context_connections: list[Connection] = []
        self._context_fields: list[Gravity] = []

        # Vector visualizations.
        self._visualizations: list["MassVector"] = []
        self._vector_mobjects: dict["MassVector", object] = {}
        self._vector_scales: dict["MassVector", float] = {}
        self._vector_max_magnitudes: dict["MassVector", float] = {}
        self._vector_samples: dict["MassVector", np.ndarray] = {}

        self._coordinate_visualizations: list[Coordinate] = []
        self._rod_length_visualizations: list[Rod] = []
        self._coordinate_mobjects: dict[Coordinate, object] = {}
        self._hinge_coordinate_visuals: dict[Coordinate, _HingeCoordinateVisual] = {}

        self._mass_labels: dict[Mass, str] = {}
        self._mass_symbols: dict[Mass, sp.Symbol] = {}
        self._spring_symbols: dict[Spring, sp.Symbol] = {}
        self._rod_symbols: dict[Rod, sp.Symbol] = {}
        self._gravity_symbol = sp.Symbol("g", real=True)
        self._parameter_values: dict[sp.Symbol, float] = {}

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

    def _register_coordinate_visualization(
        self,
        coordinate: Coordinate,
    ) -> None:
        if self._compiled:
            raise RuntimeError(
                "Cannot add coordinate visualizations after a System has been compiled."
            )

        rod = next(
            (
                item
                for item in self.objects
                if isinstance(item, Rod) and item.length is coordinate
            ),
            None,
        )
        if rod is not None:
            if rod not in self._rod_length_visualizations:
                self._rod_length_visualizations.append(rod)
            return

        if coordinate not in self._coordinate_visualizations:
            self._coordinate_visualizations.append(coordinate)

    def _register_rod_length_visualization(self, rod: Rod) -> None:
        if self._compiled:
            raise RuntimeError(
                "Cannot add rod length visualizations after a System has been compiled."
            )
        if rod not in self._rod_length_visualizations:
            self._rod_length_visualizations.append(rod)

    def _show_all_hinge_coordinates(self) -> None:
        if self._compiled:
            raise RuntimeError(
                "Cannot add coordinate visualizations after a System has been compiled."
            )

        for coordinate in self._available_hinge_coordinates():
            self._register_coordinate_visualization(coordinate)

    def _available_hinge_coordinates(self) -> tuple[Coordinate, ...]:
        hinges = (
            [
                connection
                for connection in self.connections
                if isinstance(connection, Hinge)
            ]
            if hasattr(self, "connections")
            else [
                connection
                for connection in self._context_connections
                if isinstance(connection, Hinge)
            ]
        )

        return tuple(
            hinge.rotation
            for hinge in hinges
        )

    def _register_field(self, field: Gravity) -> None:
        if self._compiled:
            raise RuntimeError(
                "Cannot add fields after a System has been compiled."
            )

        self._context_fields.append(field)

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        if self._compiled:
            return

        if self._context_fields:
            self.fields = [self._context_fields[-1]]
        elif self._uses_active_gravity and not self.fields and Gravity._active is not None:
            # Preserve standalone Gravity(...) declarations for subsequently created systems.
            self.fields = [Gravity._active]

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

        self._prepare_component_symbols()

        self.connections = self._collect_connections()
        self._node_by_point = self._make_nodes()

        self.configuration = Configuration(self)
        self._prepare_hinge_coordinate_visuals()

        self._trajectory: Trajectory | None = None

        self._view_scale = 1.0
        self._labels_view_scale = 1.0
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

        mass_matrix_numeric = self._mass_matrix.subs(
            self._parameter_values,
        )

        bias_numeric = self._bias.subs(
            self._parameter_values,
        )

        self._mass_matrix_fn = sp.lambdify(
            [*self._q, *self._dq],
            mass_matrix_numeric,
            "numpy",
        )

        self._bias_fn = sp.lambdify(
            [*self._q, *self._dq],
            bias_numeric,
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
    def coordinates(self) -> CoordinateCollection:
        """Independent generalized coordinates used by the solver."""

        if not self._compiled and not hasattr(self, "configuration"):
            hinge_coordinates = [
                connection.rotation
                for connection in self._context_connections
                if isinstance(connection, Hinge)
            ]

            return CoordinateCollection(
                tuple(hinge_coordinates),
                self,
            )

        return CoordinateCollection(
            self.configuration.intrinsic_coordinates,
            self,
        )

    @property
    def degrees_of_freedom(self) -> int:
        return len(self._q)

    def equations_of_motion(self) -> tuple[sp.Equality, ...]:
        functions, time = self._equation_coordinate_functions(
            dot_notation=False,
        )

        lagrangian = self._lagrangian_with_time_functions(
            functions,
            time,
        )

        return self._euler_lagrange_equations(
            functions,
            lagrangian,
            time,
        )

    def _equation_coordinate_functions(
        self,
        dot_notation: bool,
    ) -> tuple[list[sp.Expr], sp.Symbol]:
        time = sp.symbols("t", real=True)

        if dot_notation:
            from sympy.physics.vector import dynamicsymbols

            time = dynamicsymbols._t

            return (
                list(
                dynamicsymbols(
                    [
                        str(value)
                        for value in self._q
                    ]
                )
                ),
                time,
            )

        return (
            [
                sp.Function(str(value))(time)
                for value in self._q
            ],
            time,
        )

    def _lagrangian_with_time_functions(
        self,
        functions: list[sp.Expr],
        time: sp.Symbol,
    ) -> sp.Expr:
        mapping = (
            dict(zip(self._q, functions))
            | dict(
                zip(
                    self._dq,
                    [sp.diff(value, time) for value in functions],
                )
            )
        )

        return (
            self._kinetic - self._potential
        ).subs(mapping)

    @staticmethod
    def _euler_lagrange_equations(
        functions: list[sp.Expr],
        lagrangian: sp.Expr,
        time: sp.Symbol,
    ) -> tuple[sp.Equality, ...]:
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

    def _prepare_component_symbols(self) -> None:
        self._mass_labels = {}
        self._mass_symbols = {}
        self._spring_symbols = {}
        self._rod_symbols = {}
        self._parameter_values = {}

        mass_total = len(self.masses)
        for index, mass in enumerate(self.masses, start=1):
            label = "m" if mass_total == 1 else f"m_{{{index}}}"
            symbol_name = "m" if mass_total == 1 else f"m_{index}"
            symbol = sp.Symbol(symbol_name, real=True, positive=True)
            self._mass_labels[mass] = label
            self._mass_symbols[mass] = symbol
            self._parameter_values[symbol] = float(mass.m)

        spring_total = len(self.springs)
        for index, spring in enumerate(self.springs, start=1):
            symbol_name = "k" if spring_total == 1 else f"k_{index}"
            symbol = sp.Symbol(symbol_name, real=True, positive=True)
            self._spring_symbols[spring] = symbol
            self._parameter_values[symbol] = float(spring.k)

        rod_total = len(self.rods)
        for index, rod in enumerate(self.rods, start=1):
            symbol_name = "ell" if rod_total == 1 else f"ell_{index}"
            symbol = sp.Symbol(symbol_name, real=True, positive=True)
            self._rod_symbols[rod] = symbol
            self._parameter_values[symbol] = float(rod.length)

        self._parameter_values[self._gravity_symbol] = (
            float(self.fields[0].g)
            if self.fields
            else 0.0
        )

    def _prepare_hinge_coordinate_visuals(self) -> None:
        self._hinge_coordinate_visuals = {}

        if not self._coordinate_visualizations:
            return

        hinges = [
            connection
            for connection in self.connections
            if isinstance(connection, Hinge)
        ]

        hinge_by_rotation = {
            hinge.rotation: hinge
            for hinge in hinges
        }

        hinge_by_rod_rotation: dict[Coordinate, Hinge] = {}
        for rod, hinge in self.configuration._rod_hinge.items():
            if hinge is not None:
                hinge_by_rod_rotation[rod.rotation] = hinge

        symbol_by_coordinate = {
            spec.intrinsic: spec.symbol
            for spec in self.configuration.specs
            if isinstance(spec.intrinsic, Coordinate)
        }

        for requested in self._coordinate_visualizations:
            hinge = (
                hinge_by_rotation.get(requested)
                or hinge_by_rod_rotation.get(requested)
            )

            if hinge is None:
                raise ValueError(
                    "Coordinate visualization currently supports only "
                    "hinge rotation coordinates."
                )

            coordinate = hinge.rotation
            if coordinate in self._hinge_coordinate_visuals:
                continue

            child_rod = self._hinge_child_rod(hinge)
            if child_rod is None:
                raise ValueError(
                    "Cannot draw hinge coordinate: no controlled rod "
                    "was found for the hinge."
                )

            pivot, child_endpoint = self._hinge_pivot_and_child_endpoint(
                hinge,
                child_rod,
            )

            parent_rod = self._parent_rod_at_pivot(
                pivot,
                child_rod,
            )

            opposite = (
                hinge.second
                if pivot is hinge.first
                else hinge.first
            )

            if isinstance(opposite.owner, Rod) and opposite.owner is not child_rod:
                parent_rod = opposite.owner

            wall = self._hinge_wall(
                hinge,
            )

            symbol = symbol_by_coordinate.get(
                coordinate,
                sp.Symbol("theta", real=True),
            )

            self._hinge_coordinate_visuals[coordinate] = _HingeCoordinateVisual(
                coordinate=coordinate,
                hinge=hinge,
                child_rod=child_rod,
                parent_rod=parent_rod,
                wall=wall,
                pivot=pivot,
                child_endpoint=child_endpoint,
                label_latex=sp.latex(symbol),
            )

    @staticmethod
    def _hinge_wall(
        hinge: Hinge,
    ) -> Wall | None:
        for endpoint in (hinge.first, hinge.second):
            owner = endpoint.owner
            if isinstance(owner, Wall):
                return owner

        return None

    def _hinge_child_rod(
        self,
        hinge: Hinge,
    ) -> Rod | None:
        for rod, rod_hinge in self.configuration._rod_hinge.items():
            if rod_hinge is hinge:
                return rod
        return None

    @staticmethod
    def _hinge_pivot_and_child_endpoint(
        hinge: Hinge,
        rod: Rod,
    ) -> tuple[AttachmentPoint, AttachmentPoint]:
        pivot = (
            rod.start
            if rod.start in (hinge.first, hinge.second)
            else rod.end
        )

        child_endpoint = (
            rod.end
            if pivot is rod.start
            else rod.start
        )

        return pivot, child_endpoint

    def _parent_rod_at_pivot(
        self,
        pivot: AttachmentPoint,
        child_rod: Rod,
    ) -> Rod | None:
        pivot_node = self._node(pivot)

        for point in self._points():
            if self._node(point) != pivot_node:
                continue

            owner = point.owner
            if not isinstance(owner, Rod):
                continue

            if owner is child_rod:
                continue

            return owner

        return None

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
            and abs(np.cos(self.walls[0].rotation))
            >= abs(np.sin(self.walls[0].rotation))
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

        # Spring extensions are exposed inputs, but not generalized coordinates.
        allowed |= {
            spring.extension
            for spring in self.springs
        }

        allowed |= {
            coordinate.rate
            for coordinate in self.coordinates
        }

        allowed |= {
            spring.extension.rate
            for spring in self.springs
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

            mass_coordinates = self.configuration._mass_intrinsics.get(mass)
            if mass_coordinates is not None and all(
                coordinate in self.initial
                for coordinate in mass_coordinates
            ):
                start = self.configuration.symbols.index(
                    self.configuration._mass_symbols[mass][0]
                )
                known[node] = q[start:start + 2]
                continue

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
                    np.array((
                        np.sin(wall.rotation),
                        -np.cos(wall.rotation),
                    ))
                    if wall is not None
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
                self._mass_symbols[mass]
                * velocity.dot(velocity)
                / 2
            )

            if self.fields:
                potential += (
                    self._mass_symbols[mass]
                    * self._gravity_symbol
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
                self._spring_symbols[spring]
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
        if self._trajectory is None or not self.visual_style.auto_frame:
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

        available_size = self.visual_style.frame_size - (
            2 * self.visual_style.frame_padding
        )
        if available_size <= 0:
            raise ValueError(
                "VisualStyle frame_size must exceed twice frame_padding."
            )

        self._view_scale = available_size / max(*(high - low), 1.0)

        if self.visual_style.frame_min_scale is not None:
            self._view_scale = max(
                self._view_scale,
                self.visual_style.frame_min_scale,
            )
        if self.visual_style.frame_max_scale is not None:
            self._view_scale = min(
                self._view_scale,
                self.visual_style.frame_max_scale,
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
            self._view_scale
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

    def _view_length(self, physical_length: float) -> float:
        return physical_length * self._view_scale

    def _view_stroke_width(self, physical_width: float) -> float:
        return max(1.0, self._view_length(physical_width) * 10.0)

    def _update_readability_scale(self) -> None:
        if self._labels_view_scale == self._view_scale:
            return

        factor = self._view_scale / self._labels_view_scale

        for mobject in self._mass_mobjects.values():
            mobject[1].scale(factor)
        for label in self._rod_length_mobjects.values():
            label.scale(factor)
        for mobject in self._coordinate_mobjects.values():
            mobject[3].scale(factor)
        for mobject in self._vector_mobjects.values():
            mobject[1].scale(factor)

        self._labels_view_scale = self._view_scale

    # ------------------------------------------------------------------
    # Visual creation
    # ------------------------------------------------------------------

    def _build_visuals(self) -> None:

        style = self.visual_style
        mass_radius = self._view_length(style.mass_radius)

        # --------------------------------------------------------------
        # Masses
        # --------------------------------------------------------------

        self._mass_mobjects = {
            mass: VGroup(
                Circle(
                    radius=mass_radius,
                    color=style.mass_color,
                    fill_color=style.mass_fill_color,
                    fill_opacity=1,
                    stroke_width=self._view_stroke_width(style.mass_stroke_width),
                ),
                MathTex(
                    self._mass_labels[mass],
                    color=style.mass_color,
                ).scale(style.mass_label_scale),
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
                    color=style.rod_color,
                ),
                self._attachment_dot(style.rod_color),
            )
            for rod in self.rods
        }

        self._rod_length_mobjects = {
            rod: MathTex(
                self._rod_length_latex(rod),
                color=style.rod_color,
            ).scale(style.rod_label_scale)
            for rod in self._rod_length_visualizations
        }

        # --------------------------------------------------------------
        # Springs
        # --------------------------------------------------------------

        self._spring_mobjects = {
            spring: VGroup(
                VMobject(color=style.spring_color),
                self._attachment_dot(style.spring_color),
            )
            for spring in self.springs
        }

        # --------------------------------------------------------------
        # Walls
        # --------------------------------------------------------------

        self._wall_mobjects = {
            wall: self._wall_mobject(wall.size)
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
                stroke_width=style.vector_stroke_min_width + style.vector_stroke_width,
                max_tip_length_to_length_ratio=1,
                max_stroke_width_to_length_ratio=100,
            )

            label = MathTex(
                self._vector_label(visualization),
                color=visualization.color,
            ).scale(style.vector_label_scale)

            vector_group = VGroup(
                arrow,
                label,
            )

            vector_group.set_opacity(0)

            self._vector_mobjects[visualization] = vector_group

        # --------------------------------------------------------------
        # Coordinate visuals (hinge rotations)
        # --------------------------------------------------------------

        self._coordinate_mobjects = {}

        for coordinate, specification in self._hinge_coordinate_visuals.items():
            arc = Arc(
                radius=self._view_length(style.angle_arc_radius),
                start_angle=0.0,
                angle=np.pi / 2,
                arc_center=np.array((0.0, 0.0, 0.0)),
                color=style.wall_color,
                stroke_width=self._view_stroke_width(self.visual_style.angle_width),
                fill_opacity=0,
            )

            reference_line = Line(
                (0.0, 0.0, 0.0),
                (self._view_length(style.angle_arc_radius), 0.0, 0.0),
                color=style.wall_color,
                stroke_width=self._view_stroke_width(style.angle_radial_width),
            )

            current_line = Line(
                (0.0, 0.0, 0.0),
                (self._view_length(style.angle_arc_radius), 0.0, 0.0),
                color=style.wall_color,
                stroke_width=self._view_stroke_width(style.angle_radial_width),
            )

            label = MathTex(
                specification.label_latex,
                color=style.wall_color,
            ).scale(style.angle_label_scale)

            self._coordinate_mobjects[coordinate] = VGroup(
                arc,
                reference_line,
                current_line,
                label,
            )

        self._equation_mobject = (
            self._equation_display()
            if self.show_equations
            else None
        )

        self._gravity_mobject = (
            self._gravity_indicator()
            if (self.show_gravity or bool(self._context_fields)) and bool(self.fields)
            else None
        )

        # --------------------------------------------------------------
        # Add everything
        # --------------------------------------------------------------

        self.add(
            *self._wall_mobjects.values(),
            *self._spring_mobjects.values(),
            *self._rod_mobjects.values(),
            *self._rod_length_mobjects.values(),
            *self._mass_mobjects.values(),
            *self._coordinate_mobjects.values(),
            *self._vector_mobjects.values(),
            *([self._equation_mobject] if self._equation_mobject is not None else []),
            *([self._gravity_mobject] if self._gravity_mobject is not None else []),
        )

    def _equation_display(self):  # type: ignore[no-untyped-def]
        show_lagrangian, show_eom = self._equation_sections()

        if not show_lagrangian and not show_eom:
            return None

        functions, time = self._equation_coordinate_functions(
            dot_notation=self.equation_dot_notation,
        )

        lagrangian = sp.simplify(
            self._lagrangian_with_time_functions(
                functions,
                time,
            )
        )

        equations = self._euler_lagrange_equations(
            functions,
            lagrangian,
            time,
        )

        lines: list[object] = []

        if show_lagrangian:
            lines.append(
                MathTex(
                    r"\mathcal{L} = "
                    + self._latex_for_display(lagrangian),
                    color=self.visual_style.wall_color,
                )
            )

        if show_eom:
            lines.extend(
                MathTex(
                    self._latex_for_display(equation),
                    color=self.visual_style.wall_color,
                )
                for equation in equations
            )

        display = VGroup(*lines).arrange(
            direction=np.array((0.0, -1.0, 0.0)),
            aligned_edge=np.array((-1.0, 0.0, 0.0)),
            buff=self.visual_style.equation_line_buffer,
        )
        display.scale(self.visual_style.equation_scale)
        if display.width > config.frame_width - 2 * self.visual_style.equation_edge_buffer:
            display.scale_to_fit_width(
                config.frame_width - 2 * self.visual_style.equation_edge_buffer
            )
        display.to_edge(
            np.array((0.0, -1.0, 0.0)),
            buff=self.visual_style.equation_edge_buffer,
        )
        return display

    def _equation_sections(self) -> tuple[bool, bool]:
        if self.show_equations is False:
            return (False, False)

        if self.show_equations is True:
            return (True, True)

        if isinstance(self.show_equations, str):
            requested = {self.show_equations.lower()}
        else:
            requested = {
                str(value).lower()
                for value in self.show_equations
            }

        if requested & {"all", "everything", "both"}:
            return (True, True)

        show_lagrangian = bool(
            requested
            & {
                "lagrangian",
                "lagrange",
                "l",
            }
        )

        show_eom = bool(
            requested
            & {
                "eom",
                "equation",
                "equations",
                "motion",
            }
        )

        if not show_lagrangian and not show_eom:
            choices = ", ".join(sorted(requested))
            raise ValueError(
                "show_equations accepts False/True or values in "
                "{'lagrangian', 'eom', 'all'}; got: "
                f"{choices}."
            )

        return (show_lagrangian, show_eom)

    def _latex_for_display(self, expression: sp.Expr | sp.Equality) -> str:
        expression = self._normalize_display_numbers(
            expression,
        )

        if not self.equation_dot_notation:
            return sp.latex(expression)

        from sympy.physics.vector import vlatex

        return vlatex(expression)

    @staticmethod
    def _normalize_display_numbers(
        expression: sp.Expr | sp.Equality,
    ) -> sp.Expr | sp.Equality:
        replacements: dict[sp.Float, sp.Integer] = {}

        for value in expression.atoms(sp.Float):
            nearest_integer = int(
                round(float(value))
            )

            if abs(float(value) - nearest_integer) < 1e-10:
                replacements[value] = sp.Integer(nearest_integer)

        if not replacements:
            return expression

        return expression.xreplace(replacements)

    def _gravity_indicator(self):  # type: ignore[no-untyped-def]
        gravity_vector = np.array(
            (*self.fields[0].vector, 0.0),
            dtype=float,
        )

        magnitude = np.linalg.norm(
            gravity_vector[:2]
        )

        if magnitude < 1e-8:
            direction = np.array((0.0, -1.0, 0.0))
        else:
            direction = gravity_vector / magnitude

        start = np.array((0.0, 0.0, 0.0))
        end = start + self.visual_style.gravity_indicator_length * direction
        perpendicular = np.array(
            (
                -direction[1],
                direction[0],
                0.0,
            )
        )

        arrow = Arrow(
            start=start,
            end=end,
            color=BLACK,
            buff=0,
            stroke_width=self._view_stroke_width(
                self.visual_style.gravity_indicator_width
            ),
            max_tip_length_to_length_ratio=(
                self.visual_style.gravity_indicator_tip_ratio
            ),
        )

        label = MathTex(
            "g",
            color=BLACK,
        ).scale(self.visual_style.gravity_indicator_label_scale)

        label.move_to(
            end
            + self.visual_style.gravity_indicator_offset * direction
            + self.visual_style.gravity_indicator_offset * perpendicular
        )

        indicator = VGroup(
            arrow,
            label,
        )

        indicator.to_corner(
            np.array((1.0, 1.0, 0.0)),
            buff=self.visual_style.gravity_indicator_corner_buffer,
        )

        return indicator

    # ------------------------------------------------------------------
    # Wall visual
    # ------------------------------------------------------------------

    def _wall_mobject(self, size: float = 0.9):  # type: ignore[no-untyped-def]
        style = self.visual_style
        half_size = self._view_length(size / 2)
        hatch_count = max(2, int(np.ceil(self.visual_style.hatch_density * size)))
        tangent = np.array((1.0, 0.0, 0.0))
        normal = np.array((0.0, 1.0, 0.0))

        return VGroup(
            Line(
                -self._view_length(style.wall_support_offset) * tangent
                - half_size * normal,
                -self._view_length(style.wall_support_offset) * tangent
                + half_size * normal,
                color=style.wall_color,
                stroke_width=self._view_stroke_width(style.wall_width),
            ),
            *[
                Line(
                    -self._view_length(style.wall_support_offset) * tangent
                    + value * normal,
                    -self._view_length(style.wall_support_offset) * tangent
                    + value * normal
                    + self._view_length(style.wall_hatch_length) * normal
                    - self._view_length(style.wall_hatch_backset) * tangent,
                    color=style.wall_color,
                    stroke_width=self._view_stroke_width(style.wall_hatch_width),
                )
                for value in self._view_length(
                    np.linspace(-size / 2, size / 2, hatch_count)
                )
            ],
        )

    def _attachment_dot(self, color: str):  # type: ignore[no-untyped-def]

        return Circle(
            radius=self._view_length(self.visual_style.attachment_radius),
            color=color,
            fill_color=color,
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

    def _rod_length_latex(self, rod: Rod) -> str:
        if len(self.rods) == 1:
            return r"\ell"
        return rf"\ell_{self.rods.index(rod) + 1}"

    @staticmethod
    def _normalize_2d(
        vector: np.ndarray,
    ) -> np.ndarray:
        length = np.linalg.norm(vector)
        if length < 1e-8:
            return np.zeros(2)
        return vector / length

    @staticmethod
    def _signed_angle(
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        cross = reference[0] * current[1] - reference[1] * current[0]
        dot = reference[0] * current[0] + reference[1] * current[1]
        return float(np.arctan2(cross, dot))

    def _rod_direction_from_pivot(
        self,
        rod: Rod,
        pivot: AttachmentPoint,
        geometry: dict[object, tuple[float, float]],
    ) -> np.ndarray:
        other = rod.end if rod.start is pivot else rod.start
        pivot_point = np.array(
            geometry[pivot],
            dtype=float,
        )
        other_point = np.array(
            geometry[other],
            dtype=float,
        )
        return self._normalize_2d(other_point - pivot_point)

    def _coordinate_reference_direction(
        self,
        specification: _HingeCoordinateVisual,
        geometry: dict[object, tuple[float, float]],
    ) -> np.ndarray:
        if specification.parent_rod is None:
            if specification.wall is None:
                return np.array((1.0, 0.0))

            angle = self.configuration._wall_reference_angle(
                specification.hinge,
            )
            return np.array((
                float(sp.cos(angle)),
                float(sp.sin(angle)),
            ))

        if self.bar_angle_reference == "geometric":
            return self._rod_direction_from_pivot(
                specification.parent_rod,
                specification.pivot,
                geometry,
            )

        return self._rod_direction(specification.parent_rod, geometry)

    def _coordinate_current_direction(
        self,
        specification: _HingeCoordinateVisual,
        geometry: dict[object, tuple[float, float]],
    ) -> np.ndarray:
        if specification.parent_rod is None:
            return self._rod_direction_from_pivot(
                specification.child_rod,
                specification.pivot,
                geometry,
            )

        if self.bar_angle_reference == "geometric":
            return self._rod_direction_from_pivot(
                specification.child_rod,
                specification.pivot,
                geometry,
            )

        return self._rod_direction(specification.child_rod, geometry)

    def _rod_direction(
        self,
        rod: Rod,
        geometry: dict[object, tuple[float, float]],
    ) -> np.ndarray:
        return self._normalize_2d(
            np.array(geometry[rod.end], dtype=float)
            - np.array(geometry[rod.start], dtype=float)
        )

    def _coordinate_sweep(
        self,
        specification: _HingeCoordinateVisual,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        sweep = self._signed_angle(
            reference,
            current,
        )

        return float(sweep)

    def _wall_angle_is_on_hatched_side(
        self,
        specification: _HingeCoordinateVisual,
        reference: np.ndarray,
        sweep: float,
        geometry: dict[object, tuple[float, float]],
    ) -> bool:
        wall = specification.wall
        if wall is None:
            return False

        start_angle = float(np.arctan2(reference[1], reference[0]))
        midpoint_angle = start_angle + 0.5 * sweep
        midpoint = np.array(
            (
                np.cos(midpoint_angle),
                np.sin(midpoint_angle),
            )
        )

        hatch_side = self._wall_hatch_side(wall, geometry)
        normal = np.array(
            (-np.sin(wall.rotation), np.cos(wall.rotation))
        )
        return bool(np.dot(midpoint, normal) * hatch_side > 1e-8)

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
        dot.set(width=2 * self._view_length(self.visual_style.attachment_radius))

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

        normal = np.array(
            (-np.sin(wall.rotation), np.cos(wall.rotation))
        )
        return -1.0 if np.dot(direction, normal) >= 0 else 1.0

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
        self._update_readability_scale()

        # --------------------------------------------------------------
        # Walls
        # --------------------------------------------------------------

        for wall, mobject in self._wall_mobjects.items():

            attachment = self._visual_point(
                geometry[wall]
            )

            half_size = self._view_length(wall.size / 2)
            hatch_count = max(
                2,
                int(np.ceil(self.visual_style.hatch_density * wall.size)),
            )

            hatch_side = self._wall_hatch_side(
                wall,
                geometry,
            )

            tangent = np.array((
                np.cos(wall.rotation),
                np.sin(wall.rotation),
                0.0,
            ))
            normal = np.array((
                -np.sin(wall.rotation),
                np.cos(wall.rotation),
                0.0,
            ))
            support_start = attachment - half_size * tangent
            support_end = attachment + half_size * tangent
            hatch_starts = [
                attachment + offset * tangent
                for offset in np.linspace(-half_size, half_size, hatch_count)
            ]
            hatch_ends = [
                start
                + hatch_side * (
                    self._view_length(self.visual_style.wall_hatch_length) * normal
                    - self._view_length(self.visual_style.wall_hatch_backset) * tangent
                )
                for start in hatch_starts
            ]

            mobject[0].put_start_and_end_on(
                support_start,
                support_end,
            )
            mobject[0].set_stroke(
                width=self._view_stroke_width(self.visual_style.wall_width),
                color=self.visual_style.wall_color,
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
                hatch.set_stroke(
                    width=self._view_stroke_width(self.visual_style.wall_hatch_width),
                    color=self.visual_style.wall_color,
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
            mobject[0].set_stroke(
                width=self._view_stroke_width(self.visual_style.rod_width),
                color=self.visual_style.rod_color,
            )

            self._update_attachment_dot(
                mobject,
                rod.start,
                rod.end,
                geometry,
            )

            label = self._rod_length_mobjects.get(rod)
            if label is not None:
                start = self._visual_point(geometry[rod.start])
                end = self._visual_point(geometry[rod.end])
                direction = end - start
                normal = self._normalize_2d(
                    np.array((-direction[1], direction[0]))
                )
                label.move_to(
                    0.5 * (start + end)
                    + self._view_length(self.visual_style.angle_label_offset)
                    * np.array((*normal, 0.0))
                )

        # --------------------------------------------------------------
        # Springs
        # --------------------------------------------------------------

        for spring, mobject in self._spring_mobjects.items():

            mobject[0].set_stroke(
                width=self._view_stroke_width(self.visual_style.spring_width),
                color=self.visual_style.spring_color,
            )

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

            mobject[0].set(width=2 * self._view_length(self.visual_style.mass_radius))
            mobject[0].set_stroke(
                width=self._view_stroke_width(self.visual_style.mass_stroke_width),
                color=self.visual_style.mass_color,
            )
            mobject[0].set_fill(color=self.visual_style.mass_fill_color)

            mobject.move_to(
                self._visual_point(
                    geometry[mass]
                )
            )

        # --------------------------------------------------------------
        # Coordinate visuals (hinge rotations)
        # --------------------------------------------------------------

        for coordinate, group in self._coordinate_mobjects.items():
            specification = self._hinge_coordinate_visuals[
                coordinate
            ]

            reference = self._coordinate_reference_direction(
                specification,
                geometry,
            )

            current = self._coordinate_current_direction(
                specification,
                geometry,
            )

            if np.linalg.norm(reference) < 1e-8 or np.linalg.norm(current) < 1e-8:
                group.set_opacity(0)
                continue

            sweep = self._coordinate_sweep(
                specification,
                reference,
                current,
            )

            if abs(sweep) < 1e-6:
                sweep = 1e-6

            if self._wall_angle_is_on_hatched_side(
                specification,
                reference,
                sweep,
                geometry,
            ):
                reference = -reference
                current = -current

            center = self._visual_point(
                geometry[specification.pivot]
            )

            radius = self._view_length(self.visual_style.angle_arc_radius)

            reference_3d = np.array(
                (
                    reference[0],
                    reference[1],
                    0.0,
                )
            )

            current_3d = np.array(
                (
                    current[0],
                    current[1],
                    0.0,
                )
            )

            start_angle = float(
                np.arctan2(
                    reference[1],
                    reference[0],
                )
            )

            arc = Arc(
                radius=radius,
                start_angle=start_angle,
                angle=sweep,
                arc_center=center,
                color=self.visual_style.wall_color,
                stroke_width=self._view_stroke_width(self.visual_style.angle_width),
                fill_opacity=0,
            )

            group[0].become(arc)
            group[1].set_stroke(
                width=self._view_stroke_width(self.visual_style.angle_radial_width)
            )
            group[2].set_stroke(
                width=self._view_stroke_width(self.visual_style.angle_radial_width)
            )
            radial_extension = self._view_length(
                self.visual_style.angle_radial_extension
            )

            group[1].put_start_and_end_on(
                center,
                center + (radius + radial_extension) * reference_3d,
            )

            group[2].put_start_and_end_on(
                center,
                center + (radius + radial_extension) * current_3d,
            )
            group[1].set_stroke(
                width=self._view_stroke_width(self.visual_style.angle_radial_width)
            )
            group[2].set_stroke(
                width=self._view_stroke_width(self.visual_style.angle_radial_width)
            )

            mid_angle = start_angle + 0.5 * sweep
            label_radius = radius + self._view_length(
                self.visual_style.angle_label_offset
            )

            group[3].move_to(
                center
                + np.array(
                    (
                        np.cos(mid_angle),
                        np.sin(mid_angle),
                        0.0,
                    )
                )
                * label_radius
            )

            # Keep arc as an unfilled curve while showing stroke and label.
            group[0].set_stroke(opacity=1)
            group[0].set_fill(opacity=0)
            group[1].set_stroke(opacity=1)
            group[2].set_stroke(opacity=1)
            group[3].set_opacity(1)

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
                width=self.visual_style.vector_stroke_min_width
                + self.visual_style.vector_stroke_width * normalized_magnitude
            )
            arrow.tip.scale_to_fit_width(
                self.visual_style.vector_tip_min_width
                + self.visual_style.vector_tip_extra_width * normalized_magnitude
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
                + direction * (
                    self.visual_style.vector_label_forward_offset
                    + self.visual_style.vector_label_forward_extra
                    * normalized_magnitude
                )
                + perpendicular * (
                    self.visual_style.vector_label_side_offset
                    + self.visual_style.vector_label_side_extra
                    * normalized_magnitude
                )
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

    def _spring_points(
        self,
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
            self._view_length(self.visual_style.spring_lead_length),
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
                self._view_length(self.visual_style.spring_amplitude)
                if index % 2
                else -self._view_length(self.visual_style.spring_amplitude)
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