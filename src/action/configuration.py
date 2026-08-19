"""Generalized-coordinate discovery and shared physical geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from .components import AttachmentPoint, Mass, Rod, Spring, Wall
from .connections import Fixed, Hinge
from .coordinates import Coordinate

if TYPE_CHECKING:
    from .system import System


@dataclass(frozen=True)
class CoordinateSpec:
    """One independent generalized coordinate and its optional public owner."""

    symbol: sp.Symbol
    intrinsic: object | None


class Configuration:
    """Maps independent coordinates to every topology node's physical position.

    Walls seed a kinematic traversal. A hinged rod propagates an anchored node
    to its opposite endpoint. Mass nodes reached this way are derived; every
    remaining mass contributes two internal Cartesian generalized coordinates.
    Springs never contribute coordinates and only read endpoint geometry.
    """

    def __init__(self, system: "System") -> None:
        self.system = system
        discovered_hinges = [
            connection
            for connection in system.connections
            if isinstance(connection, Hinge)
        ]

        # Keep rotational labels deterministic: wall-referenced hinges first,
        # then internal hinges (e.g. rod-rod), preserving declaration order
        # within each group.
        self.hinges = sorted(
            discovered_hinges,
            key=self._hinge_sort_key,
        )

        self._rod_hinge = self._find_rod_hinges()
        self._fixed_rod_neighbors = self._find_fixed_rod_neighbors()
        self._wall_positions = system._layout_walls()
        free_orientation_roots = self._free_orientation_roots()

        self.specs: list[CoordinateSpec] = []

        self._hinge_symbols: dict[int, sp.Symbol] = {}
        rotational_total = len(self.hinges) + len(free_orientation_roots)

        def rotation_symbol_name(index: int) -> str:
            return "theta" if rotational_total == 1 else f"theta_{index}"

        for index, hinge in enumerate(self.hinges, start=1):
            symbol = sp.Symbol(rotation_symbol_name(index), real=True)
            self.specs.append(CoordinateSpec(symbol, hinge.rotation))
            self._hinge_symbols[id(hinge)] = symbol

        self._free_rod_symbols: dict[Rod, sp.Symbol] = {}
        for index, rod in enumerate(free_orientation_roots, start=1):
            symbol_index = len(self.hinges) + index
            symbol = sp.Symbol(rotation_symbol_name(symbol_index), real=True)
            self.specs.append(CoordinateSpec(symbol, rod.rotation))
            self._free_rod_symbols[rod] = symbol

        self._mass_symbols: dict[Mass, tuple[sp.Symbol, sp.Symbol]] = {}
        self._mass_intrinsics: dict[Mass, tuple[Coordinate, Coordinate]] = {}
        self.free_masses = self._root_masses()
        for index, mass in enumerate(self.free_masses, start=1):
            x, y = self._free_mass_coordinate_symbols(index)
            x_coord = Coordinate(str(x))
            y_coord = Coordinate(str(y))
            self.specs.extend((CoordinateSpec(x, x_coord), CoordinateSpec(y, y_coord)))
            self._mass_symbols[mass] = (x, y)
            self._mass_intrinsics[mass] = (x_coord, y_coord)

        self._symbolic_nodes, self._rod_angles = self._propagate(symbolic=True)

        for mass in system.masses:
            node = system._node(mass.attachment)
            if node in self._symbolic_nodes or mass in self.free_masses:
                continue

            index = len(self.free_masses) + 1
            x, y = self._free_mass_coordinate_symbols(index)
            x_coord = Coordinate(str(x))
            y_coord = Coordinate(str(y))
            self.specs.extend((CoordinateSpec(x, x_coord), CoordinateSpec(y, y_coord)))
            self._mass_symbols[mass] = (x, y)
            self._mass_intrinsics[mass] = (x_coord, y_coord)
            self.free_masses.append(mass)
            self._symbolic_nodes[node] = sp.Matrix((x, y))

        self._validate_all_nodes_resolved()

    @staticmethod
    def _hinge_sort_key(hinge: Hinge) -> tuple[int, int]:
        has_wall = any(
            isinstance(endpoint.owner, Wall)
            for endpoint in (hinge.first, hinge.second)
        )
        return (0 if has_wall else 1, 0)

    def _free_mass_coordinate_symbols(
        self,
        index: int,
    ) -> tuple[sp.Symbol, sp.Symbol]:
        if len(self.system.masses) == 1:
            return sp.symbols("x y", real=True)

        return sp.symbols(
            f"x_{index} y_{index}",
            real=True,
        )

    @property
    def symbols(self) -> list[sp.Symbol]:
        return [spec.symbol for spec in self.specs]

    @property
    def intrinsic_coordinates(self) -> tuple[object, ...]:
        return tuple(spec.intrinsic for spec in self.specs if spec.intrinsic is not None)

    def numeric_positions(self, values: np.ndarray) -> dict[int, np.ndarray]:
        substitutions = dict(zip(self.symbols, values))
        return {
            node: np.array([float(value.subs(substitutions)) for value in position], dtype=float)
            for node, position in self._symbolic_nodes.items()
        }

    def symbolic_positions(self) -> dict[int, sp.Matrix]:
        return dict(self._symbolic_nodes)

    def set_wall_position(self, wall: Wall, position: np.ndarray) -> None:
        """Update a wall location before System derives its symbolic energies."""
        wall.position = (float(position[0]), float(position[1]))
        self._wall_positions[wall] = position
        self._symbolic_nodes[self.system._node(wall.attachment)] = sp.Matrix(position)

    def endpoint_positions(self, values: np.ndarray) -> dict[object, tuple[float, float]]:
        nodes = self.numeric_positions(values)
        geometry: dict[object, tuple[float, float]] = {
            wall: tuple(nodes[self.system._node(wall.attachment)])
            for wall in self.system.walls
        }
        for rod in self.system.rods:
            geometry[rod.start] = tuple(nodes[self.system._node(rod.start)])
            geometry[rod.end] = tuple(nodes[self.system._node(rod.end)])
        for mass in self.system.masses:
            geometry[mass] = tuple(nodes[self.system._node(mass.attachment)])
        for spring in self.system.springs:
            geometry[spring.start] = tuple(nodes[self.system._node(spring.start)])
            geometry[spring.end] = tuple(nodes[self.system._node(spring.end)])
        return geometry

    def _find_rod_hinges(self) -> dict[Rod, Hinge | None]:
        candidates: dict[Rod, list[Hinge]] = {}
        for rod in self.system.rods:
            rod_hinges = [
                hinge
                for hinge in self.hinges
                if rod.start in (hinge.first, hinge.second)
                or rod.end in (hinge.first, hinge.second)
            ]
            unique_hinges: list[Hinge] = []
            for hinge in rod_hinges:
                if hinge not in unique_hinges:
                    unique_hinges.append(hinge)
            candidates[rod] = unique_hinges

        result: dict[Rod, Hinge | None] = {
            rod: None
            for rod in self.system.rods
        }

        assigned_hinge_ids: set[int] = set()

        # Prefer wall-referenced hinges for absolute orientation anchoring.
        for rod in self.system.rods:
            wall_hinges = [
                hinge
                for hinge in candidates[rod]
                if self._hinge_touches_wall(hinge)
            ]

            if len(wall_hinges) > 1:
                raise ValueError(
                    "A Rod cannot be controlled by multiple wall hinges."
                )

            if len(wall_hinges) == 1:
                hinge = wall_hinges[0]
                result[rod] = hinge
                assigned_hinge_ids.add(id(hinge))

        unresolved = [
            rod
            for rod in self.system.rods
            if result[rod] is None and candidates[rod]
        ]

        while unresolved:
            progressed = False

            for rod in list(unresolved):
                available = [
                    hinge
                    for hinge in candidates[rod]
                    if id(hinge) not in assigned_hinge_ids
                ]

                if len(available) == 1:
                    hinge = available[0]
                    result[rod] = hinge
                    assigned_hinge_ids.add(id(hinge))
                    unresolved.remove(rod)
                    progressed = True
                    continue

                if not available:
                    raise ValueError(
                        "Action cannot uniquely determine rod orientation: "
                        "all hinge candidates for a rod are already controlling "
                        "other rods."
                    )

            if not progressed:
                raise ValueError(
                    "Unable to assign controlling hinges uniquely for all rods."
                )

        return result

    @staticmethod
    def _hinge_touches_wall(hinge: Hinge) -> bool:
        return any(
            isinstance(endpoint.owner, Wall)
            for endpoint in (hinge.first, hinge.second)
        )

    def _find_fixed_rod_neighbors(
        self,
    ) -> dict[Rod, list[tuple[AttachmentPoint, Rod]]]:
        neighbors: dict[Rod, list[tuple[AttachmentPoint, Rod]]] = {
            rod: []
            for rod in self.system.rods
        }

        for connection in self.system.connections:
            if not isinstance(connection, Fixed):
                continue

            first_owner = connection.first.owner
            second_owner = connection.second.owner

            if not isinstance(first_owner, Rod) or not isinstance(second_owner, Rod):
                continue

            neighbors[first_owner].append((connection.first, second_owner))
            neighbors[second_owner].append((connection.second, first_owner))

        return neighbors

    def _free_orientation_roots(self) -> list[Rod]:
        hinge_controlled = {
            rod
            for rod, hinge in self._rod_hinge.items()
            if hinge is not None
        }

        visited: set[Rod] = set()
        roots: list[Rod] = []

        for rod in self.system.rods:
            if rod in visited:
                continue

            stack = [rod]
            component: list[Rod] = []

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for _, neighbor in self._fixed_rod_neighbors[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            if any(value in hinge_controlled for value in component):
                continue

            roots.append(component[0])

        return roots

    def _root_masses(self) -> list[Mass]:
        """Find free mass nodes that must seed an otherwise unanchored rod tree."""
        pivot_nodes: set[int] = set()
        output_nodes: set[int] = set()

        for rod, hinge in self._rod_hinge.items():
            if hinge is not None:
                pivot = (
                    rod.start
                    if rod.start in (hinge.first, hinge.second)
                    else rod.end
                )
            else:
                fixed_neighbors = self._fixed_rod_neighbors[rod]
                pivot = fixed_neighbors[0][0] if fixed_neighbors else rod.start

            other = rod.end if pivot is rod.start else rod.start
            pivot_nodes.add(self.system._node(pivot))
            output_nodes.add(self.system._node(other))

        wall_nodes = {
            self.system._node(wall.attachment)
            for wall in self.system.walls
        }

        return [
            mass
            for mass in self.system.masses
            if self.system._node(mass.attachment)
            in pivot_nodes - output_nodes - wall_nodes
        ]

    def _propagate(
        self,
        symbolic: bool,
    ) -> tuple[dict[int, sp.Matrix], dict[Rod, sp.Expr]]:
        del symbolic

        positions = {
            self.system._node(wall.attachment): sp.Matrix(position)
            for wall, position in self._wall_positions.items()
        }
        positions.update(
            {
                self.system._node(mass.attachment): sp.Matrix(
                    self._mass_symbols[mass]
                )
                for mass in self.free_masses
            }
        )

        angles: dict[Rod, sp.Expr] = {}
        unresolved = list(self.system.rods)

        while unresolved:
            progressed = False

            for rod in list(unresolved):
                hinge = self._rod_hinge[rod]
                selected: tuple[AttachmentPoint, sp.Expr] | None = None

                if hinge is not None:
                    pivot = (
                        rod.start
                        if rod.start in (hinge.first, hinge.second)
                        else rod.end
                    )
                    pivot_node = self.system._node(pivot)

                    if pivot_node not in positions:
                        continue

                    parent_angle = self._hinge_parent_angle(
                        hinge,
                        rod,
                        pivot,
                        angles,
                    )

                    if parent_angle is None:
                        continue

                    selected = (
                        pivot,
                        parent_angle + self._hinge_symbols[id(hinge)],
                    )

                else:
                    for endpoint, neighbor in self._fixed_rod_neighbors[rod]:
                        pivot_node = self.system._node(endpoint)
                        if pivot_node in positions and neighbor in angles:
                            selected = (endpoint, angles[neighbor])
                            break

                    if selected is None and rod in self._free_rod_symbols:
                        if self.system._node(rod.start) in positions:
                            selected = (rod.start, self._free_rod_symbols[rod])
                        elif self.system._node(rod.end) in positions:
                            selected = (rod.end, self._free_rod_symbols[rod])

                    if selected is None:
                        continue

                pivot, angle = selected
                pivot_node = self.system._node(pivot)
                other = rod.end if pivot is rod.start else rod.start
                sign = 1 if pivot is rod.start else -1
                other_node = self.system._node(other)

                if other_node in positions:
                    raise NotImplementedError(
                        "Closed rod loops are unsupported in v1."
                    )

                positions[other_node] = (
                    positions[pivot_node]
                    + sign
                    * rod.length
                    * sp.Matrix((sp.cos(angle), sp.sin(angle)))
                )

                angles[rod] = angle
                unresolved.remove(rod)
                progressed = True

            if unresolved and not progressed:
                raise NotImplementedError(
                    "Closed or unanchored rod loops are not supported in v1."
                )

        return positions, angles

    def _hinge_parent_angle(
        self,
        hinge: Hinge,
        child_rod: Rod,
        pivot: AttachmentPoint,
        angles: dict[Rod, sp.Expr],
    ) -> sp.Expr | None:
        opposite = hinge.second if pivot is hinge.first else hinge.first
        owner = opposite.owner

        if isinstance(owner, Rod) and owner is not child_rod:
            return angles.get(owner)

        if isinstance(owner, Wall):
            return self._hinge_wall_reference_angle(hinge)

        pivot_node = self.system._node(pivot)
        resolved_parent_rods = [
            point.owner
            for point in self.system._points()
            if self.system._node(point) == pivot_node
            and isinstance(point.owner, Rod)
            and point.owner is not child_rod
            and point.owner in angles
        ]

        if len(resolved_parent_rods) == 1:
            return angles[resolved_parent_rods[0]]

        if len(resolved_parent_rods) > 1:
            raise ValueError(
                "Action cannot uniquely determine hinge parent rod at a shared node."
            )

        return sp.Integer(0)

    @staticmethod
    def _hinge_wall_reference_angle(
        hinge: Hinge,
    ) -> sp.Expr:
        for endpoint in (hinge.first, hinge.second):
            owner = endpoint.owner
            if isinstance(owner, Wall):
                return sp.pi / 2 if owner.orientation == "vertical" else sp.Integer(0)

        return sp.Integer(0)

    def _validate_all_nodes_resolved(self) -> None:
        unresolved = [
            point
            for point in self.system._points()
            if self.system._node(point) not in self._symbolic_nodes
        ]
        if unresolved:
            raise NotImplementedError(
                "This topology has unresolved attachment geometry."
            )
