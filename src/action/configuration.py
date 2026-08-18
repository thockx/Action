"""Generalized-coordinate discovery and shared physical geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from .components import AttachmentPoint, Mass, Rod, Spring, Wall
from .connections import Hinge

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
        self.hinges = [connection for connection in system.connections if isinstance(connection, Hinge)]
        self._rod_hinge = self._find_rod_hinges()
        self._wall_positions = system._layout_walls()
        self.specs: list[CoordinateSpec] = []
        self._hinge_symbols: dict[int, sp.Symbol] = {}
        for index, hinge in enumerate(self.hinges):
            symbol = sp.Symbol(f"hinge_{index}", real=True)
            self.specs.append(CoordinateSpec(symbol, hinge.rotation))
            self._hinge_symbols[id(hinge)] = symbol
        self._free_rod_symbols: dict[Rod, sp.Symbol] = {}
        for index, rod in enumerate(self.system.rods):
            if self._rod_hinge[rod] is None:
                symbol = sp.Symbol(f"rod_{index}_angle", real=True)
                self.specs.append(CoordinateSpec(symbol, rod.rotation))
                self._free_rod_symbols[rod] = symbol
        self._mass_symbols: dict[Mass, tuple[sp.Symbol, sp.Symbol]] = {}
        self.free_masses = self._root_masses()
        for index, mass in enumerate(self.free_masses):
            x, y = sp.symbols(f"mass_{index}_x mass_{index}_y", real=True)
            self.specs.extend((CoordinateSpec(x, None), CoordinateSpec(y, None)))
            self._mass_symbols[mass] = (x, y)
        self._symbolic_nodes, self._rod_angles = self._propagate(symbolic=True)
        for mass in system.masses:
            if system._node(mass.attachment) not in self._symbolic_nodes and mass not in self.free_masses:
                index = len(self.free_masses)
                x, y = sp.symbols(f"mass_{index}_x mass_{index}_y", real=True)
                self.specs.extend((CoordinateSpec(x, None), CoordinateSpec(y, None)))
                self._mass_symbols[mass] = (x, y)
                self.free_masses.append(mass)
                self._symbolic_nodes[system._node(mass.attachment)] = sp.Matrix((x, y))
        self._validate_all_nodes_resolved()

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
            wall: tuple(nodes[self.system._node(wall.attachment)]) for wall in self.system.walls
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
        result: dict[Rod, Hinge | None] = {}
        for rod in self.system.rods:
            start_hinges = [hinge for hinge in self.hinges if rod.start in (hinge.first, hinge.second)]
            end_hinges = [hinge for hinge in self.hinges if rod.end in (hinge.first, hinge.second)]
            if len(start_hinges) == 1:
                result[rod] = start_hinges[0]
            elif not start_hinges and len(end_hinges) == 1:
                result[rod] = end_hinges[0]
            elif not start_hinges and not end_hinges:
                result[rod] = None
            else:
                raise ValueError("Every Rod needs exactly one controlling Hinge at rod.start or rod.end.")
        return result

    def _root_masses(self) -> list[Mass]:
        """Find free mass nodes that must seed an otherwise unanchored rod tree."""
        pivot_nodes: set[int] = set()
        output_nodes: set[int] = set()
        for rod, hinge in self._rod_hinge.items():
            pivot = rod.start if hinge is None or rod.start in (hinge.first, hinge.second) else rod.end
            other = rod.end if pivot is rod.start else rod.start
            pivot_nodes.add(self.system._node(pivot))
            output_nodes.add(self.system._node(other))
        wall_nodes = {self.system._node(wall.attachment) for wall in self.system.walls}
        return [
            mass
            for mass in self.system.masses
            if self.system._node(mass.attachment) in pivot_nodes - output_nodes - wall_nodes
        ]

    def _propagate(self, symbolic: bool) -> tuple[dict[int, sp.Matrix], dict[Rod, sp.Expr]]:
        positions = {
            self.system._node(wall.attachment): sp.Matrix(position)
            for wall, position in self._wall_positions.items()
        }
        positions.update({
            self.system._node(mass.attachment): sp.Matrix(self._mass_symbols[mass])
            for mass in self.free_masses
        })
        angles: dict[Rod, sp.Expr] = {}
        unresolved = list(self.system.rods)
        while unresolved:
            progressed = False
            for rod in list(unresolved):
                hinge = self._rod_hinge[rod]
                pivot = rod.start if hinge is None or rod.start in (hinge.first, hinge.second) else rod.end
                pivot_node = self.system._node(pivot)
                if pivot_node not in positions:
                    continue
                parent_angle = next(
                    (
                        angles[point.owner]
                        for point in self.system._points()
                        if self.system._node(point) == pivot_node
                        and isinstance(point.owner, Rod)
                        and point.owner is not rod
                        and point.owner in angles
                    ),
                    sp.Integer(0),
                )
                local_angle = self._free_rod_symbols[rod] if hinge is None else self._hinge_symbols[id(hinge)]
                angle = parent_angle + local_angle
                other = rod.end if pivot is rod.start else rod.start
                sign = 1 if pivot is rod.start else -1
                positions[self.system._node(other)] = positions[pivot_node] + sign * rod.length * sp.Matrix((sp.cos(angle), sp.sin(angle)))
                angles[rod] = angle
                unresolved.remove(rod)
                progressed = True
            if unresolved and not progressed:
                raise NotImplementedError("Closed or unanchored rod loops are not supported in v1.")
        return positions, angles

    def _validate_all_nodes_resolved(self) -> None:
        unresolved = [
            point for point in self.system._points()
            if self.system._node(point) not in self._symbolic_nodes
        ]
        if unresolved:
            raise NotImplementedError("This topology has unresolved attachment geometry.")
