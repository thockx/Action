"""Mass vector visualization instructions for Action systems."""

from __future__ import annotations

from dataclasses import dataclass

from .components import Mass
from .context import active_system

try:
    from manim import BLUE, GREEN, RED
except ImportError:  # pragma: no cover
    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"


@dataclass(eq=False)
class MassVector:
    """A visual overlay derived from a mass's solved trajectory."""

    mass: Mass
    color: object
    quantity: str

    def __post_init__(self) -> None:
        system = active_system()
        if system is None:
            raise RuntimeError("Mass vectors must be declared inside a 'with System() as system:' block.")
        system._register_visualization(self)


class Velocity(MassVector):
    def __init__(self, mass: Mass, color=BLUE) -> None:  # type: ignore[no-untyped-def]
        super().__init__(mass, color, "velocity")


class Acceleration(MassVector):
    def __init__(self, mass: Mass, color=RED) -> None:  # type: ignore[no-untyped-def]
        super().__init__(mass, color, "acceleration")


class Force(MassVector):
    def __init__(self, mass: Mass, color=GREEN) -> None:  # type: ignore[no-untyped-def]
        super().__init__(mass, color, "force")
