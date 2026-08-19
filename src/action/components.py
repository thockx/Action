"""Physical components used to describe an Action system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .context import register_object
from .coordinates import Coordinate


class RodLength(float):
    def __new__(cls, value: float, rod: "Rod | None" = None):
        instance = float.__new__(cls, value)
        instance.rod = rod
        return instance

    def show(self) -> "RodLength":
        from .context import active_system

        system = active_system()
        if system is None:
            raise RuntimeError(
                "Length visualizations must be declared inside a "
                "'with System() as system:' block."
            )
        if self.rod is None:
            raise RuntimeError("Only a Rod's length can be shown.")
        system._register_rod_length_visualization(self.rod)
        return self

if TYPE_CHECKING:
    from .connections import Connection


@dataclass(eq=False)
class AttachmentPoint:
    """A v1 attachment endpoint belonging to a component."""

    owner: object
    name: str
    connections: list["Connection"] = field(default_factory=list, repr=False)


@dataclass(eq=False)
class Mass:
    """A point mass in kilograms."""

    m: float
    label: str = "m"
    attachment: AttachmentPoint = field(init=False)

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("Mass m must be positive.")
        self.attachment = AttachmentPoint(self, "attachment")
        register_object(self)


@dataclass(eq=False)
class Rod:
    """A planar rigid rod with fixed length in metres."""

    length: float
    m: float = 0.0
    rotation: Coordinate = field(default_factory=lambda: Coordinate("rod.rotation"))
    start: AttachmentPoint = field(init=False)
    end: AttachmentPoint = field(init=False)

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("Rod length must be positive.")
        if self.m < 0:
            raise ValueError("Rod mass cannot be negative.")
        object.__setattr__(self, "length", RodLength(self.length, self))
        self.start = AttachmentPoint(self, "start")
        self.end = AttachmentPoint(self, "end")
        register_object(self)


@dataclass(eq=False)
class Wall:
    """A stationary visual support and attachment location in SI coordinates."""

    position: tuple[float, float] | None = None
    rotation: float = 0.0
    size: float = 0.9
    attachment: AttachmentPoint = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.rotation, (int, float)):
            raise ValueError("Wall rotation must be specified in radians.")
        if self.size <= 0:
            raise ValueError("Wall size must be positive.")
        self.attachment = AttachmentPoint(self, "attachment")
        register_object(self)


@dataclass(eq=False)
class Spring:
    """A linear spring. Its extension is the endpoint separation minus rest length."""

    k: float
    rest_length: float = 1.0
    start: AttachmentPoint = field(init=False)
    end: AttachmentPoint = field(init=False)
    extension: Coordinate = field(default_factory=lambda: Coordinate("spring.extension"))

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("Spring k must be positive.")
        if self.rest_length < 0:
            raise ValueError("Spring rest_length cannot be negative.")
        self.start = AttachmentPoint(self, "start")
        self.end = AttachmentPoint(self, "end")
        register_object(self)
