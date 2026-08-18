"""Physical components used to describe an Action system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .context import register_object
from .coordinates import Coordinate

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
        self.start = AttachmentPoint(self, "start")
        self.end = AttachmentPoint(self, "end")
        register_object(self)


@dataclass(eq=False)
class Wall:
    """A stationary visual support and attachment location in SI coordinates."""

    position: tuple[float, float] | None = None
    orientation: str = "horizontal"
    angle: float = 0.0
    attachment: AttachmentPoint = field(init=False)

    def __post_init__(self) -> None:
        if self.orientation not in {"vertical", "horizontal"}:
            self.orientation = "horizontal"
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
