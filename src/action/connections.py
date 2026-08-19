"""Explicit mechanical connection types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, Union

from .components import AttachmentPoint
from .context import register_connection, register_field
from .coordinates import Coordinate

class HasAttachment(Protocol):
    attachment: AttachmentPoint


Endpoint = Union[AttachmentPoint, HasAttachment]


def _attachment(endpoint: Endpoint) -> AttachmentPoint:
    if isinstance(endpoint, AttachmentPoint):
        return endpoint
    return endpoint.attachment


@dataclass
class Connection:
    """Base class for typed topology edges."""

    first: AttachmentPoint
    second: AttachmentPoint

    def __post_init__(self) -> None:
        self.first.connections.append(self)
        self.second.connections.append(self)
        register_connection(self)


@dataclass
class Hinge(Connection):
    """Position-constraining connection with one exposed relative rotation."""

    rotation: Coordinate = field(default_factory=lambda: Coordinate("hinge.rotation"))

    def __init__(self, first: Endpoint, second: Endpoint) -> None:
        super().__init__(_attachment(first), _attachment(second))
        self.rotation = Coordinate("hinge.rotation")


@dataclass
class Fixed(Connection):
    """Rigid attachment that introduces no coordinate."""

    def __init__(self, first: Endpoint, second: Endpoint) -> None:
        super().__init__(_attachment(first), _attachment(second))


@dataclass(frozen=True)
class Gravity:
    """Uniform gravitational field applied to every massive component."""

    g: float = 9.81
    _active: ClassVar["Gravity | None"] = None

    def __post_init__(self) -> None:
        if self.g <= 0:
            raise ValueError("Gravity g must be positive.")
        type(self)._active = self
        register_field(self)

    @property
    def vector(self) -> tuple[float, float]:
        return (0.0, -self.g)

    @classmethod
    def active(cls) -> "Gravity":
        """Return the most recently declared field, or Earth's standard gravity."""
        return cls._active or cls()
