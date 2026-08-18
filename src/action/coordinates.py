"""Public intrinsic-coordinate objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class CoordinateRate:
    """Time derivative associated with a :class:`Coordinate`."""

    coordinate: "Coordinate"

    @property
    def name(self) -> str:
        return f"{self.coordinate.name}.rate"


@dataclass(frozen=True, eq=False)
class Coordinate:
    """A named coordinate exposed by an Action component or connection."""

    name: str
    _rate: CoordinateRate = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rate", CoordinateRate(self))

    @property
    def rate(self) -> CoordinateRate:
        return self._rate
