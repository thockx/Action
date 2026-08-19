"""Context-local registration for declarative Action system definitions."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connections import Connection, Gravity
    from .system import System


_active_system: ContextVar["System | None"] = ContextVar("action_active_system", default=None)


def active_system() -> "System | None":
    return _active_system.get()


def push_system(system: "System"):
    return _active_system.set(system)


def pop_system(token: object) -> None:
    _active_system.reset(token)  # type: ignore[arg-type]


def register_object(component: object) -> None:
    system = active_system()
    if system is not None:
        system._register_object(component)


def register_connection(connection: "Connection") -> None:
    system = active_system()
    if system is not None:
        system._register_connection(connection)


def register_field(field: "Gravity") -> None:
    system = active_system()
    if system is not None:
        system._register_field(field)
