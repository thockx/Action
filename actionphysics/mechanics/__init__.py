"""
Mechanics module - Core physics engine for Lagrangian mechanics.

Contains:
- LagrangianSystem: Automatic Euler-Lagrange equation derivation and ODE solving
- Coordinate: Abstraction for generalized coordinates
"""

from .lagrangian_system import LagrangianSystem
from .coordinates import Coordinate

__all__ = ["LagrangianSystem", "Coordinate"]
