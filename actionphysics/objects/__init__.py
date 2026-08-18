"""
Objects module - High-level object-based API for building physics systems.

Contains:
- System: Main builder class for constructing physics systems
- Mass: Point mass component
- Spring: Linear spring component  
- Damper: Damping force component
- Gravity: Gravitational field component
"""

from .system import System
from .components import Mass, Spring, Damper, Gravity, FixedPoint

__all__ = ["System", "Mass", "Spring", "Damper", "Gravity", "FixedPoint"]
