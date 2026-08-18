"""
Visualization module - Automatic Manim rendering for physics systems.

Contains:
- PhysicsRenderer: Main rendering engine
- render_simulation: Convenience function for quick rendering
- Visual configuration and styling
"""

from .renderer import PhysicsRenderer, render_simulation
from .config import VisualConfig

__all__ = ["PhysicsRenderer", "render_simulation", "VisualConfig"]
