"""
Action - Classical Mechanics Simulation Library

A Python library for building classical mechanics simulations using Lagrangian mechanics
and automatic Manim visualization. Supports multi-DOF 2D systems with both symbolic and
object-based APIs.

Example usage:

    # Symbolic API (custom physics)
    from actionphysics import LagrangianSystem, render_simulation
    
    T = m * v**2 / 2
    V = k * x**2 / 2
    system = LagrangianSystem(T=T, V=V, coords=[x], params={m: 1.0, k: 10})
    render_simulation(system, duration=5)
    
    # Object API (standard components)
    from actionphysics import System
    
    system = System()
    mass1 = system.add_mass(1.0, position=[0, 0])
    mass2 = system.add_mass(0.5, position=[1, 0])
    system.add_spring(k=10, connect=[mass1, mass2])
    system.add_gravity(g=9.8)
    system.render(duration=5)
"""

__version__ = "0.1.0"

from .mechanics.lagrangian_system import LagrangianSystem
from .objects.system import System
from .visualization.renderer import render_simulation

__all__ = [
    "LagrangianSystem",
    "System",
    "render_simulation",
]
