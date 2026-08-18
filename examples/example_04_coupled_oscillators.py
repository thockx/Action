"""
Example 4: Coupled Oscillators (Two Masses, Three Springs)

Two masses connected by springs in a line. Demonstrates normal modes
and energy transfer between oscillators.
"""

from manim import *
from actionphysics import System
from actionphysics.visualization import PhysicsRenderer


# Create system
system = System()

# Create two masses horizontally separated
mass1 = system.add_mass(
    mass=1.0,
    position=[-1.5, 0],  # Left mass
    velocity=[0, 0]
)

mass2 = system.add_mass(
    mass=1.0,
    position=[1.5, 0],  # Right mass
    velocity=[0, 0]
)

# Fixed anchor points (walls on each side)
wall_left = system.add_fixed_point(position=[-3, 0], name="wall_left")
wall_right = system.add_fixed_point(position=[3, 0], name="wall_right")

# Three springs
k = 20.0  # Spring constant

spring_left = system.add_spring(
    k=k,
    connect=[wall_left, mass1],
    rest_length=None
)

spring_middle = system.add_spring(
    k=k * 0.5,  # Weaker coupling spring
    connect=[mass1, mass2],
    rest_length=None
)

spring_right = system.add_spring(
    k=k,
    connect=[mass2, wall_right],
    rest_length=None
)

# Initial condition: displace first mass to the right
mass1.initial_position = [-1.0, 0]  # Displaced 0.5 m to the right

# Solve
system.solve(duration=8.0)


# Create Manim scene
class CoupledOscillators(PhysicsRenderer):
    def __init__(self):
        super().__init__(system, duration=8.0)


# To render: manim -pql example_04_coupled_oscillators.py CoupledOscillators
