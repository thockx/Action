"""
Example 2: Spring-Mass System (Object-Based API)

Demonstrates the object-based API where you build systems by adding
physical components like masses and springs.
"""

from manim import *
from actionphysics import System
from actionphysics.visualization import PhysicsRenderer


# Create system using object-based API
system = System()

# Add a single mass
mass = system.add_mass(
    mass=1.0,  # 1 kg
    position=[0, -0.5],  # Start at y = -0.5 (equilibrium will be found)
    velocity=[0, 0]  # Start from rest
)

# Add fixed anchor point (wall)
wall = system.add_fixed_point(
    position=[0, 3.0],  # Wall at y = 3
    name="wall"
)

# Add spring connecting wall to mass
spring = system.add_spring(
    k=4 * 3.14159**2,  # Same spring constant as Example 1
    connect=[wall, mass],
    rest_length=None  # Will use initial distance as rest length
)

# Build and solve the system
system.solve(duration=4.0)


# Create Manim scene
class SpringMassObject(PhysicsRenderer):
    def __init__(self):
        super().__init__(system, duration=4.0)


# To render: manim -pql example_02_spring_mass_object.py SpringMassObject
