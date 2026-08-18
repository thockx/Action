"""
Example 3: Double Pendulum (Object-Based API)

A classic chaotic system with two connected pendulums.
This demonstrates multi-DOF (4 degrees of freedom: x1, y1, x2, y2).
"""

from manim import *
import numpy as np
from actionphysics import System
from actionphysics.visualization import PhysicsRenderer, VisualConfig


# Create system
system = System()

# Fixed pivot point
pivot = system.add_fixed_point(
    position=[0, 2.0],
    name="pivot"
)

# First pendulum bob
L1 = 1.5  # Length of first pendulum
theta1_init = 45 * np.pi / 180  # Initial angle (45 degrees)

mass1 = system.add_mass(
    mass=1.0,
    position=[L1 * np.sin(theta1_init), 2.0 - L1 * np.cos(theta1_init)],
    velocity=[0, 0],
    name="m1"
)

# Second pendulum bob
L2 = 1.0  # Length of second pendulum
theta2_init = -30 * np.pi / 180  # Initial angle (-30 degrees)

mass2 = system.add_mass(
    mass=0.5,
    position=[
        mass1.initial_position[0] + L2 * np.sin(theta2_init),
        mass1.initial_position[1] - L2 * np.cos(theta2_init)
    ],
    velocity=[0, 0],
    name="m2"
)

# Add rigid rods (very stiff springs to simulate rigid connections)
rod_stiffness = 10000.0

rod1 = system.add_spring(
    k=rod_stiffness,
    connect=[pivot, mass1],
    rest_length=L1,
    name="rod1"
)

rod2 = system.add_spring(
    k=rod_stiffness,
    connect=[mass1, mass2],
    rest_length=L2,
    name="rod2"
)

# Add gravity
system.add_gravity(g=9.8)

# Solve the system
print("Solving double pendulum system...")
print("This may take a moment due to the stiffness of the rods...")

system.solve(
    duration=10.0,
    method='Radau',  # Better for stiff systems
    rtol=1e-6,
    atol=1e-8
)


# Custom visual configuration
config = VisualConfig(
    show_equations=False,  # Equations would be too complex to display nicely
    mass_radius=0.2,
    spring_coils=1,  # Minimal coils for rod visualization
    spring_width=0.05,
)


# Create Manim scene
class DoublePendulum(PhysicsRenderer):
    def __init__(self):
        super().__init__(system, duration=10.0, config=config)


# To render: manim -pql example_03_double_pendulum.py DoublePendulum
