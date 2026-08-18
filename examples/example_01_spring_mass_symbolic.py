"""
Example 1: Simple Spring-Mass System (Symbolic API)

Demonstrates the symbolic API where you define kinetic and potential
energy directly using SymPy expressions.
"""

from manim import *
from sympy import symbols, diff
from sympy.physics.mechanics import dynamicsymbols
from actionphysics import LagrangianSystem
from actionphysics.visualization import PhysicsRenderer


# Define symbolic system
t = symbols('t')
y = dynamicsymbols('y')  # Vertical position
y_dot = diff(y, t)

# Parameters
m, k = symbols('m k', positive=True, real=True)

# Energies
T = m * y_dot**2 / 2  # Kinetic energy
V = k * y**2 / 2  # Potential energy (spring from equilibrium)

# Create system
system = LagrangianSystem(
    T=T,
    V=V,
    coords=[y],
    params={
        m: 1.0,  # 1 kg mass
        k: 4 * 3.14159**2,  # ω = 2π rad/s, period = 1 second
    }
)

# Solve with initial conditions: y(0) = 0.8 m, v(0) = 0 m/s
system.solve(
    duration=4.0,
    initial_conditions=[0.8, 0.0]
)


# Create Manim scene
class SpringMassSymbolic(PhysicsRenderer):
    def __init__(self):
        super().__init__(system, duration=4.0)


# To render: manim -pql example_01_spring_mass_symbolic.py SpringMassSymbolic
