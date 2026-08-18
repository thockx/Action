"""Run a readable mechanics validation report for Action v1.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe scripts\\validate_action.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable

import numpy as np
import sympy as sp

from action import Fixed, Gravity, Hinge, Mass, Rod, Spring, System, Wall


@dataclass
class Case:
    name: str
    build: Callable[[], System]
    expected_dofs: int


def simple_pendulum() -> System:
    wall, rod, mass = Wall(), Rod(2), Mass(1)
    hinge = Hinge(wall, rod.start)
    Fixed(rod.end, mass)
    Gravity(9.81)
    return System([wall, rod, mass], {hinge.rotation: pi / 3, hinge.rotation.rate: 0.2})


def double_pendulum() -> System:
    wall, first_rod, second_rod = Wall(), Rod(1.5), Rod(1.5)
    first_mass, second_mass = Mass(1), Mass(1)
    first_hinge = Hinge(wall, first_rod.start)
    second_hinge = Hinge(first_rod.end, second_rod.start)
    Fixed(first_rod.end, first_mass)
    Fixed(second_rod.end, second_mass)
    Gravity(9.81)
    return System(
        [wall, first_rod, second_rod, first_mass, second_mass],
        {first_hinge.rotation: pi / 3, first_hinge.rotation.rate: 0.1, second_hinge.rotation: pi / 4},
    )


def spring_mass() -> System:
    wall, spring, mass = Wall(), Spring(10, 1), Mass(1)
    Fixed(wall, spring.start)
    Fixed(spring.end, mass)
    Gravity(9.81)
    return System([wall, spring, mass], {spring.extension: 0.25, spring.extension.rate: 0.15})


def mass_between_springs() -> System:
    left, right, mass = Wall(), Wall(), Mass(1)
    first, second = Spring(10, 1), Spring(15, 1)
    Fixed(left, first.start)
    Fixed(first.end, mass)
    Fixed(mass, second.start)
    Fixed(second.end, right)
    Gravity(9.81)
    return System([left, right, mass, first, second], {first.extension: 0.2, second.extension: -0.2})


def coupled_springs() -> System:
    left, right = Wall(), Wall(position=(4, 0))
    first, middle, last = Spring(12, 1), Spring(18, 1), Spring(12, 1)
    first_mass, second_mass = Mass(1), Mass(1)
    Fixed(left, first.start)
    Fixed(first.end, first_mass)
    Fixed(first_mass, middle.start)
    Fixed(middle.end, second_mass)
    Fixed(second_mass, last.start)
    Fixed(last.end, right)
    Gravity(9.81)
    return System(
        [left, right, first, middle, last, first_mass, second_mass],
        {first.extension: 0.2, middle.extension: 0.3, last.extension: 0.5},
    )


def rod_spring() -> System:
    pivot, anchor = Wall(), Wall()
    rod, mass, spring = Rod(2), Mass(1), Spring(10, 1)
    hinge = Hinge(pivot, rod.start)
    Fixed(rod.end, mass)
    Fixed(mass, spring.start)
    Fixed(spring.end, anchor)
    Gravity(9.81)
    return System([pivot, anchor, rod, mass, spring], {hinge.rotation: pi / 2, hinge.rotation.rate: 0.1})


def rod_two_springs() -> System:
    pivot, upper, lower = Wall(), Wall(position=(2, 2)), Wall(position=(2, -2))
    rod, mass = Rod(2), Mass(1)
    upper_spring, lower_spring = Spring(10, 1), Spring(20, 1)
    hinge = Hinge(pivot, rod.start)
    Fixed(rod.end, mass)
    Fixed(mass, upper_spring.start)
    Fixed(upper_spring.end, upper)
    Fixed(mass, lower_spring.start)
    Fixed(lower_spring.end, lower)
    Gravity(9.81)
    return System([pivot, upper, lower, rod, mass, upper_spring, lower_spring], {hinge.rotation: pi / 2})


def double_pendulum_spring() -> System:
    pivot, anchor = Wall(), Wall(position=(3, 1))
    first_rod, second_rod = Rod(1.5), Rod(1.5)
    first_mass, second_mass = Mass(1), Mass(1)
    spring = Spring(8, 1)
    first_hinge = Hinge(pivot, first_rod.start)
    second_hinge = Hinge(first_rod.end, second_rod.start)
    Fixed(first_rod.end, first_mass)
    Fixed(second_rod.end, second_mass)
    Fixed(second_mass, spring.start)
    Fixed(spring.end, anchor)
    Gravity(9.81)
    return System(
        [pivot, anchor, first_rod, second_rod, first_mass, second_mass, spring],
        {first_hinge.rotation: pi / 3, second_hinge.rotation: pi / 4},
    )


def independent_pendulums() -> System:
    first_wall, second_wall = Wall(position=(0, 0)), Wall(position=(4, 0))
    first_rod, second_rod = Rod(1), Rod(1.5)
    first_mass, second_mass = Mass(1), Mass(2)
    first_hinge = Hinge(first_wall, first_rod.start)
    second_hinge = Hinge(second_wall, second_rod.start)
    Fixed(first_rod.end, first_mass)
    Fixed(second_rod.end, second_mass)
    Gravity(9.81)
    return System(
        [first_wall, second_wall, first_rod, second_rod, first_mass, second_mass],
        {first_hinge.rotation: pi / 3, second_hinge.rotation: pi / 4},
    )


CASES = [
    Case("Simple pendulum", simple_pendulum, 1),
    Case("Double pendulum", double_pendulum, 2),
    Case("Spring mass", spring_mass, 2),
    Case("Mass between two springs", mass_between_springs, 2),
    Case("Coupled two-mass springs", coupled_springs, 4),
    Case("Rod plus one spring", rod_spring, 1),
    Case("Rod plus two springs", rod_two_springs, 1),
    Case("Double pendulum plus spring", double_pendulum_spring, 2),
    Case("Independent pendulums", independent_pendulums, 2),
]


def energy_function(system: System):
    return sp.lambdify([*system._q, *system._dq], system._kinetic + system._potential, "numpy")


def print_case(case: Case) -> bool:
    system = case.build()
    initial = system._initial_state()
    q_count = system.degrees_of_freedom
    assert q_count == case.expected_dofs, f"expected {case.expected_dofs} DOFs, received {q_count}"
    trajectory = system.solve(2.0)
    assert np.isfinite(trajectory.state).all(), "trajectory contains a non-finite value"

    print(f"\n{'=' * 88}\n{case.name}\n{'=' * 88}")
    print(f"DOFs: {q_count}")
    print("Independent q:")
    for index, spec in enumerate(system.configuration.specs):
        source = getattr(spec.intrinsic, "name", "internal Cartesian mass coordinate")
        print(f"  q{index}: {sp.pretty(spec.symbol)} [{source}] = {initial[index]:.6g}, rate = {initial[q_count + index]:.6g}")
    print("Derived intrinsic spring extensions:")
    if system.springs:
        for spring, extension in zip(system.springs, system._spring_extensions(initial[:q_count])):
            print(f"  {spring.extension.name}: {extension:.6g}")
    else:
        print("  none")

    print("Configuration r(q):")
    for node, position in system.configuration.symbolic_positions().items():
        print(f"  node {node}: {sp.pretty(position)}")
    print("T(q, qdot):")
    sp.pprint(system._kinetic)
    print("V(q):")
    sp.pprint(system._potential)
    print("L = T - V:")
    sp.pprint(sp.simplify(system._kinetic - system._potential))
    print("Euler-Lagrange equations:")
    for equation in system.equations_of_motion():
        sp.pprint(equation)
    print("Numerical solver form M(q) qddot = -b(q, qdot):")
    print("M(q):")
    sp.pprint(system._mass_matrix)
    print("b(q, qdot):")
    sp.pprint(system._bias)

    energy = np.asarray(energy_function(system)(*trajectory.state[:q_count], *trajectory.state[q_count:]), dtype=float)
    drift = (energy.max() - energy.min()) / max(abs(energy[0]), 1e-9)
    print(f"Numerics: {len(trajectory.time)} samples, finite=True, relative energy span={drift:.3e}")
    return True


def analytic_checks() -> None:
    pendulum = simple_pendulum()
    q, dq = pendulum._q, pendulum._dq
    expected = sp.simplify(2 * dq[0] ** 2 - 9.81 * 2 * sp.sin(q[0]))
    assert sp.simplify((pendulum._kinetic - pendulum._potential) - expected) == 0
    equation = pendulum.equations_of_motion()[0].lhs
    acceleration = next(value for value in equation.atoms(sp.Derivative) if value.derivative_count == 2)
    angle = acceleration.expr
    assert sp.simplify(equation.coeff(acceleration) - 4) == 0
    assert sp.simplify((equation - 4 * acceleration).subs(angle, 0) - 19.62) == 0

    spring = spring_mass()
    x, y = spring._q
    spring_force = sp.diff(spring._potential, x).subs({x: sp.Rational(5, 4), y: 0})
    assert sp.simplify(spring_force - sp.Rational(5, 2)) == 0
    print("\nAnalytic checks: simple pendulum matches theta_ddot + (g/L) cos(theta) = 0; at x=1.25, y=0 the spring potential gradient is k(x-L0)=2.5 N.")


def main() -> None:
    passed = 0
    for case in CASES:
        passed += int(print_case(case))
    analytic_checks()
    print(f"\nValidation passed: {passed}/{len(CASES)} systems")


if __name__ == "__main__":
    main()
