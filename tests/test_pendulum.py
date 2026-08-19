import math

import numpy as np
import pytest
import sympy as sp

from action import Acceleration, Coordinate, Fixed, Force, Gravity, Hinge, Mass, Rod, Spring, System, Velocity, Wall


def pendulum(initial=None):
    wall = Wall()
    rod = Rod(length=2.0)
    mass = Mass(m=1.0)
    hinge = Hinge(wall, rod.start)
    fixed = Fixed(rod.end, mass)
    assert fixed.second is mass.attachment
    return System([wall, mass, rod], initial or {}), hinge


def test_pendulum_has_single_intrinsic_independent_coordinate():
    system, hinge = pendulum()

    assert system.degrees_of_freedom == 1
    assert system.coordinates == (hinge.rotation,)
    assert hinge.rotation.rate.name == "hinge.rotation.rate"


def test_pendulum_generates_equation_and_solves_initial_conditions():
    system, hinge = pendulum()
    system = System(system.objects, {hinge.rotation: math.pi / 4, hinge.rotation.rate: 0.0})

    equation = system.equations_of_motion()[0]
    trajectory = system.solve(1.0)

    assert "Derivative" in str(equation)
    assert trajectory.state[0, 0] == pytest.approx(math.pi / 4)
    assert trajectory.state[1, 0] == pytest.approx(0.0)
    assert len(trajectory.time) == 121


def test_geometry_is_derived_from_hinge_coordinate():
    system, hinge = pendulum()
    system = System(system.objects, {hinge.rotation: 0.0})

    geometry = system.geometry_at(0.0)

    rod = system.rods[0]
    assert geometry[rod.start] == (0.0, 0.0)
    assert geometry[rod.end] == pytest.approx((2.0, 0.0))
    assert geometry[system.masses[0]] == pytest.approx((2.0, 0.0))


def test_mass_vectors_register_without_affecting_pendulum_coordinates():
    with System() as system:
        wall, rod, mass = Wall(), Rod(length=2), Mass(m=1)
        hinge = Hinge(wall, rod.start)
        Fixed(rod.end, mass)
        velocity = Velocity(mass, color="#3366FF")
        acceleration = Acceleration(mass, color="#CC2222")
        force = Force(mass, color="#228833")
        system.initial = {hinge.rotation: math.pi / 4, hinge.rotation.rate: 2}

    assert system.degrees_of_freedom == 1
    assert [item.color for item in system._visualizations] == ["#3366FF", "#CC2222", "#228833"]
    system.solve(0.1)
    assert np.linalg.norm(system._motion_vector(velocity, 0.05)) > 0
    assert np.linalg.norm(system._motion_vector(acceleration, 0.05)) > 0
    assert system._motion_vector(force, 0.05) == pytest.approx(mass.m * system._motion_vector(acceleration, 0.05))


def test_vectors_support_multiple_free_masses_with_custom_colors():
    with System() as system:
        first_wall, second_wall = Wall(), Wall()
        first_spring, second_spring = Spring(k=10), Spring(k=10)
        first_mass, second_mass = Mass(m=1), Mass(m=2)
        Fixed(first_wall, first_spring.start)
        Fixed(first_spring.end, first_mass)
        Fixed(second_wall, second_spring.start)
        Fixed(second_spring.end, second_mass)
        first_vector = Velocity(first_mass, color="yellow")
        second_vector = Force(second_mass, color="orange")
        system.initial = {first_spring.extension: 0.2, second_spring.extension: 0.3}

    assert len(system._visualizations) == 2
    assert first_vector.color == "yellow" and second_vector.color == "orange"
    assert system.degrees_of_freedom == 4


def test_wall_visual_line_aligns_with_physical_attachment():
    system, _ = pendulum()
    wall = system.walls[0]

    system._update_visuals(0.0)

    assert system._wall_mobjects[wall][0].get_center() == pytest.approx(system._visual_point(system.geometry_at(0.0)[wall]))


def test_wall_hatches_render_opposite_the_connected_component():
    wall, spring, mass = Wall(orientation="vertical"), Spring(k=10), Mass(m=1)
    Fixed(wall, spring.start)
    Fixed(spring.end, mass)
    system = System([wall, spring, mass], {spring.extension: 0.2})

    system._update_visuals(0.0)

    support_x = system._wall_mobjects[wall][0].get_center()[0]
    assert all(hatch.get_end()[0] < support_x for hatch in system._wall_mobjects[wall][1:])


def test_horizontal_wall_support_and_hatches_follow_vertical_attachment_direction():
    wall, spring, mass = Wall(orientation="horizontal"), Spring(k=10), Mass(m=1)
    Fixed(wall, spring.start)
    Fixed(spring.end, mass)
    system = System([wall, spring, mass], {spring.extension: 0.2})

    system._update_visuals(0.0)

    support_y = system._wall_mobjects[wall][0].get_center()[1]
    assert all(hatch.get_end()[1] > support_y for hatch in system._wall_mobjects[wall][1:])


def test_standalone_gravity_applies_to_new_systems():
    Gravity(g=3.7)
    system, _ = pendulum()

    assert system.fields[0].g == 3.7


def test_invalid_initial_coordinate_is_explained():
    system, _ = pendulum()

    with pytest.raises(ValueError, match="not exposed intrinsic coordinates"):
        System(system.objects, {Coordinate("unrelated"): 0.0})


def test_wall_spring_mass_uses_extension_as_an_initial_condition():
    wall = Wall(orientation="vertical")
    spring = Spring(k=10, rest_length=1)
    mass = Mass(m=1)
    Fixed(wall, spring.start)
    Fixed(spring.end, mass)

    system = System([wall, spring, mass], {spring.extension: 0.2, spring.extension.rate: 0})

    assert system.degrees_of_freedom == 2
    assert system.geometry_at(0)[mass] == pytest.approx((1.2, 0.0))
    assert system.solve(0.1).state.shape[0] == 4


def test_mass_between_two_springs_uses_one_shared_mass_topology_node():
    left_wall, right_wall, mass = Wall(orientation="vertical"), Wall(orientation="vertical"), Mass(m=1)
    left_spring, right_spring = Spring(k=10, rest_length=1), Spring(k=10, rest_length=1)
    Fixed(left_wall, left_spring.start)
    Fixed(left_spring.end, mass)
    Fixed(mass, right_spring.start)
    Fixed(right_spring.end, right_wall)

    system = System(
        [left_wall, right_wall, mass, left_spring, right_spring],
        {left_spring.extension: 0.2, right_spring.extension: -0.2},
    )

    geometry = system.geometry_at(0)
    assert geometry[mass] == pytest.approx((1.2, 0.0))
    assert np.linalg.norm(np.array(geometry[right_spring.end]) - np.array(geometry[right_spring.start])) == pytest.approx(0.8)


def test_serial_rods_compile_to_two_hinge_coordinates():
    wall, first_mass, second_mass = Wall(), Mass(m=1), Mass(m=1)
    first_rod, second_rod = Rod(length=1), Rod(length=1)
    first_hinge = Hinge(wall, first_rod.start)
    second_hinge = Hinge(first_mass, second_rod.start)
    Fixed(first_rod.end, first_mass)
    Fixed(second_rod.end, second_mass)

    system = System(
        [wall, first_mass, second_mass, first_rod, second_rod],
        {first_hinge.rotation: math.pi / 2, second_hinge.rotation: math.pi / 2},
    )

    assert system.degrees_of_freedom == 2
    assert system.solve(0.1).state.shape == (4, 13)


def test_direct_rod_endpoint_hinge_controls_the_child_rod():
    wall, mass = Wall(), Mass(m=1)
    first_rod, second_rod = Rod(length=1), Rod(length=1)
    first_hinge = Hinge(wall, first_rod.start)
    second_hinge = Hinge(first_rod.end, second_rod.start)
    Fixed(second_rod.end, mass)

    system = System(
        [wall, first_rod, second_rod, mass],
        {first_hinge.rotation: math.pi / 2, second_hinge.rotation: math.pi / 2},
    )

    assert system.coordinates[:2] == (first_hinge.rotation, second_hinge.rotation)
    assert system.geometry_at(0)[mass] == pytest.approx((-1, 1))


def test_rod_and_spring_share_one_hinge_configuration_coordinate():
    first_wall, second_wall = Wall(orientation="vertical"), Wall(orientation="vertical")
    rod, mass, spring = Rod(length=2), Mass(m=1), Spring(k=10, rest_length=1)
    hinge = Hinge(first_wall, rod.start)
    Fixed(rod.end, mass)
    Fixed(mass, spring.start)
    Fixed(spring.end, second_wall)

    system = System(
        [first_wall, second_wall, rod, mass, spring],
        {hinge.rotation: math.pi / 2, hinge.rotation.rate: 0},
    )

    state = system._initial_state()
    geometry = system.geometry_at(0)
    assert system.degrees_of_freedom == 1
    assert spring.extension not in system.coordinates
    assert geometry[mass] == pytest.approx((-2, 0))
    assert system._spring_extensions(state[:1])[0] == pytest.approx(3.0)
    assert len(system.equations_of_motion()) == 1
    assert system.solve(0.1).state.shape == (2, 13)


def test_two_springs_are_derived_from_the_same_rod_hinge_coordinate():
    pivot, upper_wall, lower_wall = Wall(), Wall(position=(2, 2)), Wall(position=(2, -2))
    rod, mass = Rod(length=2), Mass(m=1)
    upper_spring = Spring(k=10, rest_length=1)
    lower_spring = Spring(k=20, rest_length=1)
    hinge = Hinge(pivot, rod.start)
    Fixed(rod.end, mass)
    Fixed(mass, upper_spring.start)
    Fixed(upper_spring.end, upper_wall)
    Fixed(mass, lower_spring.start)
    Fixed(lower_spring.end, lower_wall)

    system = System(
        [pivot, upper_wall, lower_wall, rod, mass, upper_spring, lower_spring],
        {hinge.rotation: math.pi / 2},
    )

    q = system._initial_state()[:1]
    extensions = system._spring_extensions(q)
    assert system.degrees_of_freedom == 1
    assert upper_spring.extension not in system.coordinates
    assert lower_spring.extension not in system.coordinates
    assert len(system.equations_of_motion()) == 1
    assert extensions[0] == pytest.approx(1.0)
    assert extensions[1] == pytest.approx(math.sqrt(20) - 1)
    assert system.solve(0.1).state.shape == (2, 13)


def test_fixed_rod_to_rod_keeps_shared_orientation():
    wall = Wall()
    first_rod = Rod(length=1)
    second_rod = Rod(length=1)
    mass = Mass(m=1)
    hinge = Hinge(wall, first_rod.start)
    Fixed(first_rod.end, second_rod.start)
    Fixed(second_rod.end, mass)

    system = System(
        [wall, first_rod, second_rod, mass],
        {hinge.rotation: math.pi / 2},
    )

    geometry = system.geometry_at(0)
    assert system.degrees_of_freedom == 1
    assert system.coordinates == (hinge.rotation,)
    assert geometry[first_rod.end] == pytest.approx((0.0, 1.0))
    assert geometry[second_rod.end] == pytest.approx((0.0, 2.0))


def test_hinged_rod_to_rod_allows_independent_relative_angle():
    wall = Wall()
    first_rod = Rod(length=1)
    second_rod = Rod(length=1)
    mass = Mass(m=1)
    first_hinge = Hinge(wall, first_rod.start)
    second_hinge = Hinge(first_rod.end, second_rod.start)
    Fixed(second_rod.end, mass)

    system = System(
        [wall, first_rod, second_rod, mass],
        {
            first_hinge.rotation: math.pi / 2,
            second_hinge.rotation: math.pi / 2,
        },
    )

    geometry = system.geometry_at(0)
    assert system.degrees_of_freedom == 2
    assert system.coordinates[:2] == (first_hinge.rotation, second_hinge.rotation)
    assert geometry[first_rod.end] == pytest.approx((0.0, 1.0))
    assert geometry[second_rod.end] == pytest.approx((-1.0, 1.0))


def test_fixed_connection_does_not_add_generalized_coordinate():
    wall = Wall()
    first_rod = Rod(length=1)
    second_rod = Rod(length=1)
    mass = Mass(m=1)
    hinge = Hinge(wall, first_rod.start)
    Fixed(first_rod.end, second_rod.start)
    Fixed(second_rod.end, mass)

    system = System(
        [wall, first_rod, second_rod, mass],
        {hinge.rotation: 0.0},
    )

    assert system.degrees_of_freedom == 1
    assert len(system.equations_of_motion()) == 1


def test_hinge_connection_adds_one_generalized_coordinate():
    wall = Wall()
    first_rod = Rod(length=1)
    second_rod = Rod(length=1)
    mass = Mass(m=1)
    first_hinge = Hinge(wall, first_rod.start)
    second_hinge = Hinge(first_rod.end, second_rod.start)
    Fixed(second_rod.end, mass)

    system = System(
        [wall, first_rod, second_rod, mass],
        {
            first_hinge.rotation: 0.0,
            second_hinge.rotation: 0.0,
        },
    )

    assert system.degrees_of_freedom == 2
    assert len(system.equations_of_motion()) == 2


def test_lagrangian_geometry_uses_fixed_orientation_locking():
    wall = Wall()
    first_rod = Rod(length=1)
    second_rod = Rod(length=1)
    mass = Mass(m=1)
    hinge = Hinge(wall, first_rod.start)
    Fixed(first_rod.end, second_rod.start)
    Fixed(second_rod.end, mass)

    system = System(
        [wall, first_rod, second_rod, mass],
        {hinge.rotation: 0.0},
    )

    positions = system.configuration.symbolic_positions()
    second_end = positions[system._node(second_rod.end)]
    theta = system.configuration.symbols[0]

    assert sp.simplify(second_end[0] - 2 * sp.cos(theta)) == 0
    assert sp.simplify(second_end[1] - 2 * sp.sin(theta)) == 0
    assert len(system.equations_of_motion()) == 1


def test_vertical_wall_hinge_uses_vertical_reference_in_geometry():
    wall = Wall(orientation="vertical")
    rod = Rod(length=2)
    mass = Mass(m=1)
    hinge = Hinge(wall, rod.start)
    Fixed(rod.end, mass)

    system = System(
        [wall, rod, mass],
        {hinge.rotation: 0.0},
    )

    geometry = system.geometry_at(0.0)
    assert geometry[rod.start] == pytest.approx((0.0, 0.0))
    assert geometry[rod.end] == pytest.approx((0.0, 2.0))
    assert geometry[mass] == pytest.approx((0.0, 2.0))


def test_vertical_wall_hinge_visual_reference_direction_points_up():
    with System() as system:
        wall = Wall(orientation="vertical")
        rod = Rod(length=1)
        mass = Mass(m=1)
        hinge = Hinge(wall, rod.start)
        Fixed(rod.end, mass)
        hinge.rotation.show()
        system.initial = {hinge.rotation: 0.0}

    specification = system._hinge_coordinate_visuals[hinge.rotation]
    geometry = system.geometry_at(0.0)
    direction = system._coordinate_reference_direction(
        specification,
        geometry,
    )

    assert direction == pytest.approx(np.array((0.0, 1.0)))


def test_wall_hinge_angle_visual_moves_to_unhatched_side():
    with System() as system:
        wall = Wall(orientation="vertical")
        rod = Rod(length=1)
        mass = Mass(m=1)
        hinge = Hinge(wall, rod.start)
        Fixed(rod.end, mass)
        hinge.rotation.show()
        system.initial = {hinge.rotation: math.pi / 2}

    system._update_visuals(0.0)
    group = system._coordinate_mobjects[hinge.rotation]
    center = system._visual_point(system.geometry_at(0.0)[rod.start])

    assert np.max(group[0].get_points()[:, 0]) > center[0]


def test_three_rods_shared_node_uses_specific_hinge_parent_endpoint():
    wall = Wall()
    root = Rod(length=1)
    left = Rod(length=1)
    right = Rod(length=1)
    left_mass = Mass(m=1)
    right_mass = Mass(m=1)

    root_hinge = Hinge(wall, root.start)
    left_hinge = Hinge(root.end, left.start)
    right_hinge = Hinge(root.end, right.start)
    Fixed(left.end, left_mass)
    Fixed(right.end, right_mass)

    # Intentionally provide non-parent rods first in objects list to verify
    # parent selection is based on hinge endpoints, not list order.
    system = System(
        [wall, left, right, root, left_mass, right_mass],
        {
            root_hinge.rotation: math.pi / 2,
            left_hinge.rotation: math.pi / 2,
            right_hinge.rotation: 0.0,
        },
    )

    geometry = system.geometry_at(0)
    assert geometry[root.end] == pytest.approx((0.0, 1.0))
    assert geometry[left_mass] == pytest.approx((-1.0, 1.0))
    assert geometry[right_mass] == pytest.approx((0.0, 2.0))


def test_consumed_hinge_topology_raises_clear_error():
    wall = Wall()
    first_rod = Rod(length=1)
    second_rod = Rod(length=1)

    Fixed(wall, first_rod.start)
    Hinge(first_rod.end, second_rod.start)

    with pytest.raises(ValueError, match="cannot uniquely determine rod orientation"):
        System([wall, first_rod, second_rod])
