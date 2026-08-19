import math

import numpy as np
import pytest
import sympy as sp

from action import Fixed, Hinge, Mass, Rod, Spring, System, Wall


def _rod_system(objects, initial, hinges, expected_dof):
    system = System(objects, initial)

    assert system.degrees_of_freedom == expected_dof
    assert system.coordinates == tuple(hinge.rotation for hinge in hinges)
    assert all(
        coordinate not in system.coordinates
        for spring in system.springs
        for coordinate in (spring.extension, spring.extension.rate)
    )
    assert len(system._q) == len(system.coordinates)
    assert system._mass_matrix.shape == (expected_dof, expected_dof)

    q = system._initial_state()[:expected_dof]
    dq = np.zeros(expected_dof)
    matrix = np.asarray(system._mass_matrix_fn(*q, *dq), dtype=float)
    assert matrix.shape == (expected_dof, expected_dof)
    assert np.linalg.matrix_rank(matrix, tol=1e-8) == expected_dof

    trajectory = system.solve(0.03, samples_per_second=40)
    assert np.isfinite(trajectory.state).all()

    for time in trajectory.time:
        geometry = system.geometry_at(float(time))
        points = np.asarray(list(geometry.values()), dtype=float)
        assert np.isfinite(points).all()
        for rod in system.rods:
            start = np.asarray(geometry[rod.start])
            end = np.asarray(geometry[rod.end])
            assert np.linalg.norm(end - start) == pytest.approx(rod.length, abs=1e-7)

    return system


def _assert_hinge_controls(system, expected):
    for rod, hinge in expected.items():
        assert system.configuration._rod_hinge[rod] is hinge


def _assert_angle_relation(system, rod, parent_angle, hinge):
    angle = system.configuration._rod_angles[rod]
    symbol = system.configuration._hinge_symbols[id(hinge)]
    assert sp.simplify(angle - parent_angle - symbol) == 0


def test_three_rods_branch_from_one_shared_node():
    wall = Wall()
    root, left, right = Rod(1), Rod(1), Rod(1)
    left_mass, right_mass = Mass(1), Mass(1)
    root_hinge = Hinge(wall, root.start)
    left_hinge = Hinge(root.end, left.start)
    right_hinge = Hinge(root.end, right.start)
    Fixed(left.end, left_mass)
    Fixed(right.end, right_mass)

    system = _rod_system(
        [wall, left_mass, right_mass, left, right, root],
        {
            root_hinge.rotation: 0.4,
            left_hinge.rotation: 0.3,
            right_hinge.rotation: -0.2,
        },
        [root_hinge, left_hinge, right_hinge],
        3,
    )

    _assert_hinge_controls(
        system,
        {root: root_hinge, left: left_hinge, right: right_hinge},
    )
    root_angle = system.configuration._rod_angles[root]
    _assert_angle_relation(system, left, root_angle, left_hinge)
    _assert_angle_relation(system, right, root_angle, right_hinge)


def test_four_rods_branch_from_one_shared_node():
    wall = Wall()
    root = Rod(1)
    branches = [Rod(1) for _ in range(3)]
    masses = [Mass(1) for _ in branches]
    root_hinge = Hinge(wall, root.start)
    branch_hinges = [Hinge(root.end, branch.start) for branch in branches]
    for branch, mass in zip(branches, masses):
        Fixed(branch.end, mass)

    system = _rod_system(
        [wall, *masses, *branches, root],
        {
            root_hinge.rotation: 0.2,
            **{hinge.rotation: value for hinge, value in zip(branch_hinges, (0.1, -0.2, 0.35))},
        },
        [root_hinge, *branch_hinges],
        4,
    )

    _assert_hinge_controls(
        system,
        {root: root_hinge} | dict(zip(branches, branch_hinges)),
    )


def test_multiple_hinges_at_same_node_use_their_opposite_endpoint_parents():
    wall = Wall()
    root = Rod(1)
    branches = [Rod(1) for _ in range(3)]
    masses = [Mass(1) for _ in branches]
    root_hinge = Hinge(wall, root.start)
    hinges = [Hinge(root.end, branch.start) for branch in branches]
    for branch, mass in zip(branches, masses):
        Fixed(branch.end, mass)

    system = _rod_system(
        [wall, *branches, *masses, root],
        {root_hinge.rotation: 0.1, **{hinge.rotation: 0.2 for hinge in hinges}},
        [root_hinge, *hinges],
        4,
    )

    _assert_hinge_controls(system, {root: root_hinge} | dict(zip(branches, hinges)))
    root_angle = system.configuration._rod_angles[root]
    for branch, hinge in zip(branches, hinges):
        _assert_angle_relation(system, branch, root_angle, hinge)


def test_hinge_and_fixed_connections_can_share_a_branch_node():
    wall = Wall()
    root, fixed_branch, hinged_branch = Rod(1), Rod(1), Rod(1)
    fixed_mass, hinged_mass = Mass(1), Mass(1)
    root_hinge = Hinge(wall, root.start)
    hinged_hinge = Hinge(root.end, hinged_branch.start)
    Fixed(root.end, fixed_branch.start)
    Fixed(fixed_branch.end, fixed_mass)
    Fixed(hinged_branch.end, hinged_mass)

    system = _rod_system(
        [wall, fixed_mass, hinged_mass, fixed_branch, hinged_branch, root],
        {root_hinge.rotation: 0.25, hinged_hinge.rotation: -0.15},
        [root_hinge, hinged_hinge],
        2,
    )

    _assert_hinge_controls(system, {root: root_hinge, hinged_branch: hinged_hinge})
    assert system.configuration._rod_hinge[fixed_branch] is None
    assert system.configuration._rod_angles[fixed_branch] == system.configuration._rod_angles[root]


def test_branched_rods_with_springs_keep_springs_derived():
    wall = Wall()
    root, left, right = Rod(1), Rod(1), Rod(1)
    left_mass, right_mass = Mass(1), Mass(1)
    left_spring, right_spring = Spring(20, 0.5), Spring(20, 0.5)
    root_hinge = Hinge(wall, root.start)
    left_hinge = Hinge(root.end, left.start)
    right_hinge = Hinge(root.end, right.start)
    Fixed(left.end, left_mass)
    Fixed(right.end, right_mass)
    Fixed(left_mass, left_spring.start)
    Fixed(left_spring.end, wall)
    Fixed(right_mass, right_spring.start)
    Fixed(right_spring.end, wall)

    system = _rod_system(
        [wall, root, left, right, left_mass, right_mass, left_spring, right_spring],
        {root_hinge.rotation: 0.3, left_hinge.rotation: 0.15, right_hinge.rotation: -0.2},
        [root_hinge, left_hinge, right_hinge],
        3,
    )

    assert len(system._spring_extensions(system._initial_state()[:3])) == 2
    assert all(spring.extension not in system.coordinates for spring in system.springs)


def test_nested_branch_has_no_unintended_free_coordinates():
    wall = Wall()
    root, branch, nested = Rod(1), Rod(1), Rod(1)
    root_mass, branch_mass, mass = Mass(1), Mass(1), Mass(1)
    root_hinge = Hinge(wall, root.start)
    branch_hinge = Hinge(root.end, branch.start)
    nested_hinge = Hinge(branch.end, nested.start)
    Fixed(root.end, root_mass)
    Fixed(branch.end, branch_mass)
    Fixed(nested.end, mass)

    system = _rod_system(
        [wall, root, branch, nested, root_mass, branch_mass, mass],
        {root_hinge.rotation: 0.2, branch_hinge.rotation: 0.1, nested_hinge.rotation: -0.2},
        [root_hinge, branch_hinge, nested_hinge],
        3,
    )

    _assert_hinge_controls(system, {root: root_hinge, branch: branch_hinge, nested: nested_hinge})
    assert not any(coordinate.name in {"x", "y"} for coordinate in system.coordinates)
    assert system.geometry_at(0)[mass] == pytest.approx(
        (
            math.cos(0.2) + math.cos(0.3) + math.cos(0.1),
            math.sin(0.2) + math.sin(0.3) + math.sin(0.1),
        )
    )


def test_multiple_independent_branches_share_one_system():
    walls = [Wall(position=(0, 0)), Wall(position=(3, 0))]
    rods = [Rod(1), Rod(1)]
    masses = [Mass(1), Mass(1)]
    hinges = [Hinge(wall, rod.start) for wall, rod in zip(walls, rods)]
    for rod, mass in zip(rods, masses):
        Fixed(rod.end, mass)

    system = _rod_system(
        [*walls, *rods, *masses],
        {hinges[0].rotation: 0.2, hinges[1].rotation: -0.4},
        hinges,
        2,
    )

    _assert_hinge_controls(system, dict(zip(rods, hinges)))


def test_hinge_rotation_is_shared_by_geometry_lagrangian_eom_trajectory_and_show():
    wall = Wall()
    rod = Rod(1)
    mass = Mass(1)
    with System() as system:
        hinge = Hinge(wall, rod.start)
        Fixed(rod.end, mass)
        hinge.rotation.show()
        system.initial = {hinge.rotation: 0.35, hinge.rotation.rate: 0.0}
        system.objects = [wall, rod, mass]

    assert system.coordinates == (hinge.rotation,)
    assert hinge.rotation in system._coordinate_visualizations
    assert system.configuration._rod_angles[rod] == system.configuration._hinge_symbols[id(hinge)]
    assert system._q[0] == system.configuration._hinge_symbols[id(hinge)]
    lagrangian = system._kinetic - system._potential
    assert lagrangian.has(system._q[0])
    functions, _ = system._equation_coordinate_functions(dot_notation=False)
    assert any(equation.has(functions[0]) for equation in system.equations_of_motion())

    trajectory = system.solve(0.02, samples_per_second=20)
    assert trajectory.state[0, 0] == pytest.approx(0.35)
    assert np.isfinite(trajectory.state).all()


def test_internal_hinge_show_renders_pi_relative_angle():
    with System() as system:
        wall = Wall()
        parent, child = Rod(1), Rod(1)
        mass = Mass(1)
        parent_hinge = Hinge(wall, parent.start)
        child_hinge = Hinge(parent.end, child.start)
        Fixed(child.end, mass)
        system.initial = {
            parent_hinge.rotation: 0.0,
            child_hinge.rotation: math.pi,
        }
        child_hinge.rotation.show()

    specification = system._hinge_coordinate_visuals[child_hinge.rotation]
    geometry = system.geometry_at(0.0)
    reference = system._coordinate_reference_direction(specification, geometry)
    current = system._coordinate_current_direction(specification, geometry)

    assert abs(system._coordinate_sweep(specification, reference, current)) == pytest.approx(math.pi)
    assert len(system._coordinate_mobjects[child_hinge.rotation][0].get_points()) > 0


def test_wall_hinge_at_rod_end_displays_its_generalized_angle():
    with System() as system:
        wall = Wall()
        rod = Rod(1)
        mass = Mass(1)
        hinge = Hinge(rod.end, wall)
        Fixed(rod.start, mass)
        hinge.rotation.show()
        system.initial = {hinge.rotation: -0.4}

    specification = system._hinge_coordinate_visuals[hinge.rotation]
    geometry = system.geometry_at(0.0)
    reference = system._coordinate_reference_direction(specification, geometry)
    current = system._coordinate_current_direction(specification, geometry)

    assert system.configuration._rod_angles[rod] == system.configuration._hinge_symbols[id(hinge)]
    assert system._coordinate_sweep(specification, reference, current) == pytest.approx(-0.4)


def test_rod_rod_hinge_displays_relative_angle_with_nonzero_parent():
    with System() as system:
        wall = Wall()
        parent, child = Rod(1), Rod(1)
        mass = Mass(1)
        parent_hinge = Hinge(wall, parent.start)
        child_hinge = Hinge(parent.end, child.start)
        Fixed(child.end, mass)
        parent_hinge.rotation.show()
        child_hinge.rotation.show()
        system.initial = {
            parent_hinge.rotation: 0.8,
            child_hinge.rotation: -0.6,
        }

    specification = system._hinge_coordinate_visuals[child_hinge.rotation]
    geometry = system.geometry_at(0.0)
    reference = system._coordinate_reference_direction(specification, geometry)
    current = system._coordinate_current_direction(specification, geometry)

    assert system._coordinate_sweep(specification, reference, current) == pytest.approx(-0.6)


def test_hinge_angle_display_wraps_across_pi():
    with System() as system:
        wall = Wall()
        rod = Rod(1)
        mass = Mass(1)
        hinge = Hinge(wall, rod.start)
        Fixed(rod.end, mass)
        hinge.rotation.show()
        system.initial = {hinge.rotation: math.pi + 0.2}

    specification = system._hinge_coordinate_visuals[hinge.rotation]
    geometry = system.geometry_at(0.0)
    reference = system._coordinate_reference_direction(specification, geometry)
    current = system._coordinate_current_direction(specification, geometry)

    assert system._coordinate_sweep(specification, reference, current) == pytest.approx(-math.pi + 0.2)


def test_representative_branch_prints_concise_physics_report():
    wall = Wall()
    root, left, right = Rod(1), Rod(1), Rod(1)
    masses = [Mass(1), Mass(1)]
    hinges = [Hinge(wall, root.start), Hinge(root.end, left.start), Hinge(root.end, right.start)]
    Fixed(left.end, masses[0])
    Fixed(right.end, masses[1])
    system = _rod_system(
        [wall, root, left, right, *masses],
        {hinge.rotation: 0.1 for hinge in hinges},
        hinges,
        3,
    )

    trajectory_ok = bool(np.isfinite(system.solve(0.01, samples_per_second=20).state).all())
    rank = np.linalg.matrix_rank(np.asarray(system._mass_matrix_fn(*system._initial_state()[:3], *np.zeros(3)), dtype=float))
    print("topology: root with two hinged child rods")
    print("independent coordinates:", [coordinate.name for coordinate in system.coordinates])
    print("controlling hinges:", [(id(rod), id(hinge)) for rod, hinge in system.configuration._rod_hinge.items() if hinge is not None])
    print("absolute angles:", [(id(rod), str(angle)) for rod, angle in system.configuration._rod_angles.items()])
    print("Lagrangian:", system._kinetic - system._potential)
    print("Euler-Lagrange equations:", system.equations_of_motion())
    print("mass-matrix rank:", rank)
    print("trajectory solved:", trajectory_ok)
    assert rank == 3
    assert trajectory_ok


def test_multiple_wall_hinges_for_one_rod_fail_early():
    wall_one, wall_two = Wall(), Wall()
    rod = Rod(1)
    Hinge(wall_one, rod.start)
    Hinge(wall_two, rod.end)

    with pytest.raises(ValueError, match="multiple wall hinges"):
        System([wall_one, wall_two, rod])


def test_consumed_hinge_candidates_fail_early():
    wall = Wall()
    first, second = Rod(1), Rod(1)
    Fixed(wall, first.start)
    Hinge(first.end, second.start)

    with pytest.raises(ValueError, match="cannot uniquely determine rod orientation"):
        System([wall, first, second])


def test_ambiguous_non_wall_hinge_assignment_fails_early():
    first, second, third = Rod(1), Rod(1), Rod(1)
    Hinge(first.end, second.start)
    Hinge(first.end, third.start)

    with pytest.raises(ValueError, match="cannot uniquely determine rod orientation"):
        System([first, second, third])


def test_closed_branched_loop_is_rejected_as_unsupported():
    wall = Wall()
    first, second = Rod(1), Rod(1)
    Hinge(wall, first.start)
    Hinge(first.end, second.start)
    Fixed(second.end, wall)

    with pytest.raises(NotImplementedError, match="loops"):
        System([wall, first, second])
