"""Preview scenes used by scripts/validate_action.py's mechanics matrix."""

from math import pi

from manim import Scene, WHITE

from action import Fixed, Gravity, Hinge, Mass, Rod, Spring, System, Wall


class ValidationScene(Scene):
    def add_system(self, system: System) -> None:
        self.camera.background_color = WHITE
        self.add(system)
        self.play(system.simulate(3))


class SimplePendulum(ValidationScene):
    def construct(self):
        wall, rod, mass = Wall(), Rod(2), Mass(1)
        hinge = Hinge(wall, rod.start)
        Fixed(rod.end, mass)
        Gravity(9.81)
        self.add_system(System([wall, rod, mass], {hinge.rotation: pi / 3, hinge.rotation.rate: 0.2}))


class DoublePendulum(ValidationScene):
    def construct(self):
        wall, first_rod, second_rod = Wall(), Rod(1.5), Rod(1.5)
        first_mass, second_mass = Mass(1), Mass(1)
        first_hinge = Hinge(wall, first_rod.start)
        second_hinge = Hinge(first_rod.end, second_rod.start)
        Fixed(first_rod.end, first_mass)
        Fixed(second_rod.end, second_mass)
        Gravity(9.81)
        self.add_system(System(
            [wall, first_rod, second_rod, first_mass, second_mass],
            {first_hinge.rotation: pi / 3, second_hinge.rotation: pi / 4},
        ))


class SpringMass(ValidationScene):
    def construct(self):
        wall, spring, mass = Wall(), Spring(10, 1), Mass(1)
        Fixed(wall, spring.start)
        Fixed(spring.end, mass)
        Gravity(9.81)
        self.add_system(System([wall, spring, mass], {spring.extension: 0.25, spring.extension.rate: 0.15}))


class MassBetweenSprings(ValidationScene):
    def construct(self):
        left, right, mass = Wall(position=(0, 0)), Wall(position=(2, 0)), Mass(1)
        first, second = Spring(10, 1), Spring(15, 1)
        Fixed(left, first.start)
        Fixed(first.end, mass)
        Fixed(mass, second.start)
        Fixed(second.end, right)
        Gravity(9.81)
        self.add_system(System([left, right, mass, first, second], {first.extension: 0.2, second.extension: -0.2}))


class CoupledSprings(ValidationScene):
    def construct(self):
        left, right = Wall(position=(0, 0)), Wall(position=(4, 0))
        first, middle, last = Spring(12, 1), Spring(18, 1), Spring(12, 1)
        first_mass, second_mass = Mass(1), Mass(1)
        Fixed(left, first.start)
        Fixed(first.end, first_mass)
        Fixed(first_mass, middle.start)
        Fixed(middle.end, second_mass)
        Fixed(second_mass, last.start)
        Fixed(last.end, right)
        Gravity(9.81)
        self.add_system(System(
            [left, right, first, middle, last, first_mass, second_mass],
            {first.extension: 0.2, middle.extension: 0.3, last.extension: 0.5},
        ))


class RodSpring(ValidationScene):
    def construct(self):
        pivot, anchor = Wall(), Wall()
        rod, mass, spring = Rod(2), Mass(1), Spring(10, 1)
        hinge = Hinge(pivot, rod.start)
        Fixed(rod.end, mass)
        Fixed(mass, spring.start)
        Fixed(spring.end, anchor)
        Gravity(9.81)
        self.add_system(System([pivot, anchor, rod, mass, spring], {hinge.rotation: pi / 2, hinge.rotation.rate: 0.1}))


class RodTwoSprings(ValidationScene):
    def construct(self):
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
        self.add_system(System([pivot, upper, lower, rod, mass, upper_spring, lower_spring], {hinge.rotation: pi / 2}))


class DoublePendulumSpring(ValidationScene):
    def construct(self):
        pivot, anchor = Wall(), Wall(position=(3, 1))
        first_rod, second_rod = Rod(1.5), Rod(1.5)
        first_mass, second_mass, spring = Mass(1), Mass(1), Spring(8, 1)
        first_hinge = Hinge(pivot, first_rod.start)
        second_hinge = Hinge(first_rod.end, second_rod.start)
        Fixed(first_rod.end, first_mass)
        Fixed(second_rod.end, second_mass)
        Fixed(second_mass, spring.start)
        Fixed(spring.end, anchor)
        Gravity(9.81)
        self.add_system(System(
            [pivot, anchor, first_rod, second_rod, first_mass, second_mass, spring],
            {first_hinge.rotation: pi / 3, second_hinge.rotation: pi / 4},
        ))
