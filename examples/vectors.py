from math import pi

from manim import ORANGE, PURPLE, YELLOW, Scene, WHITE

from action import Acceleration, Fixed, Force, Gravity, Hinge, Mass, Rod, Spring, System, Velocity, Wall


class PendulumVectors(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        with System() as system:
            wall, rod, mass = Wall(), Rod(length=2), Mass(m=1, label=r"m")
            hinge = Hinge(wall, rod.start)
            Fixed(rod.end, mass)
            Gravity(g=9.81)
            Velocity(mass)
            Acceleration(mass)
            system.initial = {hinge.rotation: pi / 4, hinge.rotation.rate: 2}
        self.add(system)
        self.play(system.simulate(10))


class SpringMassVectors(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        with System() as system:
            wall, spring, mass = Wall(), Spring(k=10), Mass(m=1)
            Fixed(wall, spring.start)
            Fixed(spring.end, mass)
            Velocity(mass, color=YELLOW)
            Acceleration(mass, color=PURPLE)
            Force(mass, color=ORANGE)
            system.initial = {spring.extension: 0.25, spring.extension.rate: 1}
        self.add(system)
        self.play(system.simulate(10))


class RodSpringVectors(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        with System() as system:
            pivot, anchor = Wall(), Wall()
            rod, mass, spring = Rod(length=2), Mass(m=1), Spring(k=10)
            hinge = Hinge(pivot, rod.start)
            Fixed(rod.end, mass)
            Fixed(mass, spring.start)
            Fixed(spring.end, anchor)
            Gravity(g=9.81)
            Velocity(mass)
            Acceleration(mass)
            Force(mass)
            system.initial = {hinge.rotation: pi / 2, hinge.rotation.rate: 1}
        self.add(system)
        self.play(system.simulate(10))