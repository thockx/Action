from math import pi

from manim import Scene, WHITE

from action import Fixed, Gravity, Hinge, Mass, Rod, Spring, System, Wall


class RodTwoSprings(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        pivot = Wall()
        upper_wall = Wall(position=(2, 2))
        lower_wall = Wall(position=(2, -2))
        rod = Rod(length=2)
        mass = Mass(m=1)
        upper_spring = Spring(k=10, rest_length=1)
        lower_spring = Spring(k=20, rest_length=1)

        hinge = Hinge(pivot, rod.start)
        Fixed(rod.end, mass)
        Fixed(mass, upper_spring.start)
        Fixed(upper_spring.end, upper_wall)
        Fixed(mass, lower_spring.start)
        Fixed(lower_spring.end, lower_wall)
        Gravity(g=9.81)

        system = System(
            objects=[pivot, upper_wall, lower_wall, rod, mass, upper_spring, lower_spring],
            initial={hinge.rotation: pi / 2, hinge.rotation.rate: 0},
        )
        self.add(system)
        self.play(system.simulate(10))
