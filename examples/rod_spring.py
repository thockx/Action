from math import pi

from manim import Scene, WHITE

from action import Fixed, Gravity, Hinge, Mass, Rod, Spring, System, Wall


class RodSpring(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        pivot_wall = Wall()
        spring_wall = Wall()
        rod = Rod(length=2)
        mass = Mass(m=1)
        spring = Spring(k=10, rest_length=1)

        hinge = Hinge(pivot_wall, rod.start)
        Fixed(rod.end, mass)
        Fixed(mass, spring.start)
        Fixed(spring.end, spring_wall)
        Gravity(g=9.81)

        system = System(
            objects=[pivot_wall, spring_wall, rod, mass, spring],
            initial={hinge.rotation: pi / 2, hinge.rotation.rate: 0},
        )
        self.add(system)
        self.play(system.simulate(10))