from manim import Scene, WHITE

from action import Fixed, Mass, Spring, System, Wall


class MassBetweenSprings(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        left_wall, right_wall, mass = Wall(), Wall(), Mass(m=1)
        left_spring = Spring(k=100, rest_length=1)
        right_spring = Spring(k=100, rest_length=1)
        Fixed(left_wall, left_spring.start)
        Fixed(left_spring.end, mass)
        Fixed(mass, right_spring.start)
        Fixed(right_spring.end, right_wall)

        system = System(
            objects=[left_wall, right_wall, mass, left_spring, right_spring],
            initial={left_spring.extension: 0.2, right_spring.extension: -0.2},
        )
        self.add(system)
        self.play(system.simulate(10))