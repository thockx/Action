from manim import Scene, WHITE

from action import Fixed, Mass, Spring, System, Wall


class CoupledSprings(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        wall = Wall()
        first_spring = Spring(k=12, rest_length=1)
        first_mass = Mass(m=1)
        second_spring = Spring(k=18, rest_length=1)
        second_mass = Mass(m=1)

        Fixed(wall, first_spring.start)
        Fixed(first_spring.end, first_mass)
        Fixed(first_mass, second_spring.start)
        Fixed(second_spring.end, second_mass)

        system = System(
            objects=[wall, first_spring, first_mass, second_spring, second_mass],
            initial={
                first_spring.extension: 0.2,
                first_spring.extension.rate: 0,
                second_spring.extension: -0.1,
                second_spring.extension.rate: 0,
            },
        )
        self.add(system)
        self.play(system.simulate(10))