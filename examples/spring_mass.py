from manim import Scene, WHITE

from action import Fixed, Mass, Spring, System, Wall


class SpringMass(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        wall = Wall()
        spring = Spring(k=10, rest_length=1)
        mass = Mass(m=1)
        Fixed(wall, spring.start)
        Fixed(spring.end, mass)

        system = System(
            objects=[wall, spring, mass],
            initial={spring.extension: 0.3, spring.extension.rate: 0},
        )
        self.add(system)
        self.play(system.simulate(10))