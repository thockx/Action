from math import pi

from manim import Scene, WHITE

from action import Fixed, Gravity, Hinge, Mass, Rod, System, Wall


class DoublePendulum(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        wall = Wall()
        first_rod = Rod(length=1.5)
        second_rod = Rod(length=1.5)
        first_mass = Mass(m=1)
        second_mass = Mass(m=1)

        first_hinge = Hinge(wall, first_rod.start)
        second_hinge = Hinge(first_rod.end, second_rod.start)
        Fixed(first_rod.end, first_mass)
        Fixed(second_rod.end, second_mass)
        Gravity(g=9.81)

        system = System(
            objects=[wall, first_rod, second_rod, first_mass, second_mass],
            initial={
                first_hinge.rotation: pi / 3,
                first_hinge.rotation.rate: 0,
                second_hinge.rotation: pi / 3,
                second_hinge.rotation.rate: 0,
            },
        )
        self.add(system)
        self.play(system.simulate(10))