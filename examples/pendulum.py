from math import pi

from manim import Scene, WHITE

from action import Fixed, Gravity, Hinge, Mass, Rod, System, Wall


class Pendulum(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        wall = Wall()
        mass1 = Mass(m=1)
        mass2 = Mass(m=1)
        rod1 = Rod(length=1)
        rod2 = Rod(length=1)

        hinge1 = Hinge(wall, rod1.start)
        hinge2 = Hinge(mass1, rod2.start)
        Fixed(rod1.end, mass1)
        Fixed(rod2.end, mass2)
        Gravity(g=9.81)

        system = System(
            objects=[wall, mass1, mass2, rod1, rod2],
            initial={
                hinge1.rotation: pi/2, hinge1.rotation.rate: 0.0,
                hinge2.rotation: pi/2, hinge2.rotation.rate: 0.0
            }
        )

        self.add(system)
        self.play(system.simulate(10))
