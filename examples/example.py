from manim import Scene, WHITE
from math import pi

from action import *


class SpringMass(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        with System(
            show_equations="lagrangian",
            equation_dot_notation=True,
        ) as system:
            wall1 = Wall(orientation="horizontal", position=(0,0))
            wall2 = Wall(orientation="horizontal", position=(0,-2))
            mass1 = Mass(m=1, label="m_1")
            mass2 = Mass(m=1, label="m_2")
            mass3 = Mass(m=1, label="m_3")
            spring1 = Spring(k=50, rest_length=1)
            rod1 = Rod(length=1)
            rod2 = Rod(length=1)
            spring2 = Spring(k=100, rest_length=1)

            Fixed(wall1, spring1.start)
            Fixed(spring1.end, mass1)
            hinge2 = Hinge(rod1.start, rod2.start)
            Fixed(mass1, rod1.start)
            Fixed(rod2.end, mass2)
            Fixed(mass2, spring2.start)
            Fixed(spring2.end, mass3)
            hinge1 = Hinge(rod1.end, wall2)

            Gravity(g=9.81)

            system.initial = {hinge1.rotation: -135 * pi/180, hinge2.rotation: pi}

            hinge1.rotation.show()
            hinge2.rotation.show()

        self.add(system)
        self.play(system.simulate(10))