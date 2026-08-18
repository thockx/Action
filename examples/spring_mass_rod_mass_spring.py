import numpy as np
from manim import Scene, WHITE

from action import *


class MassBetweenSprings(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        with System() as system:
            wall = Wall()
            rod1, rod2 = Rod(length=1), Rod(length=1)
            mass1 = Mass(m=1, label="m_1")
            mass2 = Mass(m=1, label="m_2")

            hinge = Hinge(wall, rod1.start)
            Fixed(rod1.end, mass1)
            Fixed(mass1, rod2.start)
            Fixed(rod2.end, mass2)

            Gravity(g=9.81)

            Velocity(mass2)
            #Acceleration(mass2)
            #Force(mass2)

            system.initial = {
                hinge.rotation: np.pi / 4,
                hinge.rotation.rate: 0.0,
            }
            system.show_equations = True

        self.add(system)
        self.play(system.simulate(15))