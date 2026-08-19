from manim import Scene, WHITE
from math import pi
from action import *

class SpringMass(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        with System(
            show_equations=False,
            equation_dot_notation=True,
            wall_angle_reference="wall_normal",
            bar_angle_reference="relative"
        ) as system:
            wall1 = Wall(rotation=0, position=(0,0), size=1)
            mass1 = Mass(m=1)
            mass2 = Mass(m=1)
            rod1 = Rod(length=1)
            rod2= Rod(length=1)

            hinge1 = Hinge(wall1, rod1.start)
            Fixed(rod1.end, mass1)
            hinge2 = Hinge(mass1, rod2.start)
            Fixed(rod2.end, mass2)

            Gravity(g=9.81)

            Velocity(mass1)
            Force(mass1)

            system.initial = {hinge1.rotation: pi / 2.5}

            hinge1.rotation.show()
            hinge2.rotation.show()
            rod1.length.show()
            rod2.length.show()

        self.add(system)
        self.play(system.simulate(5))