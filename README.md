# Action

`Action` is a small Python library for describing 2D classical mechanisms and playing their solved motion in a normal [Manim](https://www.manim.community/) scene.

## v1 architecture

Action models a mechanism as a graph of physical components and typed
connections. The user declares `Mass`, `Rod`, `Spring`, and `Wall`, then joins
their supported endpoints with `Hinge(...)` or `Fixed(...)`. No generic
`.connect()` API and no per-frame Manim updater are required.

`System` compiles that graph into independent generalized coordinates and a
shared configuration layer. Given $q$, the configuration resolves every wall,
rod endpoint, mass, and spring endpoint in SI coordinates. The same geometry
feeds generic kinetic and potential energy construction, numerical solving, and
reusable Manim Mobjects. A static physical-to-visual fit is applied only after
solving. Defaults are a white background with black components.

Hinged wall-rooted rod trees contribute hinge-angle coordinates. Masses not
determined by that tree contribute internal $x,y$ coordinates. Springs never
add coordinates: their extensions are derived from endpoint distance and add
elastic potential energy. This lets rods and springs coexist in one Lagrangian,
including one rod-driven mass attached to one or more springs.

## Install

```bash
python -m pip install -e ".[manim,test]"
```

## Basic workflow

```python
from math import pi
from manim import Scene
from action import *

class Pendulum(Scene):
    def construct(self):
        with System() as system:
            wall = Wall()
            mass = Mass(m=1)
            rod = Rod(length=2)

            hinge = Hinge(wall, rod.start)
            Fixed(rod.end, mass)
            Gravity(g=9.81)
            system.initial = {hinge.rotation: pi / 4, hinge.rotation.rate: 0}
        self.add(system)
        self.play(system.simulate(10))
```

The user declares objects, typed mechanical connections, fields, and initial conditions. `System` derives the dependent geometry, generates the equation of motion, solves it once, and presents the cached trajectory through Manim.

`with System() as system:` is the preferred definition form: physical objects
and typed connections created inside the block are registered with that system
automatically. The earlier `System(objects=[...], initial=...)` constructor
remains available for existing code.

## Spring systems

Springs are first-class components with their own intrinsic coordinate:

```python
with System() as system:
    wall = Wall()
    spring = Spring(k=10, rest_length=1)
    mass = Mass(m=1)
    Fixed(wall, spring.start)
    Fixed(spring.end, mass)
    system.initial = {spring.extension: 0.3, spring.extension.rate: 0}
```

For a mass connected to multiple springs, spring extensions are checked against
the shared endpoint geometry. Inconsistent values raise a clear `ValueError`.
See [examples/spring_mass.py](examples/spring_mass.py) and
[examples/mass_between_springs.py](examples/mass_between_springs.py).

## Mass vectors

Trajectory-driven vector overlays belong inside the system context. They do not
change the physical model: `Velocity` is blue, `Acceleration` is red, and net
`Force` is green by default.

```python
Velocity(mass)
Acceleration(mass)
Force(mass)
```

Use normal Manim colors to override a default:

```python
Velocity(mass, color=YELLOW)
Acceleration(mass, color=PURPLE)
Force(mass, color=ORANGE)
```

Run the included example with:

```bash
manim -pql examples/pendulum.py Pendulum
manim -pql examples/double_pendulum.py DoublePendulum
manim -pql examples/spring_mass.py SpringMass
manim -pql examples/mass_between_springs.py MassBetweenSprings
manim -pql examples/coupled_springs.py CoupledSprings
manim -pql examples/rod_spring.py RodSpring
manim -pql examples/rod_two_springs.py RodTwoSprings
manim -pql examples/vectors.py PendulumVectors
```

## Development

```bash
python -m pytest
```
