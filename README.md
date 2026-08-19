# Action

**A simple, user-friendly Python physics simulation library built on [Manim](https://www.manim.community/).**

Action lets you describe classical mechanical systems using physical components and connections, automatically derive their equations of motion, solve them numerically, and animate the resulting motion in Manim.

> Build the mechanism. Define the physics. Let Action handle the equations.

---

## ✨ Features

* 🧩 **Component-based mechanics** — build systems from masses, rods, springs, and walls.
* 🌳 **Branched mechanisms** — model pendulum trees and other branched mechanical topologies.
* 🔗 **Typed connections** — connect components using `Hinge` and `Fixed` constraints.
* ⚙️ **Automatic equations of motion** — Action constructs the system's Lagrangian from its physical configuration.
* 📐 **Generalized coordinates** — coordinates are automatically derived from the topology of the mechanism.
* 🧮 **Numerical simulation** — solve the resulting equations and cache the trajectory.
* 🎬 **Manim integration** — directly animate the simulated system in a normal Manim `Scene`.
* 🏷️ **Physical labels and vectors** — display velocities, accelerations, forces, and other useful quantities.
* 🎨 **Customizable visualization** — control the appearance of the simulation independently from its physical model.

---

## 🚀 Installation

Action is currently under active development.

For development installation:

```bash
python -m pip install -e ".[manim,test]"
```

Once installed, Action can be imported directly:

```python
from action import *
```

---

## 🔭 Quick Example

A simple pendulum can be described without manually deriving its equations of motion:

```python
from math import pi

from manim import Scene
from action import *


class Pendulum(Scene):
    def construct(self):
        with System() as system:
            wall = Wall()
            rod = Rod(length=2)
            mass = Mass(m=1)

            hinge = Hinge(wall, rod.start)
            Fixed(rod.end, mass)

            Gravity(g=9.81)

            system.initial = {
                hinge.rotation: pi / 4,
                hinge.rotation.rate: 0,
            }

        self.add(system)
        self.play(system.simulate(10))
```

Action takes care of the rest:

1. The physical components are registered with the `System`.
2. The connections determine the system's topology.
3. Generalized coordinates are generated automatically.
4. The physical configuration is resolved in SI units.
5. Kinetic and potential energies are constructed.
6. The equations of motion are obtained.
7. The system is numerically solved.
8. The resulting trajectory is rendered through Manim.

The same physical model can therefore be used for both **simulation and visualization**.

---

## 🧩 Building Systems

### Masses, rods, and walls

Physical components are declared directly:

```python
wall = Wall()
rod = Rod(length=2)
mass = Mass(m=1)
```

Connections describe how those components interact:

```python
Hinge(wall, rod.start)
Fixed(rod.end, mass)
```

There is intentionally no generic `connect()` method. Connections have physical meaning, so Action represents them explicitly.

---

## 🌳 Branched Mechanisms

Action is not limited to simple chains.

A rod can support another rod, allowing systems such as:

* double pendulums
* pendulum trees
* branched mechanisms
* mechanisms with multiple masses
* systems where different branches contain different components

The topology of the mechanism determines its generalized coordinates.

For example, a hinged rod can introduce an angular coordinate, while a free mass can contribute translational coordinates.

This means you can describe increasingly complex mechanisms without manually constructing a coordinate system for every component.

---

## 🌀 Springs

Springs are first-class physical components.

```python
with System() as system:
    wall = Wall()
    spring = Spring(k=10, rest_length=1)
    mass = Mass(m=1)

    Fixed(wall, spring.start)
    Fixed(spring.end, mass)

    system.initial = {
        spring.extension: 0.3,
        spring.extension.rate: 0,
    }

self.add(system)
self.play(system.simulate(10))
```

Spring extensions are derived from the physical geometry of their endpoints and contribute elastic potential energy to the system.

Multiple springs can act on the same mass:

```python
with System() as system:
    left = Wall()
    right = Wall()

    spring_left = Spring(k=10, rest_length=1)
    spring_right = Spring(k=10, rest_length=1)
    mass = Mass(m=1)

    Fixed(left, spring_left.start)
    Fixed(spring_left.end, mass)

    Fixed(mass, spring_right.start)
    Fixed(spring_right.end, right)
```

Action checks that spring extensions and endpoint geometry remain physically consistent.

---

## 📊 Vectors

Action can display trajectory-driven physical vectors directly on the simulated system.

```python
Velocity(mass)
Acceleration(mass)
Force(mass)
```

By default:

* **Velocity** is blue
* **Acceleration** is red
* **Force** is green

Colors can be customized using normal Manim colors:

```python
Velocity(mass, color=YELLOW)
Acceleration(mass, color=PURPLE)
Force(mass, color=ORANGE)
```

These overlays are purely visual — they do not modify the physical model.

---

## 🎨 Visualization

Action separates the **physical model** from its **visual representation**.

Physical dimensions are expressed in the same coordinate system as the simulation and are transformed into suitable Manim dimensions only when the system is rendered.

This means that changing the visual scale does not change the underlying physics.

### Custom styles

Visual appearance can be customized through `VisualStyle`:

```python
style = VisualStyle(
    mass_radius=0.14,
    rod_width=0.035,
    spring_amplitude=0.08,
)

with System(visual_style=style) as system:
    ...
```

This allows physical dimensions, colors, label sizes, vector appearance, and other visualization settings to be customized without modifying the physics implementation.

---

## 🏗️ How Action Works

At its core, Action follows a simple pipeline:

```text
Physical components
        │
        ▼
Typed connections
        │
        ▼
Mechanical topology
        │
        ▼
Generalized coordinates
        │
        ▼
Physical configuration
        │
        ├───────────────┐
        ▼               ▼
 Kinetic energy   Potential energy
        │               │
        └───────┬───────┘
                ▼
            Lagrangian
                │
                ▼
      Equations of motion
                │
                ▼
        Numerical solution
                │
                ▼
         Cached trajectory
                │
                ▼
             Manim
```

The important design principle is that **the physical configuration is the source of truth**.

The same configuration is used to:

* determine component positions,
* construct energies,
* generate equations of motion,
* solve the dynamics,
* and construct the visualization.

This keeps the simulation and animation synchronized by construction.

---

## 📁 Examples

The repository contains several complete examples:

```text
examples/
├── pendulum.py
├── double_pendulum.py
├── spring_mass.py
├── mass_between_springs.py
├── coupled_springs.py
├── rod_spring.py
├── rod_two_springs.py
└── vectors.py
```

Render an example using Manim:

```bash
manim -pql examples/pendulum.py Pendulum
```

For example:

```bash
manim -pql examples/double_pendulum.py DoublePendulum
```

---

## 🧪 Development

Run the test suite with:

```bash
python -m pytest
```

Action is currently evolving toward a more general and expressive mechanical modeling framework. Contributions, ideas, bug reports, and experiments are welcome.

---

## 🛣️ Roadmap

Action is still in its early stages. Areas of ongoing development include:

* [ ] More mechanical components
* [ ] More connection types
* [ ] More complex branched mechanisms
* [ ] Improved visualization customization
* [ ] Better documentation and examples
* [ ] Expanded automated testing
* [ ] Performance improvements for larger systems
* [ ] Stable public API and PyPI release

---

## 🤝 Contributing

Contributions are welcome!

If you have an idea for a new component, connection type, visualization feature, or physics capability, feel free to open an issue or pull request.

When contributing, please try to keep the distinction between:

**physical model → simulation → visualization**

clear. This separation is one of the core design principles of Action.

---

## 📜 License

Action is open-source software. See [`LICENSE`](LICENSE) for the full license.

---

## 🔗 Related Projects

Action is built on top of **Manim**, the Python framework for creating mathematical animations.

* [Manim](https://github.com/ManimCommunity/manim)
* [Manim documentation](https://docs.manim.community/)

---

<p align="center">
  <b>Action</b><br>
  Simulate mechanics. Visualize the motion.
</p>
