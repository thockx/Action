# Action Physics

**Classical mechanics simulation library using Lagrangian mechanics and automatic Manim visualization**

Action is a Python library that makes it easy to create beautiful, physically accurate simulations of classical mechanics systems. Named after the [principle of least action](https://en.wikipedia.org/wiki/Principle_of_least_action), it automatically derives equations of motion from energy expressions and animates the results using [Manim](https://www.manim.community/).

## ✨ Features

- **🎯 Automatic equation derivation**: Define kinetic and potential energy, get equations of motion automatically via Euler-Lagrange equations
- **🔧 Two API styles**: 
  - Symbolic API for custom physics (direct SymPy expressions)
  - Object-based API for standard components (masses, springs, gravity)
- **📊 Multi-DOF support**: Handle complex 2D systems with multiple degrees of freedom
- **🎬 Automatic visualization**: LaTeX-like automatic rendering with sensible defaults
- **🔬 Accurate numerics**: High-precision ODE solving using SciPy
- **🎨 Customizable**: Full control over visual styling when you need it

## 🚀 Quick Start

### Installation

```bash
# Install in development mode
cd path/to/actionphysics
pip install -e .
```

### Simple Example (Object-Based API)

```python
from manim import *
from actionphysics import System
from actionphysics.visualization import PhysicsRenderer

# Create system
system = System()

# Add a mass
mass = system.add_mass(mass=1.0, position=[0, 0])

# Add spring
wall = system.add_fixed_point(position=[0, 3.0])
system.add_spring(k=10, connect=[wall, mass])

# Add gravity
system.add_gravity(g=9.8)

# Solve
system.solve(duration=5.0)

# Create animation
class MySimulation(PhysicsRenderer):
    def __init__(self):
        super().__init__(system, duration=5.0)
```

Render with:
```bash
manim -pql my_simulation.py MySimulation
```

### Advanced Example (Symbolic API)

```python
from sympy import symbols, diff
from sympy.physics.mechanics import dynamicsymbols
from actionphysics import LagrangianSystem

# Define symbols
t = symbols('t')
y = dynamicsymbols('y')
y_dot = diff(y, t)
m, k = symbols('m k', positive=True)

# Define energies
T = m * y_dot**2 / 2  # Kinetic energy
V = k * y**2 / 2      # Potential energy

# Create and solve system
system = LagrangianSystem(
    T=T, V=V,
    coords=[y],
    params={m: 1.0, k: 10.0}
)
system.solve(duration=5.0, initial_conditions=[1.0, 0.0])
```

## 📚 Examples

See the [`examples/`](examples/) directory for comprehensive examples:

1. **Spring-Mass System** - Simple harmonic oscillator (symbolic API)
2. **Spring-Mass System** - Same system using object-based API  
3. **Double Pendulum** - Classic chaotic system with 4 DOF
4. **Coupled Oscillators** - Two masses with three springs

Run any example:
```bash
cd examples
manim -pql example_01_spring_mass_symbolic.py SpringMassSymbolic
```

## 🏗️ Architecture

### Package Structure

```
actionphysics/
├── mechanics/          # Physics engine
│   ├── lagrangian_system.py   # Automatic Euler-Lagrange derivation
│   └── coordinates.py         # Generalized coordinates
├── objects/            # Object-based API
│   ├── system.py              # System builder
│   └── components.py          # Mass, Spring, Damper, etc.
└── visualization/      # Manim rendering
    ├── renderer.py            # PhysicsRenderer scene
    └── config.py              # Visual configuration
```

### How It Works

1. **Define System**: Use either symbolic energy expressions or physical components
2. **Automatic Derivation**: SymPy computes Euler-Lagrange equations: `d/dt(∂L/∂q̇) - ∂L/∂q = 0`
3. **Numerical Solving**: SciPy's `solve_ivp` integrates the ODEs
4. **Visualization**: Manim animates the solution with automatic vector scaling and layout

## 🎨 Customization

Control visualization with `VisualConfig`:

```python
from actionphysics.visualization import VisualConfig

config = VisualConfig(
    show_velocity_vectors=True,
    show_acceleration_vectors=True,
    show_equations=False,
    velocity_color=BLUE,
    acceleration_color=GREEN,
    mass_radius=0.4,
    vector_max_length=0.3,  # 30% of screen
)

class MyScene(PhysicsRenderer):
    def __init__(self):
        super().__init__(system, duration=5.0, config=config)
```

## 🔬 Physical Components

### Available Components

- **Mass**: Point mass with position and velocity
- **FixedPoint**: Immovable anchor point (walls, pivots)
- **Spring**: Linear spring (Hooke's law)
- **Gravity**: Uniform gravitational field
- **Damper**: Viscous damping (coming soon)

### Creating Components

```python
system = System()

# Masses
m1 = system.add_mass(mass=1.0, position=[0, 0], velocity=[1, 0])
m2 = system.add_mass(mass=2.0, position=[1, 0])

# Springs
system.add_spring(k=10, connect=[m1, m2], rest_length=1.0)

# Gravity (default: downward)
system.add_gravity(g=9.8)

# Custom direction
system.add_gravity(g=5.0, direction=[1, -1])  # Diagonal
```

## 🧮 Coordinate Systems

Action supports:
- **Cartesian coordinates** (x, y) - Default for 2D systems
- **Generalized coordinates** (q₁, q₂, ...) - Symbolic API
- **Automatic assignment** - Object API creates coordinates automatically

## ⚙️ Advanced Usage

### Stiff Systems

For systems with widely varying timescales (like double pendulums with rigid rods):

```python
system.solve(
    duration=10.0,
    method='Radau',  # Implicit method for stiff ODEs
    rtol=1e-6,
    atol=1e-8
)
```

### Energy Conservation

Check energy conservation by increasing tolerances:

```python
system.solve(
    duration=10.0,
    rtol=1e-10,  # Stricter relative tolerance
    atol=1e-12   # Stricter absolute tolerance
)
```

### Custom Initial Conditions

```python
# Symbolic API: provide full state vector [q₁, q̇₁, q₂, q̇₂, ...]
system.solve(
    duration=5.0,
    initial_conditions=[1.0, 0.0, 0.5, 1.0]  # [pos1, vel1, pos2, vel2]
)

# Object API: set initial conditions when creating masses
mass = system.add_mass(
    mass=1.0,
    position=[1.0, 0.5],   # Initial position
    velocity=[0.0, 1.0]    # Initial velocity
)
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/yourusername/actionphysics.git
cd actionphysics
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black actionphysics/
ruff check actionphysics/
```

## 📖 API Reference

### LagrangianSystem

```python
LagrangianSystem(T, V, coords, params)
```

**Parameters:**
- `T` (Expr): Kinetic energy expression
- `V` (Expr): Potential energy expression
- `coords` (List[Coordinate | Symbol]): Generalized coordinates
- `params` (Dict[Symbol, float]): Parameter values

**Methods:**
- `solve(duration, initial_conditions, **kwargs)`: Solve ODEs
- `get_position(coord_index, t)`: Get position at time t
- `get_velocity(coord_index, t)`: Get velocity at time t
- `get_state(t)`: Get full state vector at time t

### System

```python
System()
```

**Methods:**
- `add_mass(mass, position, velocity, fixed, name)`: Add point mass
- `add_spring(k, connect, rest_length, name)`: Add spring
- `add_gravity(g, direction)`: Add gravitational field
- `add_fixed_point(position, name)`: Add fixed anchor
- `solve(duration, **kwargs)`: Solve system dynamics
- `render(duration, **kwargs)`: Render with Manim

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Constraint handling (Lagrange multipliers for true rigid bodies)
- [ ] Damping and dissipation (Rayleigh dissipation function)
- [ ] 3D visualization support
- [ ] More physical components (pulleys, inclined planes, etc.)
- [ ] Energy/phase space plotting utilities
- [ ] Interactive parameter exploration
- [ ] Performance optimization for large systems

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Manim Community** - Beautiful mathematical animations
- **SymPy** - Symbolic mathematics in Python
- **SciPy** - Numerical integration
- Built with inspiration from classical mechanics pedagogy and the principle of least action

## 📬 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

*"The laws of mechanics are not arbitrary; they are the unique result of a variational principle of stationary action."* - Richard Feynman
