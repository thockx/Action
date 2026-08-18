"""
Example Gallery for Action Physics Library

This directory contains examples demonstrating the capabilities of the
actionphysics library for classical mechanics simulations.

## Examples

### Example 1: Spring-Mass System (Symbolic API)
**File:** `example_01_spring_mass_symbolic.py`
**Demonstrates:** 
- Symbolic API using SymPy expressions
- Direct definition of kinetic and potential energy
- Simple harmonic oscillator
- Automatic Euler-Lagrange equation derivation

**Run:** `manim -pql example_01_spring_mass_symbolic.py SpringMassSymbolic`

---

### Example 2: Spring-Mass System (Object-Based API)
**File:** `example_02_spring_mass_object.py`
**Demonstrates:**
- Object-based API using Mass and Spring components
- Building systems with physical components
- Fixed anchor points
- Automatic Lagrangian construction

**Run:** `manim -pql example_02_spring_mass_object.py SpringMassObject`

---

### Example 3: Double Pendulum
**File:** `example_03_double_pendulum.py`
**Demonstrates:**
- Multi-DOF system (4 degrees of freedom)
- Chaotic dynamics
- Stiff ODE handling (using Radau method)
- Complex coupled system
- Custom visual configuration

**Run:** `manim -pql example_03_double_pendulum.py DoublePendulum`

**Note:** The rods are simulated as very stiff springs. For true rigid constraints,
a future version will implement Lagrange multipliers.

---

### Example 4: Coupled Oscillators
**File:** `example_04_coupled_oscillators.py`
**Demonstrates:**
- Multiple masses and springs
- Normal modes of oscillation
- Energy transfer between oscillators
- Horizontal 2D motion

**Run:** `manim -pql example_04_coupled_oscillators.py CoupledOscillators`

---

## General Usage Pattern

All examples follow this pattern:

```python
# 1. Import
from actionphysics import System
from actionphysics.visualization import PhysicsRenderer

# 2. Build system
system = System()
mass = system.add_mass(mass=1.0, position=[0, 0])
system.add_gravity(g=9.8)

# 3. Solve
system.solve(duration=5.0)

# 4. Create scene
class MyScene(PhysicsRenderer):
    def __init__(self):
        super().__init__(system, duration=5.0)

# 5. Render with Manim
# manim -pql my_file.py MyScene
```

## Render Quality Options

- `-ql` : Low quality (480p, 15fps) - fast preview
- `-qm` : Medium quality (720p, 30fps)
- `-qh` : High quality (1080p, 60fps)
- `-qk` : 4K quality (2160p, 60fps)

Add `-p` flag to preview immediately after rendering.

## Tips

1. **For fast iteration:** Start with `-ql` quality
2. **For stiff systems:** Use `method='Radau'` in `system.solve()`
3. **For energy conservation:** Increase `rtol` and `atol` tolerances
4. **To hide equations:** Use custom `VisualConfig(show_equations=False)`
5. **For horizontal motion:** Set initial positions with varying x-coordinates
