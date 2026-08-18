"""
System builder class for object-based API.
"""

from typing import List, Tuple, Optional
import numpy as np
from sympy import symbols, sqrt, Symbol
from sympy.physics.mechanics import dynamicsymbols

from ..mechanics.lagrangian_system import LagrangianSystem
from ..mechanics.coordinates import Coordinate
from .components import Mass, Spring, Damper, Gravity, FixedPoint


class System:
    """
    High-level system builder using object-based API.
    
    Automatically constructs Lagrangian from physical components and derives
    equations of motion.
    
    Example:
        >>> system = System()
        >>> m1 = system.add_mass(1.0, position=[0, 0])
        >>> m2 = system.add_mass(0.5, position=[1, 0])
        >>> system.add_spring(k=10, connect=[m1, m2])
        >>> system.add_gravity(g=9.8)
        >>> system.solve(duration=5.0)
    """
    
    def __init__(self):
        self.masses: List[Mass] = []
        self.springs: List[Spring] = []
        self.dampers: List[Damper] = []
        self.gravity: Optional[Gravity] = None
        self.fixed_points: List[FixedPoint] = []
        
        self.coordinates: List[Coordinate] = []
        self.lagrangian_system: Optional[LagrangianSystem] = None
        
        self._built = False
        
    def add_mass(
        self,
        mass: float,
        position: Tuple[float, float] = (0.0, 0.0),
        velocity: Tuple[float, float] = (0.0, 0.0),
        fixed: bool = False,
        name: Optional[str] = None
    ) -> Mass:
        """
        Add a point mass to the system.
        
        Args:
            mass: Mass value (kg)
            position: Initial position [x, y]
            velocity: Initial velocity [vx, vy]
            fixed: Whether this mass is fixed in space
            name: Optional name for the mass
            
        Returns:
            The created Mass object
        """
        if self._built:
            raise RuntimeError("Cannot add components after system is built")
        
        mass_obj = Mass(mass, position, velocity, fixed, name)
        self.masses.append(mass_obj)
        return mass_obj
    
    def add_spring(
        self,
        k: float,
        connect: Tuple[Mass | FixedPoint, Mass | FixedPoint],
        rest_length: Optional[float] = None,
        name: Optional[str] = None
    ) -> Spring:
        """
        Add a linear spring between two points.
        
        Args:
            k: Spring constant (N/m)
            connect: Tuple of (point1, point2) to connect
            rest_length: Natural length of spring (m). If None, uses initial distance
            name: Optional name for the spring
            
        Returns:
            The created Spring object
        """
        if self._built:
            raise RuntimeError("Cannot add components after system is built")
        
        spring = Spring(k, rest_length, connect, name)
        self.springs.append(spring)
        return spring
    
    def add_damper(
        self,
        c: float,
        connect: Tuple[Mass | FixedPoint, Mass | FixedPoint],
        name: Optional[str] = None
    ) -> Damper:
        """
        Add a linear damper between two points.
        
        Note: Damping requires Rayleigh dissipation function (not yet implemented).
        
        Args:
            c: Damping coefficient (N·s/m)
            connect: Tuple of (point1, point2) to connect
            name: Optional name for the damper
            
        Returns:
            The created Damper object
        """
        if self._built:
            raise RuntimeError("Cannot add components after system is built")
        
        damper = Damper(c, connect, name)
        self.dampers.append(damper)
        return damper
    
    def add_gravity(
        self,
        g: float = 9.8,
        direction: Tuple[float, float] = (0.0, -1.0)
    ) -> Gravity:
        """
        Add uniform gravitational field.
        
        Args:
            g: Gravitational acceleration (m/s²), positive downward
            direction: Direction vector [dx, dy] (default: [0, -1] for downward)
            
        Returns:
            The created Gravity object
        """
        if self._built:
            raise RuntimeError("Cannot add components after system is built")
        
        self.gravity = Gravity(g, direction)
        return self.gravity
    
    def add_fixed_point(
        self,
        position: Tuple[float, float],
        name: Optional[str] = None
    ) -> FixedPoint:
        """
        Add a fixed point in space.
        
        Args:
            position: Position [x, y]
            name: Optional name for the point
            
        Returns:
            The created FixedPoint object
        """
        if self._built:
            raise RuntimeError("Cannot add components after system is built")
        
        fixed_point = FixedPoint(position, name)
        self.fixed_points.append(fixed_point)
        return fixed_point
    
    def build(self):
        """
        Build the Lagrangian system from components.
        
        Automatically:
        1. Creates generalized coordinates for each free mass
        2. Constructs kinetic energy T from masses
        3. Constructs potential energy V from springs and gravity
        4. Derives equations of motion
        """
        if self._built:
            return
        
        print(f"Building system with {len(self.masses)} masses, "
              f"{len(self.springs)} springs...")
        
        # Create coordinates for free (non-fixed) masses
        self.coordinates = []
        coord_index = 0
        
        for i, mass in enumerate(self.masses):
            if not mass.fixed:
                # Create x and y coordinates
                name_suffix = f"_{mass.name}" if mass.name else f"_{i}"
                
                x_coord = Coordinate(
                    f"x{name_suffix}",
                    initial_value=mass.initial_position[0],
                    initial_velocity=mass.initial_velocity[0]
                )
                y_coord = Coordinate(
                    f"y{name_suffix}",
                    initial_value=mass.initial_position[1],
                    initial_velocity=mass.initial_velocity[1]
                )
                
                mass.x_coord = x_coord
                mass.y_coord = y_coord
                mass.coord_indices = (coord_index, coord_index + 1)
                
                self.coordinates.extend([x_coord, y_coord])
                coord_index += 2
        
        if not self.coordinates:
            raise ValueError("No free masses in system - nothing to simulate")
        
        # Build symbolic expressions
        T_expr = self._build_kinetic_energy()
        V_expr = self._build_potential_energy()
        
        # Extract parameter symbols and values
        params = self._extract_parameters()
        
        # Create LagrangianSystem
        coord_symbols = [coord.q for coord in self.coordinates]
        
        self.lagrangian_system = LagrangianSystem(
            T=T_expr,
            V=V_expr,
            coords=self.coordinates,
            params=params
        )
        
        self._built = True
        print("System built successfully!")
    
    def _build_kinetic_energy(self):
        """Construct kinetic energy T = Σ (1/2) * m_i * v_i²"""
        T = 0
        
        for mass in self.masses:
            if not mass.fixed:
                # Velocity magnitude squared: vx² + vy²
                vx = mass.x_coord.q_dot
                vy = mass.y_coord.q_dot
                v_squared = vx**2 + vy**2
                
                # Kinetic energy contribution
                m_sym = symbols(f'm_{id(mass)}', positive=True, real=True)
                mass._symbol = m_sym  # Store for parameter extraction
                
                T += m_sym * v_squared / 2
        
        return T
    
    def _build_potential_energy(self):
        """Construct potential energy V from springs and gravity."""
        V = 0
        
        # Spring potential energy: (1/2) * k * (|r₂ - r₁| - L₀)²
        for spring in self.springs:
            spring.compute_rest_length()  # Ensure rest length is set
            
            p1, p2 = spring.connect
            
            # Get positions
            if isinstance(p1, Mass) and not p1.fixed:
                x1, y1 = p1.x_coord.q, p1.y_coord.q
            else:
                pos1 = p1.initial_position if isinstance(p1, Mass) else p1.position
                x1, y1 = float(pos1[0]), float(pos1[1])
            
            if isinstance(p2, Mass) and not p2.fixed:
                x2, y2 = p2.x_coord.q, p2.y_coord.q
            else:
                pos2 = p2.initial_position if isinstance(p2, Mass) else p2.position
                x2, y2 = float(pos2[0]), float(pos2[1])
            
            # Spring extension
            dx = x2 - x1
            dy = y2 - y1
            length = sqrt(dx**2 + dy**2)
            extension = length - spring.rest_length
            
            # Potential energy
            k_sym = symbols(f'k_{id(spring)}', positive=True, real=True)
            spring._symbol = k_sym
            
            V += k_sym * extension**2 / 2
        
        # Gravitational potential energy: m * g * h (dot product with direction)
        if self.gravity is not None:
            g_sym = symbols('g', positive=True, real=True)
            self.gravity._symbol = g_sym
            
            gx, gy = self.gravity.direction
            
            for mass in self.masses:
                if not mass.fixed:
                    # Height in gravity direction
                    x, y = mass.x_coord.q, mass.y_coord.q
                    height = gx * x + gy * y
                    
                    # Potential energy (negative because positive g is magnitude)
                    V += -mass._symbol * g_sym * height
        
        return V
    
    def _extract_parameters(self):
        """Extract parameter symbols and their numerical values."""
        params = {}
        
        # Mass parameters
        for mass in self.masses:
            if not mass.fixed and hasattr(mass, '_symbol'):
                params[mass._symbol] = mass.mass
        
        # Spring parameters
        for spring in self.springs:
            if hasattr(spring, '_symbol'):
                params[spring._symbol] = spring.k
        
        # Gravity parameter
        if self.gravity is not None and hasattr(self.gravity, '_symbol'):
            params[self.gravity._symbol] = self.gravity.g
        
        return params
    
    def solve(
        self,
        duration: float,
        n_points: int = 1000,
        method: str = 'RK45',
        rtol: float = 1e-8,
        atol: float = 1e-10
    ):
        """
        Solve the system dynamics.
        
        Args:
            duration: Simulation duration in seconds
            n_points: Number of time points for solution
            method: Integration method
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        if not self._built:
            self.build()
        
        # Gather initial conditions [x1, vx1, y1, vy1, x2, vx2, y2, vy2, ...]
        initial_conditions = []
        for coord in self.coordinates:
            initial_conditions.append(coord.initial_value)
            initial_conditions.append(coord.initial_velocity)
        
        self.lagrangian_system.solve(
            duration=duration,
            initial_conditions=initial_conditions,
            n_points=n_points,
            method=method,
            rtol=rtol,
            atol=atol
        )
    
    def render(self, duration: float = 5.0, **kwargs):
        """
        Render the simulation using Manim.
        
        Args:
            duration: Simulation duration in seconds
            **kwargs: Additional arguments for rendering
        """
        if not self._built:
            self.build()
        
        if self.lagrangian_system.solution is None:
            self.solve(duration=duration)
        
        from ..visualization.renderer import render_simulation
        render_simulation(self, duration=duration, **kwargs)
