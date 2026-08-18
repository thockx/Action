"""
Core Lagrangian mechanics engine.

Automatically derives equations of motion from kinetic and potential energy,
solves ODEs numerically, and provides solution functions for animation.
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from sympy import symbols, diff, simplify, solve, lambdify, Symbol, Expr
from sympy.physics.mechanics import dynamicsymbols
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .coordinates import Coordinate


class LagrangianSystem:
    """
    Automatic Lagrangian mechanics system.
    
    Define kinetic (T) and potential (V) energy symbolically, and the system will:
    1. Derive equations of motion using Euler-Lagrange equations
    2. Solve the ODEs numerically
    3. Provide solution functions for position, velocity, acceleration, and force
    
    Args:
        T: Kinetic energy expression (SymPy expression)
        V: Potential energy expression (SymPy expression)
        coords: List of Coordinate objects or coordinate symbols
        params: Dictionary mapping parameter symbols to numerical values
        t_symbol: Time symbol (default: creates new symbol 't')
        
    Example:
        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> 
        >>> t = symbols('t')
        >>> y = dynamicsymbols('y')
        >>> y_dot = diff(y, t)
        >>> m, k = symbols('m k', positive=True)
        >>> 
        >>> T = m * y_dot**2 / 2
        >>> V = k * y**2 / 2
        >>> 
        >>> system = LagrangianSystem(
        ...     T=T, V=V,
        ...     coords=[y],
        ...     params={m: 1.0, k: 10.0}
        ... )
        >>> system.solve(duration=5.0, initial_conditions=[0.5, 0.0])
    """
    
    def __init__(
        self,
        T: Expr,
        V: Expr,
        coords: List[Coordinate | Symbol],
        params: Dict[Symbol, float],
        t_symbol: Optional[Symbol] = None
    ):
        self.T = T
        self.V = V
        self.params = params
        self.t = t_symbol if t_symbol is not None else symbols('t')
        
        # Handle coordinate input (support both Coordinate objects and raw symbols)
        self.coordinates = []
        self.coord_symbols = []
        
        for coord in coords:
            if isinstance(coord, Coordinate):
                self.coordinates.append(coord)
                self.coord_symbols.append(coord.q)
            else:
                # Raw symbol provided - create Coordinate wrapper
                coord_obj = Coordinate(str(coord), initial_value=0.0)
                self.coordinates.append(coord_obj)
                self.coord_symbols.append(coord)
        
        self.n_dof = len(self.coord_symbols)
        
        # Storage for derived quantities
        self.L = None  # Lagrangian
        self.equations_of_motion = []  # List of symbolic EOMs (one per coordinate)
        self.solution = None  # SciPy solution object
        self.solution_functions = {}  # Interpolated solution functions
        
        # Derived automatically
        self._derive_equations()
        
    def _derive_equations(self):
        """
        Derive equations of motion using Euler-Lagrange equations.
        
        For each generalized coordinate q_i:
            d/dt(∂L/∂q̇_i) - ∂L/∂q_i = 0
        
        Solves for q̈_i in terms of q_i and q̇_i.
        """
        print(f"Deriving equations of motion for {self.n_dof} DOF system...")
        
        # Construct Lagrangian
        self.L = self.T - self.V
        self.L = simplify(self.L)
        
        print(f"Lagrangian: L = {self.L}")
        
        # Derive equations for each coordinate
        self.equations_of_motion = []
        
        for i, q in enumerate(self.coord_symbols):
            q_dot = diff(q, self.t)
            q_ddot = diff(q_dot, self.t)
            
            # Euler-Lagrange equation components
            dL_dq = diff(self.L, q)  # ∂L/∂q
            dL_dq_dot = diff(self.L, q_dot)  # ∂L/∂q̇
            dt_dL_dq_dot = diff(dL_dq_dot, self.t)  # d/dt(∂L/∂q̇)
            
            # Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0
            EL_eq = dt_dL_dq_dot - dL_dq
            
            # Solve for acceleration (q̈)
            eom_solutions = solve(EL_eq, q_ddot)
            
            if not eom_solutions:
                raise ValueError(f"Could not solve Euler-Lagrange equation for {q}")
            
            eom = simplify(eom_solutions[0])
            self.equations_of_motion.append(eom)
            
            coord_name = self.coordinates[i].name if i < len(self.coordinates) else str(q)
            print(f"EOM for {coord_name}: {coord_name}̈ = {eom}")
        
    def solve(
        self,
        duration: float,
        initial_conditions: List[float],
        n_points: int = 1000,
        method: str = 'RK45',
        rtol: float = 1e-8,
        atol: float = 1e-10
    ) -> bool:
        """
        Solve the system numerically using scipy's ODE solver.
        
        Args:
            duration: Simulation duration in seconds
            initial_conditions: Initial state [q1_0, q̇1_0, q2_0, q̇2_0, ...]
                               (position and velocity for each coordinate)
            n_points: Number of time points for solution
            method: Integration method ('RK45', 'DOP853', etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If initial_conditions length doesn't match 2 * n_dof
            RuntimeError: If ODE solver fails
        """
        if len(initial_conditions) != 2 * self.n_dof:
            raise ValueError(
                f"Expected {2 * self.n_dof} initial conditions "
                f"(position and velocity for each DOF), got {len(initial_conditions)}"
            )
        
        print("Solving equations numerically...")
        
        # Create lambdified acceleration functions
        # For each EOM, substitute parameters and create numerical function
        accel_funcs = []
        
        for eom in self.equations_of_motion:
            # Create placeholder symbols for numerical evaluation
            state_symbols = []
            for q in self.coord_symbols:
                state_symbols.append(symbols(f'{q}_val'))
                state_symbols.append(symbols(f'{q}_dot_val'))
            
            # Substitute coordinate symbols with numerical placeholders
            eom_numerical = eom
            for i, q in enumerate(self.coord_symbols):
                q_dot = diff(q, self.t)
                eom_numerical = eom_numerical.subs(q, state_symbols[2*i])
                eom_numerical = eom_numerical.subs(q_dot, state_symbols[2*i + 1])
            
            # Substitute parameter values
            for param_symbol, param_value in self.params.items():
                eom_numerical = eom_numerical.subs(param_symbol, param_value)
            
            # Create numerical function
            accel_func = lambdify(state_symbols, eom_numerical, 'numpy')
            accel_funcs.append(accel_func)
        
        # Define ODE system for scipy
        def ode_system(t, state):
            """
            ODE system in state-space form.
            State: [q1, q̇1, q2, q̇2, ..., qn, q̇n]
            Returns: [q̇1, q̈1, q̇2, q̈2, ..., q̇n, q̈n]
            """
            derivatives = []
            
            for i in range(self.n_dof):
                q_val = state[2*i]
                q_dot_val = state[2*i + 1]
                
                # Position derivative is velocity
                derivatives.append(q_dot_val)
                
                # Velocity derivative is acceleration (from EOM)
                q_ddot_val = float(accel_funcs[i](*state))
                derivatives.append(q_ddot_val)
            
            return derivatives
        
        # Time span and evaluation points
        t_span = (0, duration)
        t_eval = np.linspace(0, duration, n_points)
        
        # Solve ODE
        self.solution = solve_ivp(
            ode_system,
            t_span,
            initial_conditions,
            t_eval=t_eval,
            method=method,
            dense_output=True,
            rtol=rtol,
            atol=atol
        )
        
        if not self.solution.success:
            raise RuntimeError(f"ODE solver failed: {self.solution.message}")
        
        print("Solution obtained successfully!")
        
        # Create interpolation functions for smooth access
        self._create_solution_functions()
        
        return True
    
    def _create_solution_functions(self):
        """Create interpolated functions for easy access to solution."""
        self.solution_functions = {}
        
        for i in range(self.n_dof):
            coord_name = self.coordinates[i].name
            
            # Position
            position_interp = interp1d(
                self.solution.t,
                self.solution.y[2*i],
                kind='cubic',
                fill_value='extrapolate'
            )
            self.solution_functions[f'{coord_name}_position'] = position_interp
            
            # Velocity
            velocity_interp = interp1d(
                self.solution.t,
                self.solution.y[2*i + 1],
                kind='cubic',
                fill_value='extrapolate'
            )
            self.solution_functions[f'{coord_name}_velocity'] = velocity_interp
    
    def get_position(self, coord_index: int, t: float) -> float:
        """Get position of coordinate at time t."""
        coord_name = self.coordinates[coord_index].name
        return float(self.solution_functions[f'{coord_name}_position'](t))
    
    def get_velocity(self, coord_index: int, t: float) -> float:
        """Get velocity of coordinate at time t."""
        coord_name = self.coordinates[coord_index].name
        return float(self.solution_functions[f'{coord_name}_velocity'](t))
    
    def get_state(self, t: float) -> np.ndarray:
        """Get full state vector [q1, q̇1, q2, q̇2, ...] at time t."""
        state = []
        for i in range(self.n_dof):
            state.append(self.get_position(i, t))
            state.append(self.get_velocity(i, t))
        return np.array(state)
