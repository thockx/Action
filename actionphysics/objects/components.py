"""
Physical component classes for object-based API.
"""

from typing import Optional, Tuple, List
import numpy as np
from sympy import symbols, Symbol


class Mass:
    """
    Point mass in 2D space.
    
    Args:
        mass: Mass value (kg)
        position: Initial position [x, y]
        velocity: Initial velocity [vx, vy]
        fixed: Whether this mass is fixed in space
        name: Optional name for the mass
    """
    
    def __init__(
        self,
        mass: float,
        position: Tuple[float, float] = (0.0, 0.0),
        velocity: Tuple[float, float] = (0.0, 0.0),
        fixed: bool = False,
        name: Optional[str] = None
    ):
        self.mass = mass
        self.initial_position = np.array(position, dtype=float)
        self.initial_velocity = np.array(velocity, dtype=float)
        self.fixed = fixed
        self.name = name
        
        # Assigned by System when added
        self.coord_indices = None  # Tuple of (x_index, y_index) in state vector
        self.x_coord = None  # Coordinate object for x
        self.y_coord = None  # Coordinate object for y
        
    def __repr__(self):
        return f"Mass(m={self.mass}, pos={self.initial_position}, fixed={self.fixed})"


class FixedPoint:
    """
    Fixed point in space (e.g., wall anchor for spring).
    
    Args:
        position: Position [x, y]
        name: Optional name for the point
    """
    
    def __init__(
        self,
        position: Tuple[float, float],
        name: Optional[str] = None
    ):
        self.position = np.array(position, dtype=float)
        self.name = name
        self.fixed = True  # Always fixed
        
    def __repr__(self):
        return f"FixedPoint(pos={self.position})"


class Spring:
    """
    Linear spring connecting two points (mass or fixed point).
    
    Args:
        k: Spring constant (N/m)
        rest_length: Natural length of spring (m). If None, uses initial distance
        connect: Tuple of (point1, point2) to connect
        name: Optional name for the spring
    """
    
    def __init__(
        self,
        k: float,
        rest_length: Optional[float] = None,
        connect: Optional[Tuple[Mass | FixedPoint, Mass | FixedPoint]] = None,
        name: Optional[str] = None
    ):
        self.k = k
        self.rest_length = rest_length
        self.connect = connect
        self.name = name
        
    def __repr__(self):
        return f"Spring(k={self.k}, L0={self.rest_length})"
    
    def compute_rest_length(self):
        """Compute rest length from initial positions if not specified."""
        if self.rest_length is None and self.connect is not None:
            p1, p2 = self.connect
            pos1 = p1.initial_position if isinstance(p1, Mass) else p1.position
            pos2 = p2.initial_position if isinstance(p2, Mass) else p2.position
            self.rest_length = float(np.linalg.norm(pos2 - pos1))


class Damper:
    """
    Linear damper (viscous friction) between two points.
    
    Args:
        c: Damping coefficient (N·s/m)
        connect: Tuple of (point1, point2) to connect
        name: Optional name for the damper
    """
    
    def __init__(
        self,
        c: float,
        connect: Optional[Tuple[Mass | FixedPoint, Mass | FixedPoint]] = None,
        name: Optional[str] = None
    ):
        self.c = c
        self.connect = connect
        self.name = name
        
    def __repr__(self):
        return f"Damper(c={self.c})"


class Gravity:
    """
    Uniform gravitational field.
    
    Args:
        g: Gravitational acceleration (m/s²), positive downward
        direction: Direction vector [dx, dy] (default: [0, -1] for downward)
    """
    
    def __init__(
        self,
        g: float = 9.8,
        direction: Tuple[float, float] = (0.0, -1.0)
    ):
        self.g = g
        self.direction = np.array(direction, dtype=float)
        # Normalize direction
        self.direction = self.direction / np.linalg.norm(self.direction)
        
    def __repr__(self):
        return f"Gravity(g={self.g}, dir={self.direction})"
