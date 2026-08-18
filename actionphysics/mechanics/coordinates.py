"""
Coordinate system abstractions for generalized coordinates.
"""

from typing import Optional, Tuple
from sympy import symbols, Symbol
from sympy.physics.mechanics import dynamicsymbols


class Coordinate:
    """
    Represents a generalized coordinate in a mechanical system.
    
    Supports both Cartesian coordinates (x, y, z) and custom generalized
    coordinates (θ, r, etc.) with automatic time derivative handling.
    
    Args:
        name: Name of the coordinate (e.g., 'x', 'theta', 'q1')
        initial_value: Initial position value
        initial_velocity: Initial velocity value
        is_angle: Whether this coordinate represents an angle (for display purposes)
    """
    
    def __init__(
        self,
        name: str,
        initial_value: float = 0.0,
        initial_velocity: float = 0.0,
        is_angle: bool = False
    ):
        self.name = name
        self.initial_value = initial_value
        self.initial_velocity = initial_velocity
        self.is_angle = is_angle
        
        # Create symbolic variable (time-dependent)
        self.symbol = dynamicsymbols(name)
        self.velocity_symbol = dynamicsymbols(name, 1)  # First derivative
        self.acceleration_symbol = dynamicsymbols(name, 2)  # Second derivative
        
    def __repr__(self):
        return f"Coordinate('{self.name}', initial={self.initial_value})"
    
    @property
    def q(self):
        """Generalized coordinate symbol."""
        return self.symbol
    
    @property
    def q_dot(self):
        """Generalized velocity symbol (time derivative)."""
        return self.velocity_symbol
    
    @property
    def q_ddot(self):
        """Generalized acceleration symbol (second time derivative)."""
        return self.acceleration_symbol


def create_coordinates(names: list[str], **kwargs) -> list[Coordinate]:
    """
    Convenience function to create multiple coordinates at once.
    
    Args:
        names: List of coordinate names
        **kwargs: Additional arguments passed to each Coordinate
        
    Returns:
        List of Coordinate objects
        
    Example:
        >>> coords = create_coordinates(['x', 'y', 'theta'])
    """
    return [Coordinate(name, **kwargs) for name in names]
