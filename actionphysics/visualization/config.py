"""
Visual configuration for rendering.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from manim import *


@dataclass
class VisualConfig:
    """
    Configuration for visual rendering of physics simulations.
    
    Philosophy: Automatic defaults (like LaTeX), but customizable.
    """
    
    # Background
    background_color: str = WHITE
    
    # Vector visualization
    show_position_vectors: bool = False  # Usually cluttered in 2D
    show_velocity_vectors: bool = True
    show_acceleration_vectors: bool = True
    show_force_vectors: bool = True
    
    # Vector styling
    velocity_color: str = BLUE
    acceleration_color: str = GREEN
    force_color: str = RED
    position_color: str = BLACK
    
    vector_stroke_width: float = 4.0
    vector_max_length: float = 0.25  # Fraction of frame height (8 units)
    vector_min_visible_length: float = 0.05  # Minimum length to render
    
    # Mass visualization
    mass_radius: float = 0.3
    mass_color: str = BLACK
    mass_fill_opacity: float = 0.0
    mass_stroke_width: float = 3.0
    show_mass_labels: bool = True
    
    # Spring visualization
    spring_color: str = BLACK
    spring_stroke_width: float = 3.0
    spring_coils: int = 12
    spring_width: float = 0.15
    
    # Equations
    show_equations: bool = True
    equation_scale: float = 0.6
    equation_color: str = BLACK
    equation_background_opacity: float = 0.8
    
    # Animation
    frame_rate: int = 60
    quality: str = 'low'  # 'low', 'medium', 'high'
    
    # Layout
    frame_height: float = 8.0  # Manim default
    camera_center: Optional[Tuple[float, float]] = None
    
    def get_max_vector_length(self) -> float:
        """Get maximum vector length in scene units."""
        return self.vector_max_length * self.frame_height
