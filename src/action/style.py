"""Centralized visual appearance configuration for Action."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisualStyle:
    """User-editable visual settings for Action's Manim scene objects."""

    # Physical geometry. Values are in Action's physical units and pass
    # through System's physical-to-view transformation.
    mass_radius: float = 0.15  # Physical radius of each mass marker.
    mass_stroke_width: float = 0.2  # Physical outline stroke width of masses.
    rod_width: float = 0.15  # Physical stroke width of rods.
    spring_width: float = 0.015  # Physical stroke width of springs.
    spring_amplitude: float = 0.05  # Physical sideways size of spring coils.
    spring_lead_length: float = 0.25  # Straight physical lead-in before coils.
    attachment_radius: float = 0.02  # Physical radius of attachment dots.
    wall_width: float = 0.2  # Physical stroke width of wall supports.
    wall_support_offset: float = 0.2  # Physical offset from wall anchor to support.
    wall_hatch_length: float = 0.05  # Physical length of each wall hatch.
    wall_hatch_backset: float = 0.05  # Physical hatch offset along the wall.
    wall_hatch_width: float = 0.02  # Physical stroke width of wall hatches.
    hatch_density: float = 8.0  # Number of wall hatches per physical wall unit.
    angle_arc_radius: float = 0.25  # Physical radius of hinge angle arcs.
    angle_radial_extension: float = 0.05  # Physical extension beyond angle arcs.
    angle_label_offset: float = 0.08  # Physical offset of angle labels from arcs.
    angle_width: float = 0.075  # Physical stroke width of angle indicators.
    angle_radial_width: float = 0.1  # Physical stroke width of angle radial lines.

    # Basic appearance.
    mass_color: str = "#000000"  # Outline and label color for masses.
    mass_fill_color: str = "#FFFFFF"  # Fill color inside mass markers.
    rod_color: str = "#000000"  # Rod lines, dots, and length labels color.
    spring_color: str = "#000000"  # Spring lines and attachment dots color.
    wall_color: str = "#000000"  # Wall, hatch, and angle indicator color.

    # Readability settings.
    mass_label_scale: float = 0.3  # Manim scale for mass labels.
    rod_label_scale: float = 0.3  # Manim scale for rod length labels.
    angle_label_scale: float = 0.3  # Manim scale for hinge angle labels.
    vector_label_scale: float = 0.3  # Manim scale for vector labels.

    # Equation and overlay layout.
    equation_scale: float = 0.5  # Manim scale for displayed equations.
    equation_edge_buffer: float = 0.2  # Screen-space margin around equations.
    equation_line_buffer: float = 0.08  # Spacing between displayed equations.
    gravity_indicator_length: float = 0.6  # Screen-space length of gravity arrow.
    gravity_indicator_width: float = 0.5  # Physical stroke width of gravity arrow.
    gravity_indicator_tip_ratio: float = 0.3  # Arrowhead length-to-arrow length ratio.
    gravity_indicator_label_scale: float = 0.6  # Manim scale for the gravity label.
    gravity_indicator_offset: float = 0.1  # Screen-space gravity label offset.
    gravity_indicator_corner_buffer: float = 0.5  # Margin from the screen corner.

    # Vector overlay appearance.
    vector_stroke_width: float = 4.5  # Additional vector stroke width at peak magnitude.
    vector_stroke_min_width: float = 2.5  # Minimum vector stroke width.
    vector_tip_min_width: float = 0.08  # Minimum vector arrowhead width.
    vector_tip_extra_width: float = 0.28  # Additional arrowhead width at peak magnitude.
    vector_label_forward_offset: float = 0.1  # Base label offset beyond vector tips.
    vector_label_forward_extra: float = 0.12  # Extra forward offset at peak magnitude.
    vector_label_side_offset: float = 0.08  # Base label offset perpendicular to vectors.
    vector_label_side_extra: float = 0.08  # Extra side offset at peak magnitude.

    # Automatic framing. These values are view-space settings, not physical dimensions.
    auto_frame: bool = True  # Fit the solved motion into the view automatically.
    frame_size: float = 10  # Maximum view-space extent used by automatic framing.
    frame_padding: float = 0.0  # View-space margin kept around the fitted system.
    frame_min_scale: float | None = None  # Optional lower bound for view scale.
    frame_max_scale: float | None = None  # Optional upper bound for view scale.
