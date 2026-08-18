"""
Main physics renderer for Manim visualizations.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
from manim import *
from sympy import latex

from .config import VisualConfig

if TYPE_CHECKING:
    from ..mechanics.lagrangian_system import LagrangianSystem
    from ..objects.system import System


class PhysicsRenderer(Scene):
    """
    Manim Scene that automatically renders a physics simulation.
    
    Handles both LagrangianSystem (symbolic) and System (object-based) inputs.
    """
    
    def __init__(
        self,
        physics_system,  # LagrangianSystem or System
        duration: float = 5.0,
        config: Optional[VisualConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.physics_system = physics_system
        self.duration = duration
        self.config = config if config is not None else VisualConfig()
        
    def construct(self):
        """Main Manim construction method."""
        self.camera.background_color = self.config.background_color
        
        # Determine system type
        from ..objects.system import System
        from ..mechanics.lagrangian_system import LagrangianSystem
        
        if isinstance(self.physics_system, System):
            self._render_object_system()
        elif isinstance(self.physics_system, LagrangianSystem):
            self._render_lagrangian_system()
        else:
            raise TypeError(f"Unknown physics system type: {type(self.physics_system)}")
    
    def _render_lagrangian_system(self):
        """Render a symbolic LagrangianSystem (1D for now)."""
        system = self.physics_system
        
        if system.n_dof != 1:
            raise NotImplementedError(
                "Direct LagrangianSystem rendering currently supports 1D systems only. "
                "For multi-DOF, use the object-based System API."
            )
        
        # For 1D system, use the original rendering approach from lagrangian_sim.py
        self._render_1d_vertical_system(system)
    
    def _render_1d_vertical_system(self, system):
        """Render 1D vertical system (like spring-mass)."""
        # This is extracted from the original lagrangian_sim.py
        
        # Get coordinate
        coord = system.coordinates[0]
        
        # Create solution functions
        equilibrium_offset = 0.0  # Can be customized
        
        def position_func(t):
            return system.get_position(0, t) + equilibrium_offset
        
        def velocity_func(t):
            return system.get_velocity(0, t)
        
        # For acceleration and force, we need to evaluate the EOM
        # Store acceleration function (simplified approach)
        state_at = lambda t: system.get_state(t)
        
        # Auto-scaling vectors
        max_vector_length = self.config.get_max_vector_length()
        time_samples = np.linspace(0, self.duration, 100)
        
        velocities = [velocity_func(t) for t in time_samples]
        max_velocity = max(abs(v) for v in velocities)
        
        velocity_scale = max_vector_length / max_velocity if max_velocity > 0 else 1.0
        
        # Create visual elements
        wall_y = 3.0
        
        # Wall
        wall = Line(LEFT * 2, RIGHT * 2, color=self.config.mass_color)
        wall.shift(UP * wall_y)
        wall.set_stroke(width=5)
        
        # Hatching
        hatching = self._create_wall_hatching(wall_y)
        
        # Ball
        circle = Circle(
            radius=self.config.mass_radius,
            color=self.config.mass_color,
            fill_opacity=self.config.mass_fill_opacity,
            stroke_width=self.config.mass_stroke_width
        )
        circle.move_to([0, position_func(0), 0])
        
        # Spring
        spring = self._create_spring_1d(wall.get_bottom(), circle.get_top())
        
        # Mass label
        if self.config.show_mass_labels:
            mass_label = MathTex("m", color=self.config.mass_color).scale(0.8)
            mass_label.move_to(circle.get_center())
        
        # Velocity vector
        if self.config.show_velocity_vectors:
            v_arrow = Arrow(
                start=circle.get_center(),
                end=circle.get_center() + UP * velocity_func(0) * velocity_scale,
                buff=0,
                color=self.config.velocity_color,
                stroke_width=self.config.vector_stroke_width,
                max_tip_length_to_length_ratio=0.25
            )
            v_label = MathTex("v", color=self.config.velocity_color).scale(0.7)
            v_label.next_to(v_arrow.get_end(), RIGHT, buff=0.1)
        
        # Equations
        if self.config.show_equations:
            lag_tex = MathTex(
                r"\mathcal{L} = " + latex(system.L),
                color=self.config.equation_color
            ).scale(self.config.equation_scale)
            
            eom_tex = MathTex(
                latex(coord.q_ddot) + " = " + latex(system.equations_of_motion[0]),
                color=self.config.equation_color
            ).scale(self.config.equation_scale)
            
            equations = VGroup(lag_tex, eom_tex).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            equations.to_edge(DOWN, buff=0.3).to_edge(LEFT, buff=0.5)
            
            eq_bg = SurroundingRectangle(
                equations,
                color=WHITE,
                fill_opacity=self.config.equation_background_opacity,
                buff=0.2,
                stroke_width=1,
                stroke_color=GRAY
            )
            self.add(eq_bg, equations)
        
        # Add to scene
        self.add(wall, hatching, circle, spring)
        
        if self.config.show_mass_labels:
            self.add(mass_label)
        
        if self.config.show_velocity_vectors:
            self.add(v_arrow, v_label)
        
        # Updaters
        current_time = [0]
        
        def update_spring_func(mob):
            new_spring = self._create_spring_1d(wall.get_bottom(), circle.get_top())
            mob.become(new_spring)
        
        spring.add_updater(update_spring_func)
        
        if self.config.show_mass_labels:
            mass_label.add_updater(lambda m: m.move_to(circle.get_center()))
        
        if self.config.show_velocity_vectors:
            def update_v_arrow(mob):
                t = current_time[0]
                v = velocity_func(t)
                if abs(v * velocity_scale) > self.config.vector_min_visible_length:
                    new_arrow = Arrow(
                        start=circle.get_center(),
                        end=circle.get_center() + UP * v * velocity_scale,
                        buff=0,
                        color=self.config.velocity_color,
                        stroke_width=self.config.vector_stroke_width,
                        max_tip_length_to_length_ratio=0.25
                    )
                else:
                    new_arrow = Arrow(
                        start=circle.get_center(),
                        end=circle.get_center() + UP * self.config.vector_min_visible_length * np.sign(v),
                        buff=0,
                        color=self.config.velocity_color,
                        stroke_width=self.config.vector_stroke_width,
                        max_tip_length_to_length_ratio=0.25
                    )
                mob.become(new_arrow)
            
            v_arrow.add_updater(update_v_arrow)
            v_label.add_updater(lambda m: m.next_to(v_arrow.get_end(), RIGHT, buff=0.1))
        
        # Animate
        def update_ball(mob, alpha):
            t = alpha * self.duration
            current_time[0] = t
            y_pos = position_func(t)
            mob.move_to([0, y_pos, 0])
        
        self.play(
            UpdateFromAlphaFunc(circle, update_ball),
            run_time=self.duration,
            rate_func=linear
        )
    
    def _render_object_system(self):
        """Render an object-based System."""
        from ..objects.system import System
        
        system: System = self.physics_system
        
        # Get Lagrangian system
        lag_system = system.lagrangian_system
        
        if lag_system is None or lag_system.solution is None:
            raise RuntimeError("System must be solved before rendering")
        
        # Create mobjects for masses
        mass_circles = {}
        mass_labels = {}
        
        for mass in system.masses:
            if not mass.fixed:
                circle = Circle(
                    radius=self.config.mass_radius,
                    color=self.config.mass_color,
                    fill_opacity=self.config.mass_fill_opacity,
                    stroke_width=self.config.mass_stroke_width
                )
                
                # Initial position
                x0, y0 = mass.initial_position
                circle.move_to([x0, y0, 0])
                
                mass_circles[id(mass)] = circle
                self.add(circle)
                
                if self.config.show_mass_labels:
                    label = MathTex("m", color=self.config.mass_color).scale(0.8)
                    label.move_to(circle.get_center())
                    mass_labels[id(mass)] = label
                    self.add(label)
        
        # Create springs
        spring_mobjects = {}
        
        for spring in system.springs:
            p1, p2 = spring.connect
            
            # Get initial positions
            if isinstance(p1, system.masses.__class__) and id(p1) in mass_circles:
                start = mass_circles[id(p1)].get_center()
            else:
                pos = p1.initial_position if hasattr(p1, 'initial_position') else p1.position
                start = np.array([pos[0], pos[1], 0])
            
            if isinstance(p2, system.masses.__class__) and id(p2) in mass_circles:
                end = mass_circles[id(p2)].get_center()
            else:
                pos = p2.initial_position if hasattr(p2, 'initial_position') else p2.position
                end = np.array([pos[0], pos[1], 0])
            
            spring_mob = self._create_spring_2d(start, end)
            spring_mobjects[id(spring)] = (spring, spring_mob)
            self.add(spring_mob)
        
        # Equations
        if self.config.show_equations:
            lag_tex = MathTex(
                r"\mathcal{L} = " + latex(lag_system.L),
                color=self.config.equation_color
            ).scale(self.config.equation_scale)
            
            lag_tex.to_edge(DOWN, buff=0.3).to_edge(LEFT, buff=0.5)
            
            eq_bg = SurroundingRectangle(
                lag_tex,
                color=WHITE,
                fill_opacity=self.config.equation_background_opacity,
                buff=0.2,
                stroke_width=1,
                stroke_color=GRAY
            )
            self.add(eq_bg, lag_tex)
        
        # Updaters
        current_time = [0]
        
        # Update mass positions
        for mass in system.masses:
            if not mass.fixed:
                circle = mass_circles[id(mass)]
                
                def make_updater(m):
                    def updater(mob):
                        t = current_time[0]
                        x_idx, y_idx = m.coord_indices
                        x = lag_system.get_position(x_idx, t)
                        y = lag_system.get_position(y_idx, t)
                        mob.move_to([x, y, 0])
                    return updater
                
                circle.add_updater(make_updater(mass))
                
                if self.config.show_mass_labels and id(mass) in mass_labels:
                    mass_labels[id(mass)].add_updater(lambda m: m.move_to(circle.get_center()))
        
        # Update springs
        for spring_id, (spring, spring_mob) in spring_mobjects.items():
            p1, p2 = spring.connect
            
            def make_spring_updater(sp, pt1, pt2):
                def updater(mob):
                    # Get current positions
                    if isinstance(pt1, system.masses.__class__) and id(pt1) in mass_circles:
                        start = mass_circles[id(pt1)].get_center()
                    else:
                        pos = pt1.initial_position if hasattr(pt1, 'initial_position') else pt1.position
                        start = np.array([pos[0], pos[1], 0])
                    
                    if isinstance(pt2, system.masses.__class__) and id(pt2) in mass_circles:
                        end = mass_circles[id(pt2)].get_center()
                    else:
                        pos = pt2.initial_position if hasattr(pt2, 'initial_position') else pt2.position
                        end = np.array([pos[0], pos[1], 0])
                    
                    new_spring = self._create_spring_2d(start, end)
                    mob.become(new_spring)
                return updater
            
            spring_mob.add_updater(make_spring_updater(spring, p1, p2))
        
        # Animate time
        def advance_time(mob, alpha):
            current_time[0] = alpha * self.duration
        
        # Create invisible tracker for time
        tracker = ValueTracker(0)
        
        self.play(
            tracker.animate.set_value(self.duration),
            run_time=self.duration,
            rate_func=linear
        )
    
    def _create_spring_1d(self, start_point, end_point):
        """Create vertical spring visualization."""
        return self._create_spring_2d(start_point, end_point)
    
    def _create_spring_2d(self, start_point, end_point):
        """Create spring visualization between two points."""
        spring_points = []
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        
        length = np.linalg.norm(end_point - start_point)
        direction = (end_point - start_point) / length if length > 0 else np.array([0, 1, 0])
        
        # Perpendicular direction
        perpendicular = np.array([-direction[1], direction[0], 0])
        
        # Connectors at ends
        connector_length = 0.2
        start_connector_end = start_point + direction * connector_length
        spring_points.append(start_point)
        spring_points.append(start_connector_end)
        
        # Spring coils
        spring_start = start_connector_end
        spring_end = end_point - direction * connector_length
        
        num_coils = self.config.spring_coils
        for i in range(1, num_coils):
            t = i / num_coils
            point = spring_start + t * (spring_end - spring_start)
            offset = perpendicular * (-1)**i * self.config.spring_width
            spring_points.append(point + offset)
        
        spring_points.append(spring_end)
        spring_points.append(end_point)
        
        spring = VMobject(color=self.config.spring_color)
        spring.set_points_as_corners(spring_points)
        spring.set_stroke(width=self.config.spring_stroke_width)
        return spring
    
    def _create_wall_hatching(self, wall_y: float):
        """Create hatching pattern for wall."""
        hatching = VGroup()
        num_lines = 15
        spacing = 4.0 / num_lines
        hatch_length = 0.3
        
        for i in range(num_lines + 1):
            x_pos = -2 + i * spacing
            start = [x_pos, wall_y, 0]
            end = [
                x_pos - hatch_length * np.cos(np.pi/4),
                wall_y + hatch_length * np.sin(np.pi/4),
                0
            ]
            hatch_line = Line(start, end, color=self.config.mass_color)
            hatch_line.set_stroke(width=2)
            hatching.add(hatch_line)
        
        return hatching


def render_simulation(
    physics_system,
    duration: float = 5.0,
    config: Optional[VisualConfig] = None,
    quality: str = 'low',
    preview: bool = True,
    output_file: Optional[str] = None
):
    """
    Convenience function to render a physics simulation.
    
    Args:
        physics_system: LagrangianSystem or System object
        duration: Simulation duration in seconds
        config: VisualConfig for customization
        quality: Render quality ('low', 'medium', 'high')
        preview: Whether to preview the result
        output_file: Optional output file path
        
    Example:
        >>> system = System()
        >>> m = system.add_mass(1.0)
        >>> system.add_gravity()
        >>> render_simulation(system, duration=3.0)
    """
    # This would typically be called from command line
    # For programmatic use, users should create PhysicsRenderer scene
    print(f"To render this simulation, create a scene:")
    print(f"  from actionphysics.visualization import PhysicsRenderer")
    print(f"  class MyScene(PhysicsRenderer):")
    print(f"      def __init__(self):")
    print(f"          super().__init__(your_system, duration={duration})")
    print(f"Then run: manim -pql your_file.py MyScene")
