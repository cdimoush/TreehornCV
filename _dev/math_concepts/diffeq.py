from manim import *

class DifferentialEquationField(Scene):
    def construct(self):
        # Define the vector field
        def vector_field(pos):
            x, y, _ = pos  # Ignore the z value
            return np.array([y - x, y, 0])  # Return a 3D vector (z=0)

        # Create StreamLines object to visualize the field
        stream_lines = StreamLines(
            vector_field, 
            x_range=[-3, 3], 
            y_range=[-3, 3], 
            stroke_width=1
        )
        self.add(stream_lines)
        self.wait()
