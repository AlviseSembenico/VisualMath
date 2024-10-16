# Plot sample paths
import numpy as np
from manim import *

# set the random seed
np.random.seed(0)


class MovingNormalDistribution(ThreeDScene):
    def construct(self):
        # Set up axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 1, 0.2],
            x_length=6,
            y_length=6,
            z_length=3,
        )

        # Create the normal distribution function
        def normal_distribution(x, y):
            return np.exp(-0.5 * (x**2 + y**2)) / (2 * np.pi)

        # Define a Surface representing the 2D normal distribution
        surface = Surface(
            lambda u, v: axes.c2p(u, v, normal_distribution(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )

        # Add the axes and surface to the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.add(axes, surface)

        # Animation to move the normal distribution along the z-axis
        self.play(surface.animate.shift(UP * 2))

        # Rotate the camera around to show the 3D effect
        self.play(self.camera.animate.set_phi(60 * DEGREES).set_theta(-90 * DEGREES))
        self.wait()

        # Bring the surface back to its original position
        self.play(surface.animate.shift(DOWN * 2))
        self.wait()


class StochasticProcess(Scene):
    def construct(self):
        # Title
        title = Text("What is a stochastic process?")
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Definition
        definition = Tex(
            r"""
            A \textbf{stochastic process} is a collection of random variables \\
            $\{ X_t : t \in T \}$ defined on a common probability space, \\
            where $T$ represents the index set (often time).
        """,
            font_size=36,
        )
        definition.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(definition, shift=DOWN))
        self.wait(4)
        # Clear the screen
        self.play(FadeOut(title), FadeOut(definition))

        # Number line representing time

        time_line = NumberLine(
            x_range=[0, 10, 1],
            length=10,
            include_numbers=True,
            label_direction=DOWN,
            numbers_to_include=[0, 2, 4, 6, 8, 10],
        )
        time_label = Text("Time (t)", font_size=24)
        time_label.next_to(time_line, DOWN)
        setting = Tex(
            r"""Fixing an $\omega \in \Omega$ \\
                        we can trace a path, o realization of the process
                      """,
            font_size=24,
        )
        setting.next_to(time_line, UP)
        self.play(Create(time_line), Write(time_label), Write(setting)),
        self.wait(2)

        # Transform number line into axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": False},
        )
        axes.shift(DOWN * 0.5)
        labels = axes.get_axis_labels(x_label="t", y_label="X_t")
        # move setting to the top
        self.play(
            ReplacementTransform(time_line, axes),
            FadeOut(time_label),
            Write(labels),
            setting.animate.to_edge(UP),
        )
        self.wait(2)

        objects = []
        colors = [BLUE, GREEN, ORANGE]
        for i in range(3):
            # Generate a random walk
            steps = np.random.normal(0, 0.2, 100)
            samples = np.cumsum(steps)
            x_values = np.linspace(0, 10, 100)
            graph = axes.plot_line_graph(
                x_values,
                samples,
                line_color=colors[i % len(colors)],
                add_vertex_dots=False,
            )
            # add X_\omega(t) to the graph
            graph_label = MathTex(
                r"X_{\omega" + str(i) + r"}(t)",
                color=colors[i % len(colors)],
                font_size=20,
            )
            graph_label.next_to(graph, RIGHT, buff=0.1)

            self.play(Create(graph), Write(graph_label), run_time=2)
            self.wait(1)
            objects.append(graph)
            objects.append(graph_label)

        # Clear the screen
        self.play(
            FadeOut(axes), FadeOut(labels), FadeOut(setting), *map(FadeOut, objects)
        )
        self.wait(1)
        filtration = Tex(
            r"""
            A \textbf{filtration} is a collection of sigma-algebras \\
            $\{ \mathcal{F}_t : t \in T \}$, where $\mathcal{F}_t$ represents \\
            the information available up to time $t$.
        """,
            font_size=36,
        )
        self.play(Write(filtration))
        self.wait(4)
        self.play(FadeOut(filtration))
        adapted = Tex(
            r"""
            Intuitively,  process $\{ X_t : t \in T \}$ if $\mathcal{F}_t$-measurable \\
                if the value of $X_t$ can be determined from the information \\
                available up to time $t$. Namely $\mathcal{E}[X_t | \mathcal{F}_t]= X_t$.
        """,
            font_size=36,
        )
        self.play(Write(adapted))
        self.wait(4)

        # Clear the screen
        self.play(FadeOut(adapted))
        self.wait(1)

        title = Text("Understanding Adapted Stochastic Processes", font_size=48)
        self.play(Write(title))
        self.wait(2)

        # Transition
        self.play(FadeOut(title))

        # Axes for the process
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": False},
        )
        axes.shift(DOWN * 0.5)
        labels = axes.get_axis_labels(x_label="t", y_label="X_t")
        self.play(Create(axes), Write(labels))
        self.wait(1)

        # Generate a sample path up to time t0
        t0 = 5
        x_values = np.linspace(0, 10, 100)
        dt = x_values[1] - x_values[0]
        steps = np.random.normal(0, 0.3, size=x_values.shape)
        samples = np.cumsum(steps)
        samples = samples - samples[0]  # Start at zero

        # Split the sample path at t0
        index_t0 = np.searchsorted(x_values, t0)
        x_known = x_values[:index_t0]
        y_known = samples[:index_t0]
        x_future = x_values[index_t0 - 1 :]
        y_future = samples[index_t0 - 1 :]

        # Plot known part of the sample path
        known_path = axes.plot_line_graph(
            x_known,
            y_known,
            line_color=BLUE,
            add_vertex_dots=False,
        )
        self.play(Create(known_path), run_time=2)
        self.wait(1)

        # Highlight t0
        t0_line = axes.get_vertical_line(
            axes.coords_to_point(t0, y_known[-1]),
            line_func=DashedLine,
            line_config={"color": YELLOW},
        )
        t0_label = MathTex("t_0").next_to(t0_line, DOWN)
        self.play(Create(t0_line), Write(t0_label))
        self.wait(1)

        # Indicate known information up to t0
        info_box = Rectangle(width=4, height=1, color=WHITE)
        info_box.to_corner(UP + LEFT)
        info_text = Text(
            "At time $t_0$, we know $X_t$ for $t \\leq t_0$.", font_size=24
        )
        info_text.next_to(info_box.get_center(), DOWN)
        self.play(Create(info_box), Write(info_text))
        self.wait(2)

        # Plot future possible paths
        num_future_paths = 5
        future_paths = VGroup()
        colors = [GREEN, ORANGE, PURPLE, TEAL, RED]
        for i in range(num_future_paths):
            future_steps = np.random.normal(0, 0.3, size=x_future.shape)
            future_samples = np.cumsum(future_steps)
            future_samples = future_samples + y_known[-1]

            future_path = axes.plot_line_graph(
                x_future,
                future_samples,
                add_vertex_dots=False,
            )
            future_path.set_color(colors[i % len(colors)])
            future_path.set_stroke(opacity=0.5, width=2)
            future_paths.add(future_path)
        self.play(LaggedStartMap(Create, future_paths, lag_ratio=0.2), run_time=3)
        self.wait(2)

        # Emphasize unknown future
        unknown_text = Text("Future values ($t > t_0$) are unknown.", font_size=24)
        unknown_text.next_to(info_box.get_center(), DOWN)
        self.play(ReplacementTransform(info_text, unknown_text))
        self.wait(2)

        # Conclusion
        conclusion = Text(
            "An adapted process: \nAt any time $t$, we know $X_s$ for $s \\leq t$.",
            font_size=32,
        )
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(3)
