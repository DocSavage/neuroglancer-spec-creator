"""Scene 3: 3D Shard Visualization.

Shows what shards look like in 3D space relative to the volume bounding box.
Uses a simplified small volume to keep the visualization legible.
"""

import math

from manim import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    DEGREES,
    Cube,
    FadeIn,
    FadeOut,
    Line3D,
    Text,
    ThreeDScene,
    VGroup,
    Write,
)

from ngspec.morton import compressed_morton_code
from ngspec.sharding import compute_sharding_params

SHARD_COLORS = [BLUE, GREEN, RED, ORANGE, YELLOW, "#FF69B4", "#00CED1", "#FFD700"]


class ShardVisualizationScene(ThreeDScene):
    """3D visualization of shard spatial extent in a volume."""

    def __init__(self, size=(94088, 78317, 134576), scale_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.volume_size = size
        self.scale_idx = scale_idx

    def construct(self):
        size = list(self.volume_size)
        for _ in range(self.scale_idx):
            size = [math.ceil(s / 2) for s in size]

        params = compute_sharding_params(tuple(size))

        # Use a simplified grid for visualization (max 8 chunks per dim)
        viz_grid = tuple(min(g, 8) for g in params["grid_size"])
        scale_factor = 0.6  # size of each cube unit

        # Title (shown as fixed-in-frame text)
        title = Text(f"3D Shard Visualization — Scale {self.scale_idx}", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        info = Text(
            f"Grid: {params['grid_size'][0]}×{params['grid_size'][1]}×{params['grid_size'][2]}  "
            f"(showing {viz_grid[0]}×{viz_grid[1]}×{viz_grid[2]})",
            font_size=18,
        )
        info.next_to(title, DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(info)
        self.play(FadeIn(info))

        # Set camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        # Draw wireframe bounding box
        gx, gy, gz = viz_grid
        sx, sy, sz = gx * scale_factor, gy * scale_factor, gz * scale_factor
        offset_x, offset_y, offset_z = -sx / 2, -sy / 2, -sz / 2

        edges = []
        corners = [
            (0, 0, 0), (sx, 0, 0), (sx, sy, 0), (0, sy, 0),
            (0, 0, sz), (sx, 0, sz), (sx, sy, sz), (0, sy, sz),
        ]
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edge_pairs:
            c1, c2 = corners[i], corners[j]
            line = Line3D(
                start=[c1[0] + offset_x, c1[1] + offset_y, c1[2] + offset_z],
                end=[c2[0] + offset_x, c2[1] + offset_y, c2[2] + offset_z],
                color=WHITE,
                thickness=0.01,
            )
            edges.append(line)

        bbox = VGroup(*edges)
        self.play(FadeIn(bbox))
        self.wait(0.5)

        # Compute shard IDs for each chunk in the viz grid and color them
        pre = params["preshift_bits"]
        mini = params["minishard_bits"]
        shard_bits = params["shard_bits"]
        grid_full = params["grid_size"]

        # Build shard map for viz grid
        shard_map = {}
        for x in range(viz_grid[0]):
            for y in range(viz_grid[1]):
                for z in range(viz_grid[2]):
                    code = compressed_morton_code((x, y, z), grid_full)
                    shifted = code >> pre
                    shard_id = (shifted >> mini) & ((1 << shard_bits) - 1) if shard_bits > 0 else 0
                    shard_map[(x, y, z)] = shard_id

        # Find unique shards and assign colors
        unique_shards = sorted(set(shard_map.values()))
        shard_color_map = {
            sid: SHARD_COLORS[i % len(SHARD_COLORS)]
            for i, sid in enumerate(unique_shards)
        }

        # Show a few shards at a time (first 4 unique shard IDs)
        shards_to_show = unique_shards[:min(4, len(unique_shards))]
        cubes = VGroup()

        for sid in shards_to_show:
            shard_cubes = VGroup()
            for (x, y, z), s in shard_map.items():
                if s == sid:
                    cube = Cube(
                        side_length=scale_factor * 0.85,
                        fill_color=shard_color_map[sid],
                        fill_opacity=0.4,
                        stroke_width=0.5,
                    )
                    cube.move_to([
                        offset_x + (x + 0.5) * scale_factor,
                        offset_y + (y + 0.5) * scale_factor,
                        offset_z + (z + 0.5) * scale_factor,
                    ])
                    shard_cubes.add(cube)

            if len(shard_cubes) > 0:
                cubes.add(shard_cubes)
                self.play(FadeIn(shard_cubes), run_time=0.8)

        self.wait(1)

        # Rotate camera to show 3D structure
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        # Legend
        legend_items = []
        for sid in shards_to_show:
            legend_items.append(f"Shard {sid}")
        legend_text = Text(
            f"Showing {len(shards_to_show)} of {len(unique_shards)} shards",
            font_size=16,
        )
        legend_text.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(legend_text)
        self.play(FadeIn(legend_text))
        self.wait(2)

        self.play(FadeOut(VGroup(bbox, cubes)), FadeOut(title), FadeOut(info), FadeOut(legend_text))
