"""Scene 1: Shard Visualization — Volume → Shard → Minishard → Chunks.

Shows the spatial hierarchy by starting with the full volume divided
into shard-sized blocks, zooming into one shard to reveal minishards,
then zooming into one minishard to reveal chunks.

Each level is a self-contained cube grid at a different scale:
- Volume = 8×8×8 shard blocks
- One shard = NxNxN minishard blocks (based on minishard_bits)
- One minishard = MxMxM chunk blocks (based on preshift_bits)
"""

import math

from manim import (
    DEGREES,
    DOWN,
    UP,
    WHITE,
    YELLOW,
    Cube,
    FadeIn,
    FadeOut,
    Line3D,
    Text,
    ThreeDScene,
    VGroup,
)

from scenes.common import SHARD_CUBE_COLORS

# Each cube in the grid is this fraction of one grid unit
CUBE_FILL = 0.85


def _make_grid_cubes(nx, ny, nz, scale, offset, color, opacity=0.3):
    """Create a grid of cubes. Returns (VGroup_of_all, dict[(x,y,z)] -> cube)."""
    cubes = VGroup()
    cube_map = {}
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                c = Cube(
                    side_length=scale * CUBE_FILL,
                    fill_color=color, fill_opacity=opacity,
                    stroke_width=0.5,
                )
                c.move_to([
                    offset[0] + (x + 0.5) * scale,
                    offset[1] + (y + 0.5) * scale,
                    offset[2] + (z + 0.5) * scale,
                ])
                cubes.add(c)
                cube_map[(x, y, z)] = c
    return cubes, cube_map


def _grid_lines(nx, ny, nz, scale, offset, color=WHITE, thickness=0.005, opacity=0.2):
    """Create translucent grid lines subdividing a box."""
    ox, oy, oz = offset
    sx, sy, sz = nx * scale, ny * scale, nz * scale
    lines = VGroup()
    # Lines along X (varying y, z divisions)
    for iy in range(ny + 1):
        for iz in range(nz + 1):
            lines.add(Line3D(
                [ox, oy + iy * scale, oz + iz * scale],
                [ox + sx, oy + iy * scale, oz + iz * scale],
                color=color, thickness=thickness,
            ))
    # Lines along Y
    for ix in range(nx + 1):
        for iz in range(nz + 1):
            lines.add(Line3D(
                [ox + ix * scale, oy, oz + iz * scale],
                [ox + ix * scale, oy + sy, oz + iz * scale],
                color=color, thickness=thickness,
            ))
    # Lines along Z
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            lines.add(Line3D(
                [ox + ix * scale, oy + iy * scale, oz],
                [ox + ix * scale, oy + iy * scale, oz + sz],
                color=color, thickness=thickness,
            ))
    return lines


def _cube_center(x, y, z, scale, offset):
    return [
        offset[0] + (x + 0.5) * scale,
        offset[1] + (y + 0.5) * scale,
        offset[2] + (z + 0.5) * scale,
    ]


class ShardVisualizationScene(ThreeDScene):
    """3D hierarchy: volume → shards → minishards → chunks."""

    def __init__(self, size=(94088, 78317, 134576), scale_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.volume_size = size
        self.scale_idx = scale_idx

    def construct(self):
        size = list(self.volume_size)
        for _ in range(self.scale_idx):
            size = [math.ceil(s / 2) for s in size]

        # Compute subdivision counts from bit params
        from ngspec.sharding import compute_sharding_params
        params = compute_sharding_params(tuple(size))
        pre = params["preshift_bits"]
        mini = params["minishard_bits"]

        # Minishards per shard: 2^mini, laid out as cube root per axis
        n_mini = 2 ** mini
        mini_per_axis = max(1, round(n_mini ** (1 / 3)))

        # Chunks per minishard: 2^pre, laid out as cube root per axis
        n_chunks = 2 ** pre
        chunks_per_axis = max(1, round(n_chunks ** (1 / 3)))

        # ══════════════════════════════════════════════
        # Section 1: Full volume as 8×8×8 shard grid
        # ══════════════════════════════════════════════
        title = Text("Shard Visualization", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)

        info = Text(
            f"Volume: {size[0]}x{size[1]}x{size[2]}  |  "
            f"8x8x8 shards, {n_mini} minishards/shard, "
            f"{n_chunks} chunks/minishard",
            font_size=14,
        )
        info.next_to(title, DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(info)

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        # Draw shard grid
        shard_n = 8
        shard_scale = 0.45
        shard_total = shard_n * shard_scale
        shard_offset = (-shard_total / 2, -shard_total / 2, -shard_total / 2)

        shard_color = SHARD_CUBE_COLORS[0]
        all_shards, shard_cubes = _make_grid_cubes(
            shard_n, shard_n, shard_n, shard_scale, shard_offset,
            color=shard_color, opacity=0.15,
        )
        shard_grid = _grid_lines(
            shard_n, shard_n, shard_n, shard_scale, shard_offset,
            opacity=0.15,
        )

        self.play(FadeIn(shard_grid), run_time=0.5)
        self.play(FadeIn(all_shards), run_time=0.8)

        # Rotate
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)

        self.next_section("zoom-shard")

        # ══════════════════════════════════════════════
        # Section 2: Zoom into one shard, fade others, show minishards
        # ══════════════════════════════════════════════
        # Pick a shard near center
        target = (3, 3, 3)
        target_center = _cube_center(*target, shard_scale, shard_offset)

        # Zoom in so this shard fills the screen
        self.move_camera(
            frame_center=target_center,
            zoom=5.0,
            run_time=1.5,
        )

        # Fade out ALL other shard cubes, keeping only the target
        target_cube = shard_cubes[target]
        others = VGroup(*[c for coord, c in shard_cubes.items() if coord != target])
        self.play(FadeOut(others), FadeOut(shard_grid), run_time=0.8)
        self.wait(0.5)

        # Now show minishards within this one shard
        # The shard cube occupies a region from target_center - half_size to + half_size
        half = shard_scale * CUBE_FILL / 2
        mini_scale = (shard_scale * CUBE_FILL) / mini_per_axis
        mini_offset = (
            target_center[0] - half,
            target_center[1] - half,
            target_center[2] - half,
        )

        # Fade out the shard cube, replace with minishard cubes
        self.play(FadeOut(target_cube), run_time=0.3)

        mini_color = SHARD_CUBE_COLORS[2]
        all_minis, mini_cubes = _make_grid_cubes(
            mini_per_axis, mini_per_axis, mini_per_axis,
            mini_scale, mini_offset,
            color=mini_color, opacity=0.3,
        )
        self.play(FadeIn(all_minis), run_time=0.6)
        self.wait(1.5)

        self.next_section("zoom-minishard")

        # ══════════════════════════════════════════════
        # Section 3: Zoom into one minishard, fade others, show chunks
        # ══════════════════════════════════════════════
        mini_target = (1, 1, 1)
        mini_target_center = _cube_center(*mini_target, mini_scale, mini_offset)

        # Zoom in
        self.move_camera(
            frame_center=mini_target_center,
            zoom=15.0,
            run_time=1.5,
        )

        # Fade out other minishards
        mini_target_cube = mini_cubes[mini_target]
        mini_others = VGroup(*[c for coord, c in mini_cubes.items() if coord != mini_target])
        self.play(FadeOut(mini_others), run_time=0.8)
        self.wait(0.5)

        # Show chunks within this minishard
        mini_half = mini_scale * CUBE_FILL / 2
        chunk_scale = (mini_scale * CUBE_FILL) / chunks_per_axis
        chunk_offset = (
            mini_target_center[0] - mini_half,
            mini_target_center[1] - mini_half,
            mini_target_center[2] - mini_half,
        )

        self.play(FadeOut(mini_target_cube), run_time=0.3)

        chunk_cubes_group = VGroup()
        idx = 0
        for cx in range(chunks_per_axis):
            for cy in range(chunks_per_axis):
                for cz in range(chunks_per_axis):
                    color = SHARD_CUBE_COLORS[idx % len(SHARD_CUBE_COLORS)]
                    c = Cube(
                        side_length=chunk_scale * CUBE_FILL,
                        fill_color=color, fill_opacity=0.5,
                        stroke_width=0.5,
                    )
                    c.move_to(_cube_center(cx, cy, cz, chunk_scale, chunk_offset))
                    chunk_cubes_group.add(c)
                    idx += 1

        self.play(FadeIn(chunk_cubes_group), run_time=0.6)

        # Hierarchy label
        hierarchy = Text(
            "volume -> shard -> minishard -> chunks",
            font_size=16, color=YELLOW,
        )
        hierarchy.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(hierarchy)

        self.wait(3)  # Final frame stays
