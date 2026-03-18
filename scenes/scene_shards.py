"""Scene 1: Shard Visualization — Volume → Shard → Minishard → Chunks.

The first scene in the tutorial. Shows the spatial hierarchy that
neuroglancer sharding creates, using a simplified small grid.
Split into 3 sub-videos at each zoom level.

Zoom sequence: zoom in on one shard so it fills the screen, THEN fade
out the other shards. Same pattern for minishard zoom.
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

from ngspec.morton import compressed_morton_code
from ngspec.sharding import compute_sharding_params
from scenes.common import SHARD_CUBE_COLORS

# Use a small grid so the visualization is legible
VIZ_GRID = (8, 8, 8)
SCALE_FACTOR = 0.5


def _wireframe_box(sx, sy, sz, offset, color=WHITE, thickness=0.01):
    """Create a wireframe bounding box from 12 edges."""
    ox, oy, oz = offset
    corners = [
        (ox, oy, oz), (ox + sx, oy, oz), (ox + sx, oy + sy, oz), (ox, oy + sy, oz),
        (ox, oy, oz + sz), (ox + sx, oy, oz + sz), (ox + sx, oy + sy, oz + sz), (ox, oy + sy, oz + sz),
    ]
    pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edges = VGroup()
    for i, j in pairs:
        edges.add(Line3D(
            start=list(corners[i]), end=list(corners[j]),
            color=color, thickness=thickness,
        ))
    return edges


def _chunk_cube(x, y, z, offset, scale, color, opacity=0.4):
    """Create a single chunk cube at grid position (x, y, z)."""
    ox, oy, oz = offset
    cube = Cube(
        side_length=scale * 0.85,
        fill_color=color, fill_opacity=opacity, stroke_width=0.5,
    )
    cube.move_to([
        ox + (x + 0.5) * scale,
        oy + (y + 0.5) * scale,
        oz + (z + 0.5) * scale,
    ])
    return cube


def _group_center(chunks, offset, sf):
    """Compute the 3D center of a list of chunk coordinates."""
    n = max(len(chunks), 1)
    cx = sum(offset[0] + (x + 0.5) * sf for x, y, z in chunks) / n
    cy = sum(offset[1] + (y + 0.5) * sf for x, y, z in chunks) / n
    cz = sum(offset[2] + (z + 0.5) * sf for x, y, z in chunks) / n
    return [cx, cy, cz]


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
        params = compute_sharding_params(tuple(size))
        pre = params["preshift_bits"]
        mini = params["minishard_bits"]
        shard_bits = params["shard_bits"]
        grid_full = params["grid_size"]

        gx, gy, gz = VIZ_GRID
        sf = SCALE_FACTOR
        sx, sy, sz = gx * sf, gy * sf, gz * sf
        offset = (-sx / 2, -sy / 2, -sz / 2)

        # Compute shard and minishard IDs for every chunk in viz grid
        shard_map = {}
        mini_map = {}
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    code = compressed_morton_code((x, y, z), grid_full)
                    shifted = code >> pre
                    mid = shifted & ((1 << mini) - 1) if mini > 0 else 0
                    sid = (shifted >> mini) & ((1 << shard_bits) - 1) if shard_bits > 0 else 0
                    shard_map[(x, y, z)] = sid
                    mini_map[(x, y, z)] = mid

        unique_shards = sorted(set(shard_map.values()))
        shard_colors = {s: SHARD_CUBE_COLORS[i % len(SHARD_CUBE_COLORS)]
                        for i, s in enumerate(unique_shards)}
        shards_to_show = unique_shards[:min(4, len(unique_shards))]
        target_shard = shards_to_show[0]

        # ──────────────────────────────────────────────
        # Section 1: Full volume with shards
        # ──────────────────────────────────────────────
        title = Text("Shard Visualization", font_size=28)
        title.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)

        info = Text(
            f"Volume: {size[0]}x{size[1]}x{size[2]}  |  "
            f"Showing {gx}x{gy}x{gz} chunk grid",
            font_size=16,
        )
        info.next_to(title, DOWN, buff=0.15)
        self.add_fixed_in_frame_mobjects(info)

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        bbox = _wireframe_box(sx, sy, sz, offset)
        self.play(FadeIn(bbox), run_time=0.5)

        # Color-code shard groups
        shard_groups = {}
        for sid in shards_to_show:
            grp = VGroup()
            for (x, y, z), s in shard_map.items():
                if s == sid:
                    grp.add(_chunk_cube(x, y, z, offset, sf, shard_colors[sid]))
            shard_groups[sid] = grp
            self.play(FadeIn(grp), run_time=0.6)

        # Brief rotation
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.wait(1)

        self.next_section("zoom-shard")

        # ──────────────────────────────────────────────
        # Section 2: Zoom into one shard, THEN fade others
        # ──────────────────────────────────────────────
        target_chunks = [(x, y, z) for (x, y, z), s in shard_map.items() if s == target_shard]
        center = _group_center(target_chunks, offset, sf)

        # Zoom in first so the target shard fills the screen
        self.move_camera(
            frame_center=center,
            zoom=3.0,
            run_time=1.5,
        )

        # Now fade out the other shards (and the bounding box)
        fade_others = [FadeOut(shard_groups[s]) for s in shards_to_show if s != target_shard]
        fade_others.append(FadeOut(bbox))
        self.play(*fade_others, run_time=0.8)
        self.wait(0.5)

        # Recolor chunks within this shard by minishard ID
        unique_minis = sorted(set(mini_map[c] for c in target_chunks))
        mini_colors = {m: SHARD_CUBE_COLORS[i % len(SHARD_CUBE_COLORS)]
                       for i, m in enumerate(unique_minis)}
        minis_to_show = unique_minis[:min(6, len(unique_minis))]
        target_mini = minis_to_show[0]

        # Replace shard-colored cubes with minishard-colored ones
        self.play(FadeOut(shard_groups[target_shard]), run_time=0.3)

        mini_groups = {}
        for mid in minis_to_show:
            grp = VGroup()
            for c in target_chunks:
                if mini_map[c] == mid:
                    grp.add(_chunk_cube(*c, offset, sf, mini_colors[mid], opacity=0.5))
            mini_groups[mid] = grp
            self.play(FadeIn(grp), run_time=0.3)

        self.wait(2)
        self.next_section("zoom-minishard")

        # ──────────────────────────────────────────────
        # Section 3: Zoom into one minishard, THEN fade others
        # ──────────────────────────────────────────────
        mini_chunks = [c for c in target_chunks if mini_map[c] == target_mini]
        mini_center = _group_center(mini_chunks, offset, sf)

        # Zoom in first
        self.move_camera(
            frame_center=mini_center,
            zoom=6.0,
            run_time=1.5,
        )

        # Fade out other minishards
        fade_minis = [FadeOut(mini_groups[m]) for m in minis_to_show if m != target_mini]
        if fade_minis:
            self.play(*fade_minis, run_time=0.8)

        # Recolor individual chunks with distinct colors
        self.play(FadeOut(mini_groups[target_mini]), run_time=0.3)
        chunk_cubes = VGroup()
        for i, c in enumerate(mini_chunks):
            color = SHARD_CUBE_COLORS[i % len(SHARD_CUBE_COLORS)]
            chunk_cubes.add(_chunk_cube(*c, offset, sf, color, opacity=0.6))
        self.play(FadeIn(chunk_cubes), run_time=0.5)

        # Hierarchy label
        hierarchy = Text(
            "volume -> shard -> minishard -> chunks",
            font_size=16, color=YELLOW,
        )
        hierarchy.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(hierarchy)

        self.wait(3)  # Final frame stays
