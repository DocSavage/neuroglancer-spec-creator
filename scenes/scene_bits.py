"""Scene 3: Bit Allocation Breakdown.

Shows how the morton code bits are partitioned into preshift, minishard,
and shard regions.  Uses little-endian ordering (MSB=shard on left,
LSB=preshift on right) with a visual 3D chunk connection.
"""

import math

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Brace,
    FadeIn,
    Indicate,
    Polygon,
    Scene,
    Text,
    VGroup,
)

from ngspec.morton import compressed_morton_code, bits_per_dimension
from ngspec.sharding import compute_sharding_params
from scenes.common import (
    MINISHARD_COLOR,
    PRESHIFT_COLOR,
    SHARD_COLOR,
    instant_title,
    make_colored_bit_strip,
)


def _isometric_cube(center, size=0.6):
    """Draw a small isometric cube as three filled polygons (2D scene)."""
    cx, cy = center
    s = size / 2
    # Isometric projection offsets
    dx, dy = s * 0.87, s * 0.5

    top = Polygon(
        [cx, cy + s, 0], [cx + dx, cy + dy, 0],
        [cx, cy, 0], [cx - dx, cy + dy, 0],
        fill_color="#4488FF", fill_opacity=0.5,
        stroke_color=WHITE, stroke_width=1,
    )
    left_face = Polygon(
        [cx - dx, cy + dy, 0], [cx, cy, 0],
        [cx, cy - s, 0], [cx - dx, cy - dy, 0],
        fill_color="#2266CC", fill_opacity=0.5,
        stroke_color=WHITE, stroke_width=1,
    )
    right_face = Polygon(
        [cx, cy, 0], [cx + dx, cy + dy, 0],
        [cx + dx, cy - dy, 0], [cx, cy - s, 0],
        fill_color="#3377DD", fill_opacity=0.5,
        stroke_color=WHITE, stroke_width=1,
    )
    return VGroup(left_face, right_face, top)


class BitAllocationScene(Scene):
    """Visualize how morton code bits split into shard/minishard/preshift."""

    def __init__(self, size=(94088, 78317, 134576), scale_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.volume_size = size
        self.scale_idx = scale_idx

    def construct(self):
        size = list(self.volume_size)
        for _ in range(self.scale_idx):
            size = [math.ceil(s / 2) for s in size]
        size = tuple(size)

        params = compute_sharding_params(size)
        pre = params["preshift_bits"]
        mini = params["minishard_bits"]
        shard = params["shard_bits"]
        total = params["total_chunk_bits"]
        grid = params["grid_size"]

        # ── Title and subtitle ──
        title = instant_title(self, f"Bit Allocation — Scale {self.scale_idx}")

        subtitle = Text(
            f"Size: {size[0]}x{size[1]}x{size[2]}  |  "
            f"Grid: {grid[0]}x{grid[1]}x{grid[2]}  |  "
            f"{total} total bits",
            font_size=18,
        )
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(subtitle), run_time=0.3)

        # ── Build colored bit strip (little-endian: shard|mini|pre) ──
        max_cells = min(total, 40)
        cell_w = min(0.3, 12.0 / (max_cells + 1))
        strip, shard_cells, mini_cells, pre_cells = make_colored_bit_strip(
            min(shard, max_cells),
            min(mini, max(max_cells - shard, 0)),
            min(pre, max(max_cells - shard - mini, 0)),
            cell_width=cell_w,
        )
        strip.move_to(UP * 0.8)
        self.play(FadeIn(strip), run_time=0.5)

        # ── MSB / LSB labels ──
        msb_label = Text("MSB", font_size=14, color=WHITE)
        lsb_label = Text("LSB", font_size=14, color=WHITE)
        msb_label.next_to(strip, UP, buff=0.15).align_to(strip, LEFT)
        lsb_label.next_to(strip, UP, buff=0.15).align_to(strip, RIGHT)
        self.play(FadeIn(msb_label), FadeIn(lsb_label), run_time=0.3)

        # ── Braces and labels ──
        braces = VGroup()

        # Minishard brace ABOVE the strip
        if mini > 0 and len(mini_cells) > 0:
            mini_brace = Brace(mini_cells, UP, color=MINISHARD_COLOR)
            mini_text = Text(
                f"minishard_bits: {mini}\n"
                f"2^{mini} = {2**mini} minishards/shard",
                font_size=12, color=MINISHARD_COLOR,
            )
            mini_text.next_to(mini_brace, UP, buff=0.08)
            braces.add(mini_brace, mini_text)

        # Shard brace BELOW the strip
        if shard > 0 and len(shard_cells) > 0:
            shard_brace = Brace(shard_cells, DOWN, color=SHARD_COLOR)
            shard_text = Text(
                f"shard_bits: {shard}\n"
                f"2^{shard} = {2**shard:,} shards",
                font_size=12, color=SHARD_COLOR,
            )
            shard_text.next_to(shard_brace, DOWN, buff=0.08)
            braces.add(shard_brace, shard_text)

        # Preshift brace BELOW the strip
        if pre > 0 and len(pre_cells) > 0:
            pre_brace = Brace(pre_cells, DOWN, color=PRESHIFT_COLOR)
            pre_text = Text(
                f"preshift_bits: {pre}\n"
                f"2^{pre} = {2**pre} chunks/minishard",
                font_size=12, color=PRESHIFT_COLOR,
            )
            pre_text.next_to(pre_brace, DOWN, buff=0.08)
            braces.add(pre_brace, pre_text)

        self.play(FadeIn(braces), run_time=0.5)
        self.wait(1)

        # ── 3D chunk visual with coordinate ──
        chunk_y = -2.2
        cube = _isometric_cube((0, chunk_y), size=0.7)

        # Example coordinate
        ex_coord = (42, 17, 83)
        coord_text = Text(
            f"chunk ({ex_coord[0]}, {ex_coord[1]}, {ex_coord[2]})",
            font_size=14,
        )
        coord_text.next_to(cube, RIGHT, buff=0.3)

        # Compute its morton code
        code = compressed_morton_code(ex_coord, grid)
        code_text = Text(
            f"morton = {code}",
            font_size=14, color=YELLOW,
        )
        code_text.next_to(coord_text, DOWN, buff=0.15)

        self.play(FadeIn(cube), FadeIn(coord_text), FadeIn(code_text), run_time=0.5)

        # Arrow from chunk to bit strip
        chunk_arrow = Arrow(
            cube.get_top(), strip.get_bottom(),
            color=YELLOW, stroke_width=2, buff=0.1,
        )
        self.play(FadeIn(chunk_arrow), run_time=0.3)

        # ── Highlight regions to show extraction ──
        self.wait(0.5)

        # Compute shard and minishard IDs for the example
        shifted = code >> pre
        mini_id = shifted & ((1 << mini) - 1) if mini > 0 else 0
        shard_id = (shifted >> mini) & ((1 << shard) - 1) if shard > 0 else 0

        if shard > 0 and len(shard_cells) > 0:
            shard_id_text = Text(
                f"shard_id = {shard_id}", font_size=14, color=SHARD_COLOR,
            )
            shard_id_text.next_to(shard_cells, LEFT, buff=0.3)
            self.play(Indicate(shard_cells, color=SHARD_COLOR), run_time=0.5)
            self.play(FadeIn(shard_id_text), run_time=0.3)

        if mini > 0 and len(mini_cells) > 0:
            mini_id_text = Text(
                f"minishard_id = {mini_id}", font_size=14, color=MINISHARD_COLOR,
            )
            mini_id_text.next_to(mini_cells, RIGHT, buff=0.3).shift(UP * 0.5)
            self.play(Indicate(mini_cells, color=MINISHARD_COLOR), run_time=0.5)
            self.play(FadeIn(mini_id_text), run_time=0.3)

        # ── Numeric summary ──
        summary = Text(
            f"{shard} + {mini} + {pre} = {total} bits",
            font_size=16, color=WHITE,
        )
        summary.to_edge(DOWN, buff=0.15)
        self.play(FadeIn(summary), run_time=0.3)

        self.wait(3)  # Final frame stays
