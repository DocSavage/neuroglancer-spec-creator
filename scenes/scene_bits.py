"""Scene 2: Bit Allocation Breakdown.

Shows how the morton code bits are partitioned into preshift, minishard,
and shard regions.
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
    Brace,
    FadeIn,
    FadeOut,
    MathTex,
    Rectangle,
    Scene,
    Text,
    VGroup,
    Write,
)

from ngspec.morton import bits_per_dimension, total_chunk_bits
from ngspec.sharding import compute_sharding_params

PRESHIFT_COLOR = GREEN
MINISHARD_COLOR = YELLOW
SHARD_COLOR = RED


def _make_region_cells(n_bits, color, start_x, y, cell_width=0.35):
    """Create a row of colored cells representing a bit region."""
    cells = VGroup()
    for i in range(n_bits):
        rect = Rectangle(
            width=cell_width, height=0.4,
            color=color, fill_opacity=0.4,
            stroke_width=1,
        )
        rect.move_to(RIGHT * (start_x + i * (cell_width + 0.02)) + UP * y)
        cells.add(rect)
    return cells


class BitAllocationScene(Scene):
    """Visualize how morton code bits split into preshift/minishard/shard."""

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

        # Title
        title = Text(f"Bit Allocation — Scale {self.scale_idx}", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Subtitle with dimensions
        grid = params["grid_size"]
        subtitle = Text(
            f"Size: {size[0]}×{size[1]}×{size[2]}  |  Grid: {grid[0]}×{grid[1]}×{grid[2]}  |  {total} total bits",
            font_size=20,
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(subtitle))

        # Build the morton code strip
        max_cells = min(total, 40)
        cell_width = min(0.35, 13.0 / (max_cells + 1))
        strip_start_x = -max_cells * (cell_width + 0.02) / 2
        y_pos = 1.0

        # Color each bit by its region
        # Bits 0..pre-1 = preshift, pre..pre+mini-1 = minishard, pre+mini..total = shard
        all_cells = VGroup()

        # Preshift cells
        pre_display = min(pre, max_cells)
        pre_cells = _make_region_cells(pre_display, PRESHIFT_COLOR, strip_start_x, y_pos, cell_width)
        all_cells.add(pre_cells)

        # Minishard cells
        mini_display = min(mini, max_cells - pre_display)
        if mini_display > 0:
            mini_start = strip_start_x + pre_display * (cell_width + 0.02)
            mini_cells = _make_region_cells(mini_display, MINISHARD_COLOR, mini_start, y_pos, cell_width)
            all_cells.add(mini_cells)
        else:
            mini_cells = VGroup()

        # Shard cells
        shard_display = min(shard, max_cells - pre_display - mini_display)
        if shard_display > 0:
            shard_start = strip_start_x + (pre_display + mini_display) * (cell_width + 0.02)
            shard_cells = _make_region_cells(shard_display, SHARD_COLOR, shard_start, y_pos, cell_width)
            all_cells.add(shard_cells)
        else:
            shard_cells = VGroup()

        self.play(FadeIn(all_cells))
        self.wait(0.5)

        # Add braces and labels
        labels = VGroup()

        if pre_display > 0:
            pre_brace = Brace(pre_cells, DOWN, color=PRESHIFT_COLOR)
            pre_label = Text(
                f"preshift_bits: {pre}\n2^{pre} = {2**pre} chunks/minishard",
                font_size=16, color=PRESHIFT_COLOR,
            )
            pre_label.next_to(pre_brace, DOWN, buff=0.1)
            labels.add(pre_brace, pre_label)

        if mini_display > 0:
            mini_brace = Brace(mini_cells, DOWN, color=MINISHARD_COLOR)
            mini_label = Text(
                f"minishard_bits: {mini}\n2^{mini} = {2**mini} minishards/shard",
                font_size=16, color=MINISHARD_COLOR,
            )
            mini_label.next_to(mini_brace, DOWN, buff=0.1)
            labels.add(mini_brace, mini_label)

        if shard_display > 0:
            shard_brace = Brace(shard_cells, DOWN, color=SHARD_COLOR)
            shard_label = Text(
                f"shard_bits: {shard}\n2^{shard} = {2**shard:,} shards",
                font_size=16, color=SHARD_COLOR,
            )
            shard_label.next_to(shard_brace, DOWN, buff=0.1)
            labels.add(shard_brace, shard_label)

        self.play(FadeIn(labels))
        self.wait(1)

        # Pipeline diagram
        pipeline_y = -2.0
        pipeline = Text(
            "chunk_coord → morton_code → >>preshift → hash → extract_minishard → extract_shard",
            font_size=18,
        )
        pipeline.move_to(UP * pipeline_y)
        self.play(FadeIn(pipeline))
        self.wait(1)

        # Show numeric example
        example = Text(
            f"Total: {pre} + {mini} + {shard} = {total} bits  |  "
            f"{2**shard:,} shards × {2**mini} minishards × {2**pre} chunks",
            font_size=18, color=WHITE,
        )
        example.next_to(pipeline, DOWN, buff=0.4)
        self.play(FadeIn(example))
        self.wait(3)

        self.play(FadeOut(VGroup(title, subtitle, all_cells, labels, pipeline, example)))
