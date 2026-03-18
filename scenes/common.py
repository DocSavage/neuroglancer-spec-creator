"""Shared constants and helpers for all Manim scenes."""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Rectangle,
    Text,
    VGroup,
)

# Dimension colors for compressed morton code (X, Y, Z)
DIM_COLORS = [BLUE, GREEN, RED]
DIM_LABELS = ["X", "Y", "Z"]

# Bit region colors (consistent across scenes 3 and 4)
SHARD_COLOR = RED
MINISHARD_COLOR = YELLOW
PRESHIFT_COLOR = GREEN

# Shard cube colors for 3D visualization
SHARD_CUBE_COLORS = [BLUE, GREEN, RED, "#FFA500", YELLOW, "#FF69B4", "#00CED1", "#FFD700"]


def make_bit_cell(label, color, width=0.35, height=0.35, font_size=12):
    """Create a single bit cell: colored rectangle with centered text label."""
    rect = Rectangle(width=width, height=height, color=color, fill_opacity=0.3)
    text = Text(str(label), font_size=font_size, color=WHITE)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def make_colored_bit_strip(shard, mini, pre, cell_width=0.3, cell_height=0.35,
                           font_size=10, gap=0.02):
    """Build a horizontal bit strip in little-endian order.

    Layout left to right: shard (MSB) | minishard | preshift (LSB).

    Returns (strip, shard_cells, mini_cells, pre_cells) where strip is the
    full VGroup and each *_cells is a VGroup of that region's cells (may be
    empty if that region has 0 bits).
    """
    strip = VGroup()
    shard_cells = VGroup()
    mini_cells = VGroup()
    pre_cells = VGroup()

    idx = 0
    for n, color, group in [
        (shard, SHARD_COLOR, shard_cells),
        (mini, MINISHARD_COLOR, mini_cells),
        (pre, PRESHIFT_COLOR, pre_cells),
    ]:
        for _ in range(n):
            cell = Rectangle(
                width=cell_width, height=cell_height,
                color=color, fill_opacity=0.4, stroke_width=1,
            )
            cell.move_to(RIGHT * idx * (cell_width + gap))
            group.add(cell)
            strip.add(cell)
            idx += 1

    return strip, shard_cells, mini_cells, pre_cells


def instant_title(scene, text_str, font_size=36, buff=0.3):
    """Add a title instantly (no animation) at the top of the frame."""
    title = Text(text_str, font_size=font_size)
    title.to_edge(UP, buff=buff)
    scene.add(title)
    return title
