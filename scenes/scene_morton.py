"""Scene 1: Compressed Morton Code Explained.

Shows how chunk coordinates become a single uint64 morton code, and why
compressed interleaving differs from standard interleaving.
"""

import math

from manim import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Create,
    FadeIn,
    FadeOut,
    GrowArrow,
    MathTex,
    Rectangle,
    ReplacementTransform,
    Scene,
    SurroundingRectangle,
    Text,
    VGroup,
    Wait,
    Write,
)

from ngspec.morton import bits_per_dimension, interleave_table, total_chunk_bits

DIM_COLORS = [BLUE, GREEN, RED]  # X, Y, Z
DIM_LABELS = ["X", "Y", "Z"]


def _make_bit_cell(label: str, color, width=0.45, height=0.45):
    """Create a single bit cell (rectangle with text)."""
    rect = Rectangle(width=width, height=height, color=color, fill_opacity=0.3)
    text = Text(label, font_size=16, color=WHITE)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def _make_bit_row(labels: list[str], color, start_pos, width=0.45):
    """Create a horizontal row of bit cells."""
    cells = VGroup()
    for i, label in enumerate(labels):
        cell = _make_bit_cell(label, color, width=width)
        cell.move_to(start_pos + RIGHT * i * (width + 0.05))
        cells.add(cell)
    return cells


class CompressedMortonScene(Scene):
    """Animate the compressed morton code interleaving process."""

    def __init__(self, size=(94088, 78317, 134576), scale_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.volume_size = size
        self.scale_idx = scale_idx

    def construct(self):
        # Compute dimensions for the requested scale
        size = list(self.volume_size)
        for _ in range(self.scale_idx):
            size = [math.ceil(s / 2) for s in size]
        size = tuple(size)

        chunk_size = 64
        grid = tuple(math.ceil(s / chunk_size) for s in size)
        bpd = bits_per_dimension(grid)
        total = total_chunk_bits(grid)
        table = interleave_table(grid)

        # Title
        title = Text("Compressed Morton Code", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Step 1: Show volume dimensions
        vol_text = Text(
            f"Volume: {size[0]} × {size[1]} × {size[2]} voxels",
            font_size=24,
        )
        vol_text.next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(vol_text))
        self.wait(1)

        # Step 2: Divide by chunk size
        grid_text = Text(
            f"÷ {chunk_size} → Grid: {grid[0]} × {grid[1]} × {grid[2]} chunks",
            font_size=24,
        )
        grid_text.next_to(vol_text, DOWN, buff=0.3)
        self.play(FadeIn(grid_text))
        self.wait(1)

        # Step 3: Show bits per dimension
        bits_text = Text(
            f"Bits: X={bpd[0]}, Y={bpd[1]}, Z={bpd[2]}  (total={total})",
            font_size=24,
        )
        bits_text.next_to(grid_text, DOWN, buff=0.3)
        self.play(FadeIn(bits_text))
        self.wait(1)

        # Fade out intro text
        self.play(FadeOut(vol_text), FadeOut(grid_text), FadeOut(bits_text))

        # Step 4: Build the input bit rows
        # Limit display to manageable width
        max_display_bits = min(max(bpd), 14)
        cell_width = min(0.45, 12.0 / (max_display_bits + 2))

        input_rows = VGroup()
        input_cells = {}  # (dim, bit_pos) -> cell

        for dim in range(3):
            n_bits = min(bpd[dim], max_display_bits)
            label = Text(f"{DIM_LABELS[dim]}:", font_size=20, color=DIM_COLORS[dim])
            y_offset = 1.5 - dim * 0.8

            labels = [f"{DIM_LABELS[dim]}{i}" for i in range(n_bits)]
            row = _make_bit_row(labels, DIM_COLORS[dim],
                                start_pos=LEFT * 5 + UP * y_offset,
                                width=cell_width)
            label.next_to(row, LEFT, buff=0.2)

            for i in range(n_bits):
                input_cells[(dim, i)] = row[i]

            input_rows.add(label, row)

        self.play(FadeIn(input_rows))
        self.wait(1)

        # Step 5: Build output row and animate interleaving
        output_label = Text("Morton:", font_size=20, color=YELLOW)
        output_y = -1.5
        n_output = min(total, max_display_bits * 3)

        output_cells = VGroup()
        for j in range(n_output):
            cell = _make_bit_cell(str(j), YELLOW, width=cell_width, height=cell_width)
            cell.move_to(LEFT * 5 + UP * output_y + RIGHT * j * (cell_width + 0.05))
            output_cells.add(cell)

        output_label.next_to(output_cells, LEFT, buff=0.2)
        self.play(FadeIn(output_label), FadeIn(output_cells))
        self.wait(0.5)

        # Animate bits flying from input to output
        # Show first several bits animated, then fast-forward
        n_animated = min(len(table), 12)
        for j in range(n_animated):
            dim, bit_pos = table[j]
            if (dim, bit_pos) in input_cells and j < len(output_cells):
                source = input_cells[(dim, bit_pos)]
                target = output_cells[j]

                # Highlight source
                highlight = SurroundingRectangle(source, color=WHITE, buff=0.05)
                self.play(Create(highlight), run_time=0.3)

                # Create arrow from source to target
                arrow = Arrow(
                    source.get_bottom(), target.get_top(),
                    color=DIM_COLORS[dim], buff=0.1,
                    stroke_width=2,
                )
                self.play(GrowArrow(arrow), run_time=0.3)

                # Color the output cell
                new_cell = _make_bit_cell(
                    f"{DIM_LABELS[dim]}{bit_pos}",
                    DIM_COLORS[dim], width=cell_width, height=cell_width,
                )
                new_cell.move_to(target.get_center())
                self.play(
                    ReplacementTransform(target, new_cell),
                    FadeOut(highlight),
                    FadeOut(arrow),
                    run_time=0.3,
                )
                output_cells[j] = new_cell

        # Fast-forward remaining bits
        if len(table) > n_animated:
            remaining_text = Text(
                f"... {len(table) - n_animated} more bits interleaved ...",
                font_size=20, color=YELLOW,
            )
            remaining_text.next_to(output_cells, DOWN, buff=0.5)
            self.play(FadeIn(remaining_text))
            self.wait(1)
            self.play(FadeOut(remaining_text))

        # Step 6: Show the key insight
        self.wait(0.5)
        insight_group = VGroup()

        insight_title = Text("Key Insight", font_size=28, color=YELLOW)
        insight_title.to_edge(DOWN, buff=1.5)

        insight1 = Text(
            "Standard morton: always cycles X→Y→Z→X→Y→Z...",
            font_size=20,
        )
        insight1.next_to(insight_title, DOWN, buff=0.3)

        # Build the actual pattern for compressed morton
        pattern_parts = []
        for j, (dim, _) in enumerate(table[:18]):
            pattern_parts.append(DIM_LABELS[dim])
        pattern_str = "→".join(pattern_parts) + "→..."

        insight2 = Text(
            f"Compressed: {pattern_str}",
            font_size=18,
        )
        insight2.next_to(insight1, DOWN, buff=0.2)

        insight3 = Text(
            "Dimensions drop out when they run out of bits!",
            font_size=20, color=YELLOW,
        )
        insight3.next_to(insight2, DOWN, buff=0.2)

        insight_group.add(insight_title, insight1, insight2, insight3)
        self.play(FadeIn(insight_group))
        self.wait(3)

        self.play(FadeOut(VGroup(title, input_rows, output_label, output_cells, insight_group)))
