"""Scene 2: Compressed Morton Code Explained.

Shows how chunk coordinates become a single uint64 morton code.
Uses a single output row with an ellipsis for the uniform middle,
and animates arrows for visible bits — especially the endgame where
dimensions drop out.
"""

import math

from manim import (
    AnimationGroup,
    Arrow,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    ReplacementTransform,
    Scene,
    Text,
    VGroup,
)

from ngspec.morton import bits_per_dimension, interleave_table, total_chunk_bits
from scenes.common import DIM_COLORS, DIM_LABELS, instant_title, make_bit_cell


class CompressedMortonScene(Scene):
    """Animate the compressed morton code interleaving process."""

    def __init__(self, size=(94088, 78317, 134576), scale_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.volume_size = size
        self.scale_idx = scale_idx

    def construct(self):
        size = list(self.volume_size)
        for _ in range(self.scale_idx):
            size = [math.ceil(s / 2) for s in size]
        size = tuple(size)

        chunk_size = 64
        grid = tuple(math.ceil(s / chunk_size) for s in size)
        bpd = bits_per_dimension(grid)
        total = total_chunk_bits(grid)
        table = interleave_table(grid)

        # ── Determine which output bits to show vs skip ──
        # Show first N bits (head), skip uniform middle, show last M bits (tail)
        # The tail includes the final uniform cycles PLUS the endgame so
        # viewers see the dimension dropout happen.
        min_bpd = min(bpd)
        uniform_end = min_bpd * 3
        first_show = min(4 * 3, uniform_end)  # first 4 full X→Y→Z cycles

        # Tail: at least 2 full cycles before dropout + all endgame bits
        tail_start = max(first_show, uniform_end - 2 * 3)  # 2 cycles before endgame
        tail_bits = list(range(tail_start, total))

        visible_bits = list(range(first_show))
        ellipsis_pos = len(visible_bits)
        visible_bits.extend(tail_bits)

        has_ellipsis = tail_start > first_show
        n_visible = len(visible_bits) + (1 if has_ellipsis else 0)

        # ── Layout constants ──
        gap = 0.03
        cell_w = min(0.35, (13.0 - gap * n_visible) / max(n_visible, 1))
        cell_h = cell_w
        font = max(8, int(cell_w * 28))

        # ── Section 1: Intro — keep text visible ──
        title = instant_title(self, "Compressed Morton Code")

        vol_text = Text(
            f"Volume: {size[0]} x {size[1]} x {size[2]} voxels",
            font_size=20,
        )
        vol_text.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(vol_text), run_time=0.5)
        self.wait(0.5)

        grid_text = Text(
            f"/ {chunk_size}  ->  Grid: {grid[0]} x {grid[1]} x {grid[2]} chunks",
            font_size=20,
        )
        grid_text.next_to(vol_text, DOWN, buff=0.2)
        self.play(FadeIn(grid_text), run_time=0.5)
        self.wait(0.5)

        bits_text = Text(
            f"Bits: X={bpd[0]}, Y={bpd[1]}, Z={bpd[2]}  (total {total})",
            font_size=20,
        )
        bits_text.next_to(grid_text, DOWN, buff=0.2)
        self.play(FadeIn(bits_text), run_time=0.5)
        self.wait(1)

        self.next_section("interleaving")

        # ── Section 2: Input rows + single output row with ellipsis ──
        # Input bit rows (compact, below the intro text)
        input_rows = VGroup()
        input_cells = {}
        input_top_y = bits_text.get_bottom()[1] - 0.5

        for dim in range(3):
            n = bpd[dim]
            y = input_top_y - dim * (cell_h + 0.15)
            label = Text(
                f"{DIM_LABELS[dim]} ({n}):",
                font_size=font + 2,
                color=DIM_COLORS[dim],
            )
            row = VGroup()
            for i in range(n):
                c = make_bit_cell(
                    f"{DIM_LABELS[dim]}{i}", DIM_COLORS[dim],
                    width=cell_w, height=cell_h, font_size=font,
                )
                c.move_to(LEFT * 5.5 + RIGHT * i * (cell_w + gap) + UP * y)
                row.add(c)
                input_cells[(dim, i)] = c
            label.next_to(row, LEFT, buff=0.15)
            input_rows.add(label, row)

        self.play(FadeIn(input_rows), run_time=0.5)

        # Single output row with ellipsis in the middle
        output_label = Text("Morton:", font_size=font + 2, color=YELLOW)
        out_y = input_top_y - 3 * (cell_h + 0.15) - 0.4

        output_cells = {}  # table_index -> cell mobject
        output_group = VGroup()
        col = 0
        x_start = -5.5

        for vi, table_j in enumerate(visible_bits):
            if has_ellipsis and vi == ellipsis_pos:
                # Insert ellipsis
                ell = Text("...", font_size=font + 4, color=YELLOW)
                ell.move_to(RIGHT * (x_start + col * (cell_w + gap)) + UP * out_y)
                output_group.add(ell)
                col += 1

            x = x_start + col * (cell_w + gap)
            c = make_bit_cell(
                table_j, YELLOW, width=cell_w, height=cell_h, font_size=font,
            )
            c.move_to(RIGHT * x + UP * out_y)
            output_cells[table_j] = c
            output_group.add(c)
            col += 1

        output_label.move_to(LEFT * 5.5 + UP * (out_y + cell_h * 0.7))
        output_label.align_to(output_group, LEFT)
        output_label.shift(LEFT * 0.8)
        self.play(FadeIn(output_label), FadeIn(output_group), run_time=0.5)
        self.wait(0.3)

        # ── Phase 1: animate arrows for first visible bits ──
        for table_j in visible_bits[:first_show]:
            dim, bit_pos = table[table_j]
            if (dim, bit_pos) not in input_cells or table_j not in output_cells:
                continue
            source = input_cells[(dim, bit_pos)]
            target = output_cells[table_j]

            arrow = Arrow(
                source.get_bottom(), target.get_top(),
                color=DIM_COLORS[dim], buff=0.05,
                stroke_width=2, max_tip_length_to_length_ratio=0.15,
            )
            new_cell = make_bit_cell(
                f"{DIM_LABELS[dim]}{bit_pos}", DIM_COLORS[dim],
                width=cell_w, height=cell_h, font_size=font,
            )
            new_cell.move_to(target.get_center())

            self.play(Create(arrow), run_time=0.2)
            self.play(ReplacementTransform(target, new_cell), run_time=0.15)
            output_cells[table_j] = new_cell
            self.play(FadeOut(arrow), run_time=0.1)

        # ── Phase 2: endgame — all bits after uniform_end with arrows ──
        last_pos_for_dim = {}
        for j, (dim, _) in enumerate(table):
            last_pos_for_dim[dim] = j

        prev_active = {0, 1, 2}

        for table_j in visible_bits[first_show:]:
            dim, bit_pos = table[table_j]
            active = {d for d in range(3) if last_pos_for_dim[d] >= table_j}

            if active != prev_active:
                prev_active = active

            if (dim, bit_pos) not in input_cells or table_j not in output_cells:
                continue
            source = input_cells[(dim, bit_pos)]
            target = output_cells[table_j]

            arrow = Arrow(
                source.get_bottom(), target.get_top(),
                color=DIM_COLORS[dim], buff=0.05,
                stroke_width=2, max_tip_length_to_length_ratio=0.15,
            )
            new_cell = make_bit_cell(
                f"{DIM_LABELS[dim]}{bit_pos}", DIM_COLORS[dim],
                width=cell_w, height=cell_h, font_size=font,
            )
            new_cell.move_to(target.get_center())

            self.play(Create(arrow), run_time=0.25)
            self.play(ReplacementTransform(target, new_cell), run_time=0.15)
            output_cells[table_j] = new_cell
            self.play(FadeOut(arrow), run_time=0.1)

        self.wait(3)  # Final frame stays — no subtitles
