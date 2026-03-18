"""Scene 2: Compressed Morton Code Explained.

Shows how chunk coordinates become a single uint64 morton code.
Animates the full interleave table with one-at-a-time arrows so the
viewer sees dimensions drop out near the end.
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
    Rectangle,
    ReplacementTransform,
    Scene,
    SurroundingRectangle,
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

        # ──────────────────────────────────────────────
        # Section 1: Intro — dimensions, grid, bits
        # ──────────────────────────────────────────────
        title = instant_title(self, "Compressed Morton Code")

        vol_text = Text(
            f"Volume: {size[0]} x {size[1]} x {size[2]} voxels",
            font_size=22,
        )
        vol_text.next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(vol_text), run_time=0.5)
        self.wait(0.8)

        grid_text = Text(
            f"/ {chunk_size}  ->  Grid: {grid[0]} x {grid[1]} x {grid[2]} chunks",
            font_size=22,
        )
        grid_text.next_to(vol_text, DOWN, buff=0.25)
        self.play(FadeIn(grid_text), run_time=0.5)
        self.wait(0.8)

        bits_text = Text(
            f"Bits: X={bpd[0]}, Y={bpd[1]}, Z={bpd[2]}  (total {total})",
            font_size=22,
        )
        bits_text.next_to(grid_text, DOWN, buff=0.25)
        self.play(FadeIn(bits_text), run_time=0.5)
        self.wait(1.5)

        self.play(FadeOut(vol_text), FadeOut(grid_text), FadeOut(bits_text),
                  run_time=0.4)

        self.next_section("interleaving")

        # ──────────────────────────────────────────────
        # Section 2: Interleaving animation
        # ──────────────────────────────────────────────
        # Layout constants
        cols_per_row = 18
        gap = 0.03
        cell_w = min(0.35, (13.0 - gap * cols_per_row) / cols_per_row)
        cell_h = cell_w
        font = max(8, int(cell_w * 28))

        # --- Input bit rows ---
        input_rows = VGroup()
        input_cells = {}

        for dim in range(3):
            n = bpd[dim]
            y = 1.8 - dim * (cell_h + 0.2)
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

        # --- Output grid (multi-row) ---
        output_label = Text("Morton output:", font_size=font + 2, color=YELLOW)
        out_base_y = -0.6
        output_label.move_to(LEFT * 5.5 + UP * (out_base_y + cell_h * 0.8))

        output_cells = []
        output_group = VGroup()
        for j in range(total):
            row_idx = j // cols_per_row
            col_idx = j % cols_per_row
            x = -5.5 + col_idx * (cell_w + gap)
            y = out_base_y - row_idx * (cell_h + gap + 0.05)
            c = make_bit_cell(
                j, YELLOW, width=cell_w, height=cell_h, font_size=font,
            )
            c.move_to(RIGHT * x + UP * y)
            output_cells.append(c)
            output_group.add(c)

        self.play(FadeIn(output_label), FadeIn(output_group), run_time=0.5)
        self.wait(0.3)

        # --- Determine animation phases ---
        last_pos_for_dim = {}
        for j, (dim, _) in enumerate(table):
            last_pos_for_dim[dim] = j

        min_bpd = min(bpd)
        uniform_end = min_bpd * 3
        first_slow = min(3 * 3, uniform_end)  # first 3 full cycles

        # --- Phase 1: first cycles with arrows ---
        phase_label = Text(
            "All 3 dimensions contribute (X -> Y -> Z):",
            font_size=14, color=WHITE,
        )
        phase_label.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(phase_label), run_time=0.2)

        for j in range(min(first_slow, total)):
            dim, bit_pos = table[j]
            source = input_cells[(dim, bit_pos)]
            target = output_cells[j]

            # Arrow from source to output
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
            self.play(
                ReplacementTransform(target, new_cell),
                run_time=0.15,
            )
            output_cells[j] = new_cell
            self.play(FadeOut(arrow), run_time=0.1)

        # --- Phase 2: batch-fill uniform middle ---
        if uniform_end > first_slow:
            batch_text = Text(
                f"... same X->Y->Z pattern for bits {first_slow}-{uniform_end - 1} ...",
                font_size=14, color=YELLOW,
            )
            batch_text.to_edge(DOWN, buff=0.6)
            self.play(FadeIn(batch_text), run_time=0.2)

            batch_anims = []
            for j in range(first_slow, uniform_end):
                dim, bit_pos = table[j]
                target = output_cells[j]
                new_cell = make_bit_cell(
                    f"{DIM_LABELS[dim]}{bit_pos}", DIM_COLORS[dim],
                    width=cell_w, height=cell_h, font_size=font,
                )
                new_cell.move_to(target.get_center())
                batch_anims.append(ReplacementTransform(target, new_cell))
                output_cells[j] = new_cell

            self.play(AnimationGroup(*batch_anims), run_time=0.8)
            self.play(FadeOut(batch_text), run_time=0.2)

        # --- Phase 3: endgame with arrows ---
        if uniform_end < total:
            self.play(FadeOut(phase_label), run_time=0.2)

            prev_active = {0, 1, 2}
            endgame_label = None

            for j in range(uniform_end, total):
                dim, bit_pos = table[j]
                active = {d for d in range(3) if last_pos_for_dim[d] >= j}

                # Announce dimension dropout
                if active != prev_active:
                    dropped = prev_active - active
                    for d in dropped:
                        msg = Text(
                            f"{DIM_LABELS[d]} exhausted (all {bpd[d]} bits used)",
                            font_size=16, color=DIM_COLORS[d],
                        )
                        msg.to_edge(DOWN, buff=0.3)
                        if endgame_label:
                            self.play(FadeOut(endgame_label), run_time=0.15)
                        endgame_label = msg
                        self.play(FadeIn(msg), run_time=0.3)
                        self.wait(0.6)

                    active_names = " -> ".join(DIM_LABELS[d] for d in sorted(active))
                    remain = Text(
                        f"Now only: {active_names}", font_size=16, color=YELLOW,
                    )
                    remain.to_edge(DOWN, buff=0.6)
                    self.play(FadeOut(endgame_label), FadeIn(remain), run_time=0.3)
                    endgame_label = remain
                    prev_active = active

                source = input_cells[(dim, bit_pos)]
                target = output_cells[j]

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
                self.play(
                    ReplacementTransform(target, new_cell),
                    run_time=0.15,
                )
                output_cells[j] = new_cell
                self.play(FadeOut(arrow), run_time=0.1)

            if endgame_label:
                self.play(FadeOut(endgame_label), run_time=0.2)
        else:
            self.play(FadeOut(phase_label), run_time=0.2)

        # --- Final summary ---
        pattern_parts = []
        i = 0
        while i < len(table):
            dim = table[i][0]
            count = 1
            while i + count < len(table) and table[i + count][0] == dim:
                count += 1
            if count == 1:
                pattern_parts.append(DIM_LABELS[dim])
            else:
                pattern_parts.append(f"{DIM_LABELS[dim]}x{count}")
            i += count

        summary = Text(
            "Pattern: " + " -> ".join(pattern_parts),
            font_size=14, color=YELLOW,
        )
        summary.to_edge(DOWN, buff=0.5)

        insight = Text(
            "Dimensions drop out when they exhaust their bits!",
            font_size=16, color=YELLOW,
        )
        insight.to_edge(DOWN, buff=0.2)

        self.play(FadeIn(summary), FadeIn(insight), run_time=0.5)
        self.wait(3)  # Final frame stays
