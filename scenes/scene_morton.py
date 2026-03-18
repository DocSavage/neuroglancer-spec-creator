"""Scene 1: Compressed Morton Code Explained.

Shows how chunk coordinates become a single uint64 morton code, and why
compressed interleaving differs from standard interleaving.  The animation
runs through the ENTIRE interleave table so the viewer can see dimensions
drop out near the end.
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
    AnimationGroup,
    Create,
    FadeIn,
    FadeOut,
    Rectangle,
    ReplacementTransform,
    Scene,
    SurroundingRectangle,
    Text,
    VGroup,
    Write,
)

from ngspec.morton import bits_per_dimension, interleave_table, total_chunk_bits

DIM_COLORS = [BLUE, GREEN, RED]  # X, Y, Z
DIM_LABELS = ["X", "Y", "Z"]


def _make_bit_cell(label, color, width=0.35, height=0.35, font_size=12):
    rect = Rectangle(width=width, height=height, color=color, fill_opacity=0.3)
    text = Text(str(label), font_size=font_size, color=WHITE)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


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

        # --- Title ---
        title = Text("Compressed Morton Code", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # --- Intro: dimensions → grid → bits ---
        vol_text = Text(
            f"Volume: {size[0]} × {size[1]} × {size[2]} voxels",
            font_size=22,
        )
        vol_text.next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(vol_text))
        self.wait(1)

        grid_text = Text(
            f"÷ {chunk_size} → Grid: {grid[0]} × {grid[1]} × {grid[2]} chunks",
            font_size=22,
        )
        grid_text.next_to(vol_text, DOWN, buff=0.25)
        self.play(FadeIn(grid_text))
        self.wait(1)

        bits_text = Text(
            f"Bits needed: X={bpd[0]}, Y={bpd[1]}, Z={bpd[2]}  (total {total})",
            font_size=22,
        )
        bits_text.next_to(grid_text, DOWN, buff=0.25)
        self.play(FadeIn(bits_text))
        self.wait(1.5)

        self.play(FadeOut(vol_text), FadeOut(grid_text), FadeOut(bits_text))

        # --- Layout constants ---
        # Fit all bits on screen using a multi-row output grid
        cols_per_row = 18  # output cells per row
        gap = 0.03
        cell_w = min(0.35, (13.0 - gap * cols_per_row) / cols_per_row)
        cell_h = cell_w
        font = max(8, int(cell_w * 28))

        # --- Input bit rows (top area) ---
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
                c = _make_bit_cell(
                    f"{DIM_LABELS[dim]}{i}", DIM_COLORS[dim],
                    width=cell_w, height=cell_h, font_size=font,
                )
                c.move_to(LEFT * 5.5 + RIGHT * i * (cell_w + gap) + UP * y)
                row.add(c)
                input_cells[(dim, i)] = c
            label.next_to(row, LEFT, buff=0.15)
            input_rows.add(label, row)

        self.play(FadeIn(input_rows))
        self.wait(0.5)

        # --- Output grid (bottom area, multi-row) ---
        # Use a plain list so ReplacementTransform doesn't break indexing.
        output_label = Text("Morton output:", font_size=font + 2, color=YELLOW)
        out_base_y = -0.6
        output_label.move_to(LEFT * 5.5 + UP * (out_base_y + cell_h * 0.8))

        output_cells = []  # plain list — VGroup used only for initial FadeIn
        output_group = VGroup()
        for j in range(total):
            row_idx = j // cols_per_row
            col_idx = j % cols_per_row
            x = -5.5 + col_idx * (cell_w + gap)
            y = out_base_y - row_idx * (cell_h + gap + 0.05)
            c = _make_bit_cell(
                j, YELLOW,
                width=cell_w, height=cell_h, font_size=font,
            )
            c.move_to(RIGHT * x + UP * y)
            output_cells.append(c)
            output_group.add(c)

        self.play(FadeIn(output_label), FadeIn(output_group))
        self.wait(0.5)

        # --- Determine which bits are "interesting" ---
        # Find where each dimension drops out: the last bit position i
        # where that dimension contributes.
        last_pos_for_dim = {}
        for j, (dim, _) in enumerate(table):
            last_pos_for_dim[dim] = j

        # The "endgame" starts when the first dimension drops out.
        # Show the beginning (first cycle), batch the uniform middle,
        # and animate the endgame one-by-one.
        min_bpd = min(bpd)
        # Bits in the uniform region: first min_bpd full X/Y/Z cycles = min_bpd * 3
        uniform_end = min_bpd * 3
        # Animate first full cycle (3 bits) individually, batch rest of uniform, then endgame
        first_slow = min(3 * 3, uniform_end)  # first 3 cycles or all uniform

        # --- Phase 1: animate first few cycles slowly ---
        phase_label = Text(
            "All 3 dimensions contribute (X→Y→Z):",
            font_size=16, color=WHITE,
        )
        phase_label.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(phase_label))

        arrows = VGroup()

        for j in range(min(first_slow, total)):
            dim, bit_pos = table[j]
            source = input_cells[(dim, bit_pos)]
            target = output_cells[j]

            highlight = SurroundingRectangle(source, color=WHITE, buff=0.03,
                                             stroke_width=2)
            new_cell = _make_bit_cell(
                f"{DIM_LABELS[dim]}{bit_pos}", DIM_COLORS[dim],
                width=cell_w, height=cell_h, font_size=font,
            )
            new_cell.move_to(target.get_center())

            self.play(
                Create(highlight),
                ReplacementTransform(target, new_cell),
                run_time=0.25,
            )
            output_cells[j] = new_cell
            self.play(FadeOut(highlight), run_time=0.1)

        # --- Phase 2: batch-fill the uniform middle ---
        if uniform_end > first_slow:
            batch_text = Text(
                f"… same X→Y→Z pattern for bits {first_slow}–{uniform_end - 1} …",
                font_size=16, color=YELLOW,
            )
            batch_text.to_edge(DOWN, buff=0.6)
            self.play(FadeIn(batch_text), run_time=0.3)

            batch_anims = []
            for j in range(first_slow, uniform_end):
                dim, bit_pos = table[j]
                target = output_cells[j]
                new_cell = _make_bit_cell(
                    f"{DIM_LABELS[dim]}{bit_pos}", DIM_COLORS[dim],
                    width=cell_w, height=cell_h, font_size=font,
                )
                new_cell.move_to(target.get_center())
                batch_anims.append(ReplacementTransform(target, new_cell))
                output_cells[j] = new_cell

            self.play(AnimationGroup(*batch_anims), run_time=0.8)
            self.play(FadeOut(batch_text), run_time=0.3)

        # --- Phase 3: endgame — dimensions drop out one by one ---
        if uniform_end < total:
            self.play(FadeOut(phase_label), run_time=0.2)

            # Figure out which dimensions are still active at each step
            remaining_dims_at = []
            for j in range(uniform_end, total):
                active = [d for d in range(3) if last_pos_for_dim[d] >= j]
                remaining_dims_at.append(active)

            prev_active_set = {0, 1, 2}
            endgame_label = None

            for idx, j in enumerate(range(uniform_end, total)):
                dim, bit_pos = table[j]
                active = set(remaining_dims_at[idx])

                # Announce when a dimension drops out
                if active != prev_active_set:
                    dropped = prev_active_set - active
                    for d in dropped:
                        drop_msg = Text(
                            f"{DIM_LABELS[d]} exhausted (all {bpd[d]} bits used)",
                            font_size=18,
                            color=DIM_COLORS[d],
                        )
                        drop_msg.to_edge(DOWN, buff=0.3)
                        if endgame_label:
                            self.play(FadeOut(endgame_label), run_time=0.15)
                        endgame_label = drop_msg
                        self.play(FadeIn(drop_msg), run_time=0.3)
                        self.wait(0.8)
                    prev_active_set = active

                    # Show which dims remain
                    active_names = " → ".join(DIM_LABELS[d] for d in sorted(active))
                    remain_msg = Text(
                        f"Now only: {active_names}",
                        font_size=18, color=YELLOW,
                    )
                    remain_msg.to_edge(DOWN, buff=0.6)
                    self.play(FadeOut(endgame_label), FadeIn(remain_msg), run_time=0.3)
                    endgame_label = remain_msg

                source = input_cells[(dim, bit_pos)]
                target = output_cells[j]

                highlight = SurroundingRectangle(source, color=WHITE, buff=0.03,
                                                 stroke_width=2)
                new_cell = _make_bit_cell(
                    f"{DIM_LABELS[dim]}{bit_pos}", DIM_COLORS[dim],
                    width=cell_w, height=cell_h, font_size=font,
                )
                new_cell.move_to(target.get_center())

                self.play(
                    Create(highlight),
                    ReplacementTransform(target, new_cell),
                    run_time=0.3,
                )
                output_cells[j] = new_cell
                self.play(FadeOut(highlight), run_time=0.1)

            if endgame_label:
                self.play(FadeOut(endgame_label), run_time=0.2)
        else:
            self.play(FadeOut(phase_label), run_time=0.2)

        # --- Final summary ---
        self.wait(0.5)

        # Build the dimension pattern string showing the full sequence
        # Group consecutive same-dimension runs for readability
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
                pattern_parts.append(f"{DIM_LABELS[dim]}×{count}")
            i += count

        summary = Text(
            "Full pattern: " + " → ".join(pattern_parts),
            font_size=16, color=YELLOW,
        )
        summary.to_edge(DOWN, buff=0.6)

        insight = Text(
            "Dimensions drop out when they run out of bits — the pattern is NOT always X→Y→Z!",
            font_size=18, color=YELLOW,
        )
        insight.to_edge(DOWN, buff=0.25)

        self.play(FadeIn(summary), FadeIn(insight))
        self.wait(4)

        self.play(
            FadeOut(VGroup(title, input_rows, output_label, output_group,
                           summary, insight))
        )
