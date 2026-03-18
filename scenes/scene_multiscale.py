"""Scene 4: Multi-Scale Bit Comparison.

Shows how the bit allocation changes across resolution scales by stacking
LSB-aligned colored bit strips.  Shard bits visibly shrink from the left
while preshift and minishard stay stable on the right.
"""

import math

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    FadeIn,
    Scene,
    Text,
    VGroup,
)

from ngspec.sharding import compute_sharding_params
from scenes.common import instant_title, make_colored_bit_strip


class MultiscaleBitComparison(Scene):
    """Stack LSB-aligned bit strips to show shard bits shrinking across scales."""

    def __init__(self, size=(94088, 78317, 134576), num_scales=11, **kwargs):
        super().__init__(**kwargs)
        self.volume_size = size
        self.num_scales = num_scales

    def construct(self):
        title = instant_title(self, "Multi-Scale Bit Comparison")

        # ── Compute params for all scales ──
        rows = []
        current_size = list(self.volume_size)
        for i in range(self.num_scales):
            params = compute_sharding_params(tuple(current_size))
            rows.append({
                "scale": i,
                "size": tuple(current_size),
                "grid": params["grid_size"],
                "shard": params["shard_bits"],
                "mini": params["minishard_bits"],
                "pre": params["preshift_bits"],
                "total": params["total_chunk_bits"],
            })
            current_size = [math.ceil(s / 2) for s in current_size]

        # ── Layout constants ──
        max_total = rows[0]["total"]
        # Fit the widest strip (scale 0) within frame
        cell_w = min(0.28, 11.0 / (max_total + 1))
        cell_h = 0.25
        gap = 0.02
        row_spacing = cell_h + 0.35
        start_y = 2.2

        # Right edge: all strips align here (LSB = right side)
        right_x = 4.5

        # ── Show strips one at a time ──
        strips = []
        labels = []

        # Limit to scales that fit on screen
        max_rows = min(self.num_scales, int((start_y + 3.5) / row_spacing))

        for i in range(max_rows):
            r = rows[i]
            y = start_y - i * row_spacing

            # Build strip
            strip, shard_cells, mini_cells, pre_cells = make_colored_bit_strip(
                r["shard"], r["mini"], r["pre"],
                cell_width=cell_w, cell_height=cell_h,
                font_size=max(7, int(cell_w * 26)), gap=gap,
            )

            # Right-align: position so right edge of strip aligns with right_x
            strip.move_to(UP * y)
            if len(strip) > 0:
                offset = right_x - strip.get_right()[0]
                strip.shift(RIGHT * offset)

            # Row label on the left
            grid_str = f"{r['grid'][0]}x{r['grid'][1]}x{r['grid'][2]}"
            label = Text(
                f"Scale {i}  ({grid_str})  {r['total']}b",
                font_size=11,
            )
            label.move_to(UP * y)
            if len(strip) > 0:
                label.next_to(strip, LEFT, buff=0.2)
            else:
                label.move_to(LEFT * 3 + UP * y)

            strips.append(strip)
            labels.append(label)

            if i == 0:
                self.play(FadeIn(strip), FadeIn(label), run_time=0.6)
                self.wait(0.8)
            elif i <= 3:
                self.play(FadeIn(strip), FadeIn(label), run_time=0.4)
                self.wait(0.4)
            else:
                self.play(FadeIn(strip), FadeIn(label), run_time=0.2)

        # ── Column legend ──
        if max_rows > 0 and len(strips[0]) > 0:
            # MSB / LSB markers aligned with scale 0 strip
            msb = Text("MSB", font_size=10, color=WHITE)
            lsb = Text("LSB", font_size=10, color=WHITE)
            msb.next_to(strips[0], UP, buff=0.08).align_to(strips[0], LEFT)
            lsb.next_to(strips[0], UP, buff=0.08).align_to(strips[0], RIGHT)
            self.play(FadeIn(msb), FadeIn(lsb), run_time=0.3)

        # ── Final insight ──
        insight = Text(
            "Shard bits shrink from the left as resolution decreases",
            font_size=16, color=YELLOW,
        )
        insight.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(insight), run_time=0.4)

        self.wait(3)  # Final frame stays
