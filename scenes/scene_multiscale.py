"""Scene 4: Multi-Scale Walkthrough.

Ties it all together — shows how sharding parameters change across
resolution scales with a summary table and the final JSON output.
"""

import json
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
    FadeIn,
    FadeOut,
    Scene,
    Table,
    Text,
    VGroup,
    Write,
)

from ngspec.morton import bits_per_dimension
from ngspec.sharding import compute_sharding_params
from ngspec.spec_generator import generate_spec


class MultiscaleWalkthroughScene(Scene):
    """Animate the multi-scale sharding parameter breakdown."""

    def __init__(self, size=(94088, 78317, 134576), num_scales=11, **kwargs):
        super().__init__(**kwargs)
        self.volume_size = size
        self.num_scales = num_scales

    def construct(self):
        title = Text("Multi-Scale Sharding Parameters", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Build data for all scales
        rows = []
        current_size = list(self.volume_size)

        for i in range(self.num_scales):
            params = compute_sharding_params(tuple(current_size))
            bpd = bits_per_dimension(params["grid_size"])
            rows.append({
                "scale": i,
                "size": tuple(current_size),
                "grid": params["grid_size"],
                "bpd": bpd,
                "total": params["total_chunk_bits"],
                "shard": params["shard_bits"],
                "mini": params["minishard_bits"],
                "pre": params["preshift_bits"],
                "num_shards": params["num_shards"],
            })
            current_size = [math.ceil(s / 2) for s in current_size]

        # Show scales one at a time, then build the summary table
        # First show a few scales animated
        n_animated = min(4, self.num_scales)
        prev_group = None

        for i in range(n_animated):
            r = rows[i]
            scale_group = VGroup()

            scale_title = Text(f"Scale {i}", font_size=28, color=YELLOW)
            scale_title.next_to(title, DOWN, buff=0.5)

            size_text = Text(
                f"Size: {r['size'][0]} × {r['size'][1]} × {r['size'][2]}",
                font_size=22,
            )
            size_text.next_to(scale_title, DOWN, buff=0.3)

            grid_text = Text(
                f"Grid: {r['grid'][0]} × {r['grid'][1]} × {r['grid'][2]}",
                font_size=22,
            )
            grid_text.next_to(size_text, DOWN, buff=0.2)

            bits_text = Text(
                f"Bits: {r['bpd'][0]}+{r['bpd'][1]}+{r['bpd'][2]} = {r['total']}",
                font_size=22,
            )
            bits_text.next_to(grid_text, DOWN, buff=0.2)

            alloc_text = Text(
                f"shard={r['shard']}  mini={r['mini']}  pre={r['pre']}  →  {r['num_shards']:,} shards",
                font_size=22,
            )
            alloc_text.next_to(bits_text, DOWN, buff=0.2)

            # Color the allocation
            shard_colored = Text(f"shard={r['shard']}", font_size=22, color=RED)
            mini_colored = Text(f"mini={r['mini']}", font_size=22, color=YELLOW)
            pre_colored = Text(f"pre={r['pre']}", font_size=22, color=GREEN)

            scale_group.add(scale_title, size_text, grid_text, bits_text, alloc_text)

            if prev_group:
                self.play(FadeOut(prev_group), FadeIn(scale_group), run_time=0.8)
            else:
                self.play(FadeIn(scale_group))

            self.wait(1.5)
            prev_group = scale_group

        if prev_group:
            self.play(FadeOut(prev_group))

        # Build summary table
        header = ["Scale", "Bits", "Shard", "Mini", "Pre", "#Shards"]
        table_data = []
        for r in rows:
            table_data.append([
                str(r["scale"]),
                f"{r['bpd'][0]}+{r['bpd'][1]}+{r['bpd'][2]}={r['total']}",
                str(r["shard"]),
                str(r["mini"]),
                str(r["pre"]),
                f"{r['num_shards']:,}",
            ])

        # Manim Table can be large; limit if needed
        max_rows = min(len(table_data), 11)
        display_data = table_data[:max_rows]

        table = Table(
            display_data,
            col_labels=[Text(h, font_size=16) for h in header],
            include_outer_lines=True,
            line_config={"stroke_width": 1},
            element_to_mobject_config={"font_size": 14},
        )
        table.scale(0.7)
        table.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(table))
        self.wait(3)

        # Show total shards
        total_shards = sum(r["num_shards"] for r in rows)
        total_text = Text(
            f"Total shard files across all scales: {total_shards:,}",
            font_size=22, color=YELLOW,
        )
        total_text.next_to(table, DOWN, buff=0.4)
        self.play(FadeIn(total_text))
        self.wait(2)

        # Final: show JSON output preview
        self.play(FadeOut(table), FadeOut(total_text))

        json_title = Text("Output: neuroglancer_spec.json", font_size=24, color=GREEN)
        json_title.next_to(title, DOWN, buff=0.5)

        spec = generate_spec(
            self.volume_size,
            num_scales=self.num_scales,
        )
        # Show abbreviated JSON
        json_str = json.dumps(spec, indent=2)
        lines = json_str.split("\n")
        preview_lines = lines[:12] + ["  ..."] + lines[-3:]
        json_preview = Text(
            "\n".join(preview_lines),
            font_size=12,
            font="Monospace",
        )
        json_preview.next_to(json_title, DOWN, buff=0.3)

        self.play(FadeIn(json_title), FadeIn(json_preview))
        self.wait(4)

        self.play(FadeOut(VGroup(title, json_title, json_preview)))
