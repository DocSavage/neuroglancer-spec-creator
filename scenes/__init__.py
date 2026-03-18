"""Manim scene definitions for neuroglancer sharding visualization."""

import math

from ngspec.morton import bits_per_dimension, total_chunk_bits
from ngspec.sharding import compute_sharding_params


def render_scene(size, scales, scene, scale_idx, preview, quality):
    """Render the requested Manim scene(s)."""
    from manim import config, tempconfig

    renderer = "opengl" if preview else "cairo"

    with tempconfig({
        "quality": quality,
        "renderer": renderer,
        "preview": preview,
    }):
        scene_map = {
            "morton": _render_morton,
            "bits": _render_bits,
            "shards": _render_shards,
            "multiscale": _render_multiscale,
        }

        if scene == "all":
            for name in ["morton", "bits", "shards", "multiscale"]:
                scene_map[name](size, scales, scale_idx)
        else:
            scene_map[scene](size, scales, scale_idx)


def _render_morton(size, scales, scale_idx):
    from scenes.scene_morton import CompressedMortonScene
    scene = CompressedMortonScene(size=size, scale_idx=scale_idx)
    scene.render()


def _render_bits(size, scales, scale_idx):
    from scenes.scene_bits import BitAllocationScene
    scene = BitAllocationScene(size=size, scale_idx=scale_idx)
    scene.render()


def _render_shards(size, scales, scale_idx):
    from scenes.scene_shards import ShardVisualizationScene
    scene = ShardVisualizationScene(size=size, scale_idx=scale_idx)
    scene.render()


def _render_multiscale(size, scales, scale_idx):
    from scenes.scene_multiscale import MultiscaleWalkthroughScene
    scene = MultiscaleWalkthroughScene(size=size, num_scales=scales)
    scene.render()
