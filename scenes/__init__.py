"""Manim scene definitions for neuroglancer sharding visualization."""


def render_scene(size, scales, scene, scale_idx, preview, quality):
    """Render the requested Manim scene(s)."""
    scene_map = {
        "morton": _render_morton,
        "bits": _render_bits,
        "shards": _render_shards,
        "multiscale": _render_multiscale,
    }

    renderer = "opengl" if preview else "cairo"

    if scene == "all":
        for name in ["morton", "bits", "shards", "multiscale"]:
            scene_map[name](size, scales, scale_idx, quality, renderer, preview)
    else:
        scene_map[scene](size, scales, scale_idx, quality, renderer, preview)


def _render_with_config(scene_cls, quality, renderer, preview, **kwargs):
    """Render a single scene with an isolated manim config.

    Each scene gets its own tempconfig so that manim's file writer
    produces a separate output file per scene class.
    """
    from manim import tempconfig

    with tempconfig({
        "quality": quality,
        "renderer": renderer,
        "preview": preview,
        "output_file": scene_cls.__name__,
    }):
        scene = scene_cls(**kwargs)
        scene.render()


def _render_morton(size, scales, scale_idx, quality, renderer, preview):
    from scenes.scene_morton import CompressedMortonScene
    _render_with_config(CompressedMortonScene, quality, renderer, preview,
                        size=size, scale_idx=scale_idx)


def _render_bits(size, scales, scale_idx, quality, renderer, preview):
    from scenes.scene_bits import BitAllocationScene
    _render_with_config(BitAllocationScene, quality, renderer, preview,
                        size=size, scale_idx=scale_idx)


def _render_shards(size, scales, scale_idx, quality, renderer, preview):
    from scenes.scene_shards import ShardVisualizationScene
    _render_with_config(ShardVisualizationScene, quality, renderer, preview,
                        size=size, scale_idx=scale_idx)


def _render_multiscale(size, scales, scale_idx, quality, renderer, preview):
    from scenes.scene_multiscale import MultiscaleWalkthroughScene
    _render_with_config(MultiscaleWalkthroughScene, quality, renderer, preview,
                        size=size, num_scales=scales)
