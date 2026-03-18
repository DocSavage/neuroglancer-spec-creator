"""Manim scene definitions for neuroglancer sharding visualization."""

import multiprocessing


def render_scene(size, scales, scene, scale_idx, preview, quality):
    """Render the requested Manim scene(s)."""
    renderer = "opengl" if preview else "cairo"

    if scene == "all":
        names = ["morton", "bits", "shards", "multiscale"]
        jobs = [
            _make_job(name, size, scales, scale_idx, quality, renderer, preview)
            for name in names
        ]
        _run_parallel(jobs)
    else:
        job = _make_job(scene, size, scales, scale_idx, quality, renderer, preview)
        _run_parallel([job])


def _make_job(scene_name, size, scales, scale_idx, quality, renderer, preview):
    """Build the arguments tuple for a single scene render worker."""
    scene_info = {
        "morton": ("scenes.scene_morton", "CompressedMortonScene",
                   {"size": size, "scale_idx": scale_idx}),
        "bits": ("scenes.scene_bits", "BitAllocationScene",
                 {"size": size, "scale_idx": scale_idx}),
        "shards": ("scenes.scene_shards", "ShardVisualizationScene",
                   {"size": size, "scale_idx": scale_idx}),
        "multiscale": ("scenes.scene_multiscale", "MultiscaleWalkthroughScene",
                       {"size": size, "num_scales": scales}),
    }
    module_path, class_name, kwargs = scene_info[scene_name]
    return (module_path, class_name, kwargs, quality, renderer, preview)


def _run_parallel(jobs):
    """Run scene render jobs in parallel using multiprocessing."""
    if len(jobs) == 1:
        _render_worker(jobs[0])
        return

    with multiprocessing.Pool(processes=len(jobs)) as pool:
        pool.map(_render_worker, jobs)


def _render_worker(args):
    """Worker function that runs in a subprocess.

    Imports manim and the scene class fresh in each process to avoid
    sharing global config state.
    """
    module_path, class_name, kwargs, quality, renderer, preview = args
    import importlib
    from manim import tempconfig

    module = importlib.import_module(module_path)
    scene_cls = getattr(module, class_name)

    with tempconfig({
        "quality": quality,
        "renderer": renderer,
        "preview": preview,
        "output_file": class_name,
    }):
        scene = scene_cls(**kwargs)
        scene.render()
