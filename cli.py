"""CLI entry point for neuroglancer spec creator.

Usage:
    pixi run generate --size 94088,77248,134592 --scales 11
    pixi run animate --size 94088,77248,134592 --scene morton
"""

import json
import math
import sys

import click

from ngspec.morton import bits_per_dimension, total_chunk_bits
from ngspec.sharding import compute_sharding_params
from ngspec.spec_generator import generate_spec


def parse_size(ctx, param, value):
    """Parse comma-separated volume dimensions."""
    try:
        parts = [int(x.strip()) for x in value.split(",")]
        if len(parts) != 3:
            raise ValueError
        return tuple(parts)
    except (ValueError, AttributeError):
        raise click.BadParameter("Expected 3 comma-separated integers (e.g., 94088,77248,134592)")


def parse_resolution(ctx, param, value):
    """Parse comma-separated resolution values."""
    try:
        parts = [float(x.strip()) for x in value.split(",")]
        if len(parts) == 1:
            return (parts[0], parts[0], parts[0])
        if len(parts) == 3:
            return tuple(parts)
        raise ValueError
    except (ValueError, AttributeError):
        raise click.BadParameter("Expected 1 or 3 comma-separated numbers (e.g., 8 or 8,8,8)")


@click.group()
def cli():
    """Neuroglancer precomputed volume spec generator."""
    pass


@cli.command()
@click.option("--size", required=True, callback=parse_size,
              help="Volume dimensions as X,Y,Z (e.g., 94088,77248,134592)")
@click.option("--scales", default=11, type=int, help="Number of resolution scales")
@click.option("--resolution", default="8", callback=parse_resolution,
              help="Base voxel resolution (e.g., 8 or 8,8,8)")
@click.option("--chunk-size", default=64, type=int, help="Chunk size in voxels")
@click.option("--data-type", default="uint8",
              type=click.Choice(["uint8", "uint16", "uint32", "uint64", "float32"]),
              help="Voxel data type")
@click.option("--volume-type", default="image",
              type=click.Choice(["image", "segmentation"]),
              help="Volume type")
@click.option("--encoding", default=None, help="Encoding (default: jpeg for image, compressed_segmentation for segmentation)")
@click.option("--output", "-o", default="neuroglancer_spec.json",
              help="Output JSON file path")
@click.option("--target-preshift", default=9, type=int, help="Target preshift_bits")
@click.option("--target-minishard", default=6, type=int, help="Target minishard_bits")
def generate(size, scales, resolution, chunk_size, data_type, volume_type,
             encoding, output, target_preshift, target_minishard):
    """Generate a neuroglancer multiscale volume spec JSON."""

    # Print summary table
    click.echo()
    click.echo(f"{'Scale':<6} {'Size':<28} {'Grid':<22} {'Bits':<16} {'Shard':>5} {'Mini':>5} {'Pre':>5} {'Total':>6} {'#Shards':>10}")
    click.echo("-" * 105)

    current_size = size
    for i in range(scales):
        params = compute_sharding_params(
            current_size,
            chunk_size=chunk_size,
            target_preshift=target_preshift,
            target_minishard=target_minishard,
        )
        bpd = bits_per_dimension(params["grid_size"])
        size_str = f"{current_size[0]}x{current_size[1]}x{current_size[2]}"
        grid_str = f"{params['grid_size'][0]}x{params['grid_size'][1]}x{params['grid_size'][2]}"
        bits_str = f"{bpd[0]}+{bpd[1]}+{bpd[2]}={params['total_chunk_bits']}"

        click.echo(
            f"{i:<6} {size_str:<28} {grid_str:<22} {bits_str:<16} "
            f"{params['shard_bits']:>5} {params['minishard_bits']:>5} "
            f"{params['preshift_bits']:>5} {params['total_chunk_bits']:>6} "
            f"{params['num_shards']:>10}"
        )

        current_size = tuple(math.ceil(s / 2) for s in current_size)

    click.echo()

    # Generate and write spec
    spec = generate_spec(
        size,
        num_scales=scales,
        voxel_resolution=resolution,
        data_type=data_type,
        volume_type=volume_type,
        chunk_size=chunk_size,
        encoding=encoding,
        target_preshift=target_preshift,
        target_minishard=target_minishard,
    )

    with open(output, "w") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")

    click.echo(f"Written: {output}")


@cli.command()
@click.option("--size", required=True, callback=parse_size,
              help="Volume dimensions as X,Y,Z")
@click.option("--scales", default=3, type=int, help="Number of scales to animate")
@click.option("--scene", default="all",
              type=click.Choice(["all", "morton", "bits", "shards", "multiscale"]),
              help="Which scene to render")
@click.option("--scale", default=0, type=int, help="Which scale index to visualize")
@click.option("--preview", is_flag=True, help="Use OpenGL interactive preview")
@click.option("--quality", default="medium_quality",
              type=click.Choice(["low_quality", "medium_quality", "high_quality"]),
              help="Render quality")
def animate(size, scales, scene, scale, preview, quality):
    """Render Manim educational animations."""
    try:
        from scenes import render_scene
    except ImportError:
        click.echo("Error: Manim is required for animations. Install with: pixi install -e viz", err=True)
        sys.exit(1)

    render_scene(size, scales, scene, scale, preview, quality)


if __name__ == "__main__":
    cli()
