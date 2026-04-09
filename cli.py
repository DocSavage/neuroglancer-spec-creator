"""CLI entry point for neuroglancer spec creator.

Usage:
    pixi run generate --size 94088,77248,134592 --scales 11
    pixi run generate --size 204800,114688,10254 --resolution 8,8,30 --chunk-size 64
    pixi run animate --size 94088,77248,134592 --scene morton
"""

import json
import math
import sys

import click

from ngspec.morton import bits_per_dimension, total_chunk_bits
from ngspec.sharding import compute_sharding_params
from ngspec.spec_generator import (
    _dims_to_halve,
    _double_resolution,
    _halve_size,
    _normalize_chunk_size,
    compute_num_scales,
    generate_spec,
)


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


def parse_chunk_size(ctx, param, value):
    """Parse chunk size as a single int or 3 comma-separated ints."""
    try:
        parts = [int(x.strip()) for x in value.split(",")]
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 3:
            return tuple(parts)
        raise ValueError
    except (ValueError, AttributeError):
        raise click.BadParameter("Expected 1 or 3 comma-separated integers (e.g., 64 or 128,128,32)")


@click.group()
def cli():
    """Neuroglancer precomputed volume spec generator."""
    pass


@cli.command()
@click.option("--size", required=True, callback=parse_size,
              help="Volume dimensions as X,Y,Z (e.g., 94088,77248,134592)")
@click.option("--scales", default=None, type=int,
              help="Number of resolution scales (default: auto-computed, stops at 1x1x1 grid)")
@click.option("--resolution", default="8", callback=parse_resolution,
              help="Base voxel resolution in nm (e.g., 8 or 8,8,30)")
@click.option("--chunk-size", default="64", callback=parse_chunk_size,
              help="Chunk size in voxels (e.g., 64 or 128,128,32)")
@click.option("--em", "preset", flag_value="em",
              help="EM grayscale preset (uint8, jpeg, image)")
@click.option("--seg", "preset", flag_value="seg", default=True,
              help="Segmentation preset [default] (uint64, compressed_segmentation, gzip)")
@click.option("--target-preshift", default=9, type=int, help="Target preshift_bits")
@click.option("--target-minishard", default=6, type=int, help="Target minishard_bits")
@click.option("--decimal-keys", is_flag=True,
              help="Use decimal format for resolution keys (e.g., 8.0x8.0x30.0 instead of 8x8x30)")
def generate(size, scales, resolution, chunk_size, preset,
             target_preshift, target_minishard, decimal_keys):
    """Generate a neuroglancer multiscale volume spec JSON to stdout.

    By default generates a segmentation spec (--seg). Use --em for EM grayscale.
    The summary table is printed to stderr; JSON goes to stdout for redirection.
    """

    # Apply preset
    if preset == "em":
        data_type = "uint8"
        volume_type = "image"
        encoding = "jpeg"
        data_encoding = None
        voxel_offset = None
        compressed_segmentation_block_size = None
    else:  # seg (default)
        data_type = "uint64"
        volume_type = "segmentation"
        encoding = "compressed_segmentation"
        data_encoding = "gzip"
        voxel_offset = (0, 0, 0)
        compressed_segmentation_block_size = (8, 8, 8)

    cs = _normalize_chunk_size(chunk_size)
    max_initial_res = max(resolution)
    is_anisotropic = not all(r == resolution[0] for r in resolution)

    # Resolve scale count
    max_scales = compute_num_scales(size, chunk_size, resolution)
    if scales is None:
        scales = max_scales
    else:
        if scales > max_scales:
            click.echo(
                f"Note: clamping --scales {scales} to {max_scales} "
                f"(grid reaches 1x1x1 at scale {max_scales - 1})",
                err=True,
            )
            scales = max_scales

    # Print summary table to stderr
    click.echo(err=True)
    click.echo(f"Mode: {preset} ({data_type}, {encoding})", err=True)
    click.echo(err=True)
    res_header = " Resolution" if is_anisotropic else ""
    click.echo(
        f"{'Scale':<6} {'Size':<28} {'Grid':<22} {'Bits':<16} "
        f"{'Shard':>5} {'Mini':>5} {'Pre':>5} {'Total':>6} {'#Shards':>10}"
        f"{res_header}",
        err=True,
    )
    click.echo("-" * (105 + (20 if is_anisotropic else 0)), err=True)

    current_size = size
    current_res = resolution
    for i in range(scales):
        params = compute_sharding_params(
            current_size,
            chunk_size=cs,
            target_preshift=target_preshift,
            target_minishard=target_minishard,
        )
        bpd = bits_per_dimension(params["grid_size"])
        size_str = f"{current_size[0]}x{current_size[1]}x{current_size[2]}"
        grid_str = f"{params['grid_size'][0]}x{params['grid_size'][1]}x{params['grid_size'][2]}"
        bits_str = f"{bpd[0]}+{bpd[1]}+{bpd[2]}={params['total_chunk_bits']}"

        res_str = ""
        if is_anisotropic:
            res_str = f" [{current_res[0]:.0f},{current_res[1]:.0f},{current_res[2]:.0f}]"

        click.echo(
            f"{i:<6} {size_str:<28} {grid_str:<22} {bits_str:<16} "
            f"{params['shard_bits']:>5} {params['minishard_bits']:>5} "
            f"{params['preshift_bits']:>5} {params['total_chunk_bits']:>6} "
            f"{params['num_shards']:>10}"
            f"{res_str}",
            err=True,
        )

        dims = _dims_to_halve(current_res, max_initial_res)
        current_size = _halve_size(current_size, dims)
        current_res = _double_resolution(current_res, dims)

    click.echo(err=True)

    # Generate spec and write JSON to stdout
    spec = generate_spec(
        size,
        num_scales=scales,
        voxel_resolution=resolution,
        data_type=data_type,
        volume_type=volume_type,
        chunk_size=chunk_size,
        encoding=encoding,
        data_encoding=data_encoding,
        target_preshift=target_preshift,
        target_minishard=target_minishard,
        decimal_keys=decimal_keys,
        voxel_offset=voxel_offset,
        compressed_segmentation_block_size=compressed_segmentation_block_size,
    )

    click.echo(json.dumps(spec, indent=2))


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
