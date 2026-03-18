"""JSON spec output for neuroglancer multiscale volume.

Generates a complete neuroglancer precomputed volume info JSON with
per-scale sharding parameters computed from volume dimensions.
"""

import math

from ngspec.sharding import compute_sharding_params


def _halve_size(size: tuple[int, int, int]) -> tuple[int, int, int]:
    """Halve each dimension with ceiling division."""
    return tuple(math.ceil(s / 2) for s in size)


def _resolution_key(resolution: tuple[float, float, float]) -> str:
    """Format resolution as the scale key string (e.g., '8x8x8')."""
    return "x".join(str(int(r)) for r in resolution)


def _encoding_for_type(volume_type: str) -> str:
    """Return the default encoding for the given volume type."""
    if volume_type == "segmentation":
        return "compressed_segmentation"
    return "jpeg"


def compute_num_scales(
    volume_size: tuple[int, int, int],
    chunk_size: int = 64,
) -> int:
    """Compute the number of useful scales for a volume.

    A scale is useful when at least one grid dimension exceeds 1, meaning
    there are multiple chunks to organize into shards.  The last included
    scale is the final one with a multi-chunk grid; scales beyond that
    would have a 1x1x1 grid (0 total bits, single chunk) and are omitted.

    Always returns at least 1 so that even a single-chunk volume gets one
    scale entry.
    """
    n = 0
    current_size = tuple(volume_size)
    while True:
        grid = tuple(math.ceil(s / chunk_size) for s in current_size)
        if all(g <= 1 for g in grid):
            break
        n += 1
        current_size = _halve_size(current_size)
    return max(n, 1)


def generate_spec(
    volume_size: tuple[int, int, int],
    num_scales: int | None = None,
    voxel_resolution: tuple[float, float, float] = (8.0, 8.0, 8.0),
    data_type: str = "uint8",
    volume_type: str = "image",
    chunk_size: int = 64,
    encoding: str | None = None,
    hash_type: str = "identity",
    minishard_index_encoding: str = "gzip",
    target_preshift: int = 9,
    target_minishard: int = 6,
) -> dict:
    """Generate a complete neuroglancer multiscale volume spec.

    Computes correct per-scale sharding parameters by halving the volume
    at each scale and recalculating bit allocations.

    If num_scales is None, automatically stops at the last scale where the
    grid has more than one chunk in at least one dimension (plus that final
    1x1x1 scale).  If num_scales is given explicitly, it is clamped to this
    maximum so that no redundant all-zero scales are emitted.
    """
    max_scales = compute_num_scales(volume_size, chunk_size)
    if num_scales is None:
        num_scales = max_scales
    else:
        num_scales = min(num_scales, max_scales)

    if encoding is None:
        encoding = _encoding_for_type(volume_type)

    scales = []
    current_size = tuple(volume_size)
    current_res = tuple(voxel_resolution)

    for _ in range(num_scales):
        params = compute_sharding_params(
            current_size,
            chunk_size=chunk_size,
            target_preshift=target_preshift,
            target_minishard=target_minishard,
        )

        scale = {
            "chunk_sizes": [[chunk_size, chunk_size, chunk_size]],
            "encoding": encoding,
            "key": _resolution_key(current_res),
            "resolution": list(current_res),
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": hash_type,
                "minishard_bits": params["minishard_bits"],
                "minishard_index_encoding": minishard_index_encoding,
                "preshift_bits": params["preshift_bits"],
                "shard_bits": params["shard_bits"],
            },
            "size": list(current_size),
        }
        scales.append(scale)

        current_size = _halve_size(current_size)
        current_res = tuple(r * 2 for r in current_res)

    spec = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": data_type,
        "num_channels": 1,
        "scales": scales,
        "type": volume_type,
    }

    return spec
