"""JSON spec output for neuroglancer multiscale volume.

Generates a complete neuroglancer precomputed volume info JSON with
per-scale sharding parameters computed from volume dimensions.

Supports anisotropic voxel resolutions: when the base resolution differs
across dimensions, only the finer dimensions are downsampled until they
catch up to the coarsest dimension, then all dimensions downsample together.
"""

import math

from ngspec.sharding import compute_sharding_params


def _normalize_chunk_size(
    chunk_size: int | tuple[int, int, int],
) -> tuple[int, int, int]:
    """Normalize chunk_size to a 3-tuple."""
    if isinstance(chunk_size, (list, tuple)):
        if len(chunk_size) != 3:
            raise ValueError(f"chunk_size must have 3 elements, got {len(chunk_size)}")
        return tuple(chunk_size)
    return (chunk_size, chunk_size, chunk_size)


def _dims_to_halve(
    resolution: tuple[float, float, float],
    max_initial_res: float,
) -> list[int]:
    """Determine which dimensions should be halved at this scale.

    A dimension is "fine" (needs catching up) only if doubling its resolution
    would bring it closer to the coarsest initial resolution — i.e., halving
    would improve isotropy.  This is equivalent to res[d] < (2/3) * max_res,
    which avoids treating nearly isotropic volumes (e.g., [16,16,15]) as
    anisotropic while correctly handling genuinely anisotropic cases like
    [8,8,30] or [16,16,30].

    Once all dimensions have caught up (none are finer), all are halved.
    """
    threshold = (2 / 3) * max_initial_res
    fine = [d for d in range(3) if resolution[d] < threshold]
    return fine if fine else [0, 1, 2]


def _halve_size(
    size: tuple[int, int, int],
    dims: list[int] | None = None,
) -> tuple[int, int, int]:
    """Halve specified dimensions with ceiling division.

    If dims is None, halve all dimensions (backward-compatible default).
    """
    if dims is None:
        dims = [0, 1, 2]
    return tuple(
        math.ceil(size[d] / 2) if d in dims else size[d]
        for d in range(3)
    )


def _double_resolution(
    resolution: tuple[float, float, float],
    dims: list[int] | None = None,
) -> tuple[float, float, float]:
    """Double resolution for the specified dimensions."""
    if dims is None:
        dims = [0, 1, 2]
    return tuple(
        resolution[d] * 2 if d in dims else resolution[d]
        for d in range(3)
    )


def _resolution_key(
    resolution: tuple[float, float, float],
    decimal: bool = False,
) -> str:
    """Format resolution as the scale key string.

    With decimal=False (default): '8x8x8' (integer format).
    With decimal=True: '8.0x8.0x30.0' (one decimal place).
    """
    if decimal:
        parts = []
        for r in resolution:
            if r == int(r):
                parts.append(f"{r:.1f}")
            else:
                parts.append(str(r))
        return "x".join(parts)
    return "x".join(str(int(r)) for r in resolution)


def _encoding_for_type(volume_type: str) -> str:
    """Return the default encoding for the given volume type."""
    if volume_type == "segmentation":
        return "compressed_segmentation"
    return "jpeg"


def compute_num_scales(
    volume_size: tuple[int, int, int],
    chunk_size: int | tuple[int, int, int] = 64,
    voxel_resolution: tuple[float, float, float] | None = None,
) -> int:
    """Compute the number of useful scales for a volume.

    A scale is useful when at least one grid dimension exceeds 1, meaning
    there are multiple chunks to organize into shards.  The last included
    scale is the final one with a multi-chunk grid; scales beyond that
    would have a 1x1x1 grid (0 total bits, single chunk) and are omitted.

    When voxel_resolution is provided, uses anisotropic downsampling:
    only finer dimensions are halved until they catch up to the coarsest.

    Always returns at least 1 so that even a single-chunk volume gets one
    scale entry.
    """
    cs = _normalize_chunk_size(chunk_size)
    if voxel_resolution is None:
        max_initial_res = None
    else:
        max_initial_res = max(voxel_resolution)

    n = 0
    current_size = tuple(volume_size)
    current_res = tuple(voxel_resolution) if voxel_resolution else (1.0, 1.0, 1.0)
    while True:
        grid = tuple(math.ceil(current_size[d] / cs[d]) for d in range(3))
        if all(g <= 1 for g in grid):
            break
        n += 1
        if max_initial_res is not None:
            dims = _dims_to_halve(current_res, max_initial_res)
        else:
            dims = [0, 1, 2]
        current_size = _halve_size(current_size, dims)
        current_res = _double_resolution(current_res, dims)
    return max(n, 1)


def generate_spec(
    volume_size: tuple[int, int, int],
    num_scales: int | None = None,
    voxel_resolution: tuple[float, float, float] = (8.0, 8.0, 8.0),
    data_type: str = "uint8",
    volume_type: str = "image",
    chunk_size: int | tuple[int, int, int] = 64,
    encoding: str | None = None,
    hash_type: str = "identity",
    minishard_index_encoding: str = "gzip",
    data_encoding: str | None = None,
    target_preshift: int = 9,
    target_minishard: int = 6,
    decimal_keys: bool = False,
    voxel_offset: tuple[int, int, int] | None = None,
    compressed_segmentation_block_size: tuple[int, int, int] | None = None,
) -> dict:
    """Generate a complete neuroglancer multiscale volume spec.

    Computes correct per-scale sharding parameters by halving the volume
    at each scale and recalculating bit allocations.

    When voxel_resolution is anisotropic, only the finer dimensions are
    downsampled until they catch up to the coarsest, then all dimensions
    are downsampled together.

    chunk_size can be a single int (isotropic) or a 3-tuple for
    per-dimension chunk sizes.

    If num_scales is None, automatically stops at the last scale where the
    grid has more than one chunk in at least one dimension (plus that final
    1x1x1 scale).  If num_scales is given explicitly, it is clamped to this
    maximum so that no redundant all-zero scales are emitted.
    """
    cs = _normalize_chunk_size(chunk_size)
    max_initial_res = max(voxel_resolution)

    max_scales = compute_num_scales(volume_size, chunk_size, voxel_resolution)
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
            chunk_size=cs,
            target_preshift=target_preshift,
            target_minishard=target_minishard,
        )

        sharding = {
            "@type": "neuroglancer_uint64_sharded_v1",
            "hash": hash_type,
            "minishard_bits": params["minishard_bits"],
            "minishard_index_encoding": minishard_index_encoding,
            "preshift_bits": params["preshift_bits"],
            "shard_bits": params["shard_bits"],
        }
        if data_encoding is not None:
            sharding["data_encoding"] = data_encoding

        scale = {
            "chunk_sizes": [list(cs)],
            "encoding": encoding,
            "key": _resolution_key(current_res, decimal=decimal_keys),
            "resolution": list(current_res),
            "sharding": sharding,
            "size": list(current_size),
        }
        if compressed_segmentation_block_size is not None:
            scale["compressed_segmentation_block_size"] = list(
                compressed_segmentation_block_size
            )
        if voxel_offset is not None:
            scale["voxel_offset"] = list(voxel_offset)

        scales.append(scale)

        dims = _dims_to_halve(current_res, max_initial_res)
        current_size = _halve_size(current_size, dims)
        current_res = _double_resolution(current_res, dims)

    spec = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": data_type,
        "num_channels": 1,
        "scales": scales,
        "type": volume_type,
    }

    return spec
