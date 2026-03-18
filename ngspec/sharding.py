"""Bit parameter computation for neuroglancer precomputed sharding.

Computes shard_bits, minishard_bits, and preshift_bits from volume/grid
dimensions. The algorithm is derived from analyzing known-good specs in DVID.
"""

import math

from ngspec.morton import total_chunk_bits


def compute_sharding_params(
    volume_size: tuple[int, int, int],
    chunk_size: int = 64,
    target_preshift: int = 9,
    target_minishard: int = 6,
) -> dict:
    """Compute optimal shard_bits, minishard_bits, preshift_bits.

    The total bits must equal the sum of chunk coordinate bits across
    all dimensions (so the morton code covers the full coordinate space).

    Priority: preshift_bits >= minishard_bits >= shard_bits
    - preshift controls spatial locality within minishards
    - minishard controls internal shard organization
    - shard controls number of shard files (2^shard_bits)
    """
    grid_size = tuple(math.ceil(s / chunk_size) for s in volume_size)
    total = total_chunk_bits(grid_size)

    preshift = min(target_preshift, total)
    remaining = total - preshift
    minishard = min(target_minishard, remaining)
    shard = remaining - minishard

    return {
        "shard_bits": shard,
        "minishard_bits": minishard,
        "preshift_bits": preshift,
        "total_chunk_bits": total,
        "grid_size": grid_size,
        "num_shards": 2**shard,
        "num_minishards_per_shard": 2**minishard,
        "chunks_per_minishard": 2**preshift,
    }
