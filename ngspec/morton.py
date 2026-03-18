"""Compressed morton code implementation for neuroglancer precomputed sharding.

Implements the algorithm from the neuroglancer spec:
https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#compressed-morton-code
"""

import math


def bits_per_dimension(grid_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return how many bits each dimension contributes to the compressed morton code.

    For a grid_size of g, the number of bits is the count of bit positions i
    where (1 << i) < g, which equals (g - 1).bit_length() for g >= 1.
    A dimension of size 1 contributes 0 bits (only one possible coordinate: 0).
    """
    return tuple((max(g, 1) - 1).bit_length() for g in grid_size)


def total_chunk_bits(grid_size: tuple[int, int, int]) -> int:
    """Total number of bits in the compressed morton code for this grid."""
    return sum(bits_per_dimension(grid_size))


def interleave_table(grid_size: tuple[int, int, int]) -> list[tuple[int, int]]:
    """Return the full mapping of output bit positions to (dimension, input_bit).

    Returns a list where index j gives (dim, i) meaning output bit j comes from
    bit i of dimension dim. This is the interleaving pattern the animation visualizes.
    """
    table = []
    max_bits = max(grid_size).bit_length() if max(grid_size) > 0 else 0
    for i in range(max_bits):
        for dim in range(3):
            if (1 << i) < grid_size[dim]:
                table.append((dim, i))
    return table


def compressed_morton_code(coord: tuple[int, int, int],
                           grid_size: tuple[int, int, int]) -> int:
    """Compute compressed morton code for a chunk coordinate.

    Unlike standard morton interleaving which always interleaves all bits
    from all dimensions equally, compressed morton skips bit positions
    where a dimension has no more significant bits. This means:
    - Dimensions with larger extents contribute more bits
    - The interleaving pattern is asymmetric
    - Bits are never wasted on dimensions that don't need them
    """
    output = 0
    j = 0
    max_bits = max(grid_size).bit_length() if max(grid_size) > 0 else 0
    for i in range(max_bits):
        for dim in range(3):
            if (1 << i) < grid_size[dim]:
                bit = (coord[dim] >> i) & 1
                output |= bit << j
                j += 1
    return output
