"""Tests for compressed morton code implementation.

Verified against the neuroglancer reference spec and the Go implementation
in DVID's export.go.
"""

import math

from ngspec.morton import (
    bits_per_dimension,
    compressed_morton_code,
    interleave_table,
    total_chunk_bits,
)


class TestBitsPerDimension:
    def test_symmetric_grid(self):
        # 8x8x8 grid: each dim needs 3 bits (values 0-7)
        assert bits_per_dimension((8, 8, 8)) == (3, 3, 3)

    def test_asymmetric_grid(self):
        # DVID mcns scale 0: grid (1471, 1224, 2103)
        assert bits_per_dimension((1471, 1224, 2103)) == (11, 11, 12)

    def test_power_of_two_grid(self):
        # grid_size=2 means values 0-1, needs 1 bit
        assert bits_per_dimension((2, 2, 2)) == (1, 1, 1)
        # grid_size=4 means values 0-3, needs 2 bits
        assert bits_per_dimension((4, 4, 4)) == (2, 2, 2)

    def test_single_chunk(self):
        # grid_size=1 means only value 0, needs 0 bits
        assert bits_per_dimension((1, 1, 1)) == (0, 0, 0)

    def test_one_dimension_much_larger(self):
        assert bits_per_dimension((2, 2, 1024)) == (1, 1, 10)

    def test_mixed_sizes(self):
        # grid (3, 5, 17)
        assert bits_per_dimension((3, 5, 17)) == (2, 3, 5)


class TestTotalChunkBits:
    def test_mcns_scale0(self):
        # 1471x1224x2103 -> 11+11+12 = 34
        assert total_chunk_bits((1471, 1224, 2103)) == 34

    def test_small_grid(self):
        # (2, 2, 3) -> 1+1+2 = 4
        assert total_chunk_bits((2, 2, 3)) == 4

    def test_single_chunk(self):
        assert total_chunk_bits((1, 1, 1)) == 0


class TestInterleaveTable:
    def test_symmetric_2x2x2(self):
        table = interleave_table((2, 2, 2))
        # grid=2: only bit 0 from each dim (since 1<<0=1 < 2)
        assert table == [(0, 0), (1, 0), (2, 0)]

    def test_asymmetric_2x2x3(self):
        table = interleave_table((2, 2, 3))
        # bit 0: all three dims (1<2, 1<2, 1<3)
        # bit 1: only dim 2 (2<2 false, 2<2 false, 2<3 true)
        assert table == [(0, 0), (1, 0), (2, 0), (2, 1)]

    def test_length_matches_total_bits(self):
        grid = (1471, 1224, 2103)
        table = interleave_table(grid)
        assert len(table) == total_chunk_bits(grid)

    def test_each_dim_contributes_correct_bits(self):
        grid = (3, 5, 17)
        table = interleave_table(grid)
        per_dim = bits_per_dimension(grid)
        for dim in range(3):
            count = sum(1 for d, _ in table if d == dim)
            assert count == per_dim[dim]


class TestCompressedMortonCode:
    def test_origin(self):
        assert compressed_morton_code((0, 0, 0), (8, 8, 8)) == 0

    def test_unit_x(self):
        # (1,0,0) in 8x8x8: bit 0 of dim 0 goes to output bit 0
        assert compressed_morton_code((1, 0, 0), (8, 8, 8)) == 1

    def test_unit_y(self):
        # (0,1,0) in 8x8x8: bit 0 of dim 1 goes to output bit 1
        assert compressed_morton_code((0, 1, 0), (8, 8, 8)) == 2

    def test_unit_z(self):
        # (0,0,1) in 8x8x8: bit 0 of dim 2 goes to output bit 2
        assert compressed_morton_code((0, 0, 1), (8, 8, 8)) == 4

    def test_one_one_one(self):
        # (1,1,1) in 8x8x8: bits at positions 0,1,2 all set = 0b111 = 7
        assert compressed_morton_code((1, 1, 1), (8, 8, 8)) == 7

    def test_symmetric_small(self):
        # In a symmetric 4x4x4 grid, compressed morton = standard morton (Z-order)
        # (2,1,0) = x=10, y=01, z=00
        # Interleave: bit0: x0=0, y0=1, z0=0 -> 010
        #             bit1: x1=1, y1=0, z1=0 -> 001
        # Output: 001_010 = 0b001010 = 10
        assert compressed_morton_code((2, 1, 0), (4, 4, 4)) == 10

    def test_asymmetric_z_continues(self):
        # Grid (2, 2, 4): X and Y have 1 bit, Z has 2 bits
        # (1, 1, 3): interleave bit0: X0=1, Y0=1, Z0=1 -> output bits 0,1,2 = 111
        #            bit1: only Z (1<<1=2 < 4, but 2<2=false for X,Y)
        #            Z1=1 -> output bit 3 = 1
        # Total: 0b1111 = 15
        assert compressed_morton_code((1, 1, 3), (2, 2, 4)) == 15

    def test_single_chunk_grid(self):
        # All coordinates must be 0, morton code is 0
        assert compressed_morton_code((0, 0, 0), (1, 1, 1)) == 0

    def test_matches_go_reference_simple(self):
        """Verify against the Go mortonCode implementation logic.

        For grid (4, 4, 8), coord (3, 2, 5):
        X=3=0b11, Y=2=0b10, Z=5=0b101
        Bits per dim: X=2, Y=2, Z=3
        Interleave:
          i=0: X0=1(j=0), Y0=0(j=1), Z0=1(j=2)
          i=1: X1=1(j=3), Y1=1(j=4), Z1=0(j=5)
          i=2: Z2=1(j=6)
        Output = 1 + 0 + 4 + 8 + 16 + 0 + 64 = 93
        """
        assert compressed_morton_code((3, 2, 5), (4, 4, 8)) == 0b1011101
        assert compressed_morton_code((3, 2, 5), (4, 4, 8)) == 93

    def test_mcns_max_coord(self):
        """Verify that the maximum coordinate in the DVID mcns grid produces
        a morton code that fits in the expected number of bits."""
        grid = (1471, 1224, 2103)
        max_coord = (1470, 1223, 2102)
        code = compressed_morton_code(max_coord, grid)
        assert code.bit_length() <= total_chunk_bits(grid)

    def test_injective_small_grid(self):
        """All coordinates in a small grid produce unique morton codes."""
        grid = (3, 4, 5)
        codes = set()
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    codes.add(compressed_morton_code((x, y, z), grid))
        assert len(codes) == grid[0] * grid[1] * grid[2]
