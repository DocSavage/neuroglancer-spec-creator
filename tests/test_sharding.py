"""Tests for sharding bit parameter computation.

Ground truth: DVID repo mcns-ng-specs.json (male CNS dataset, 11 scales).
"""

import json

from ngspec.sharding import compute_sharding_params


# Ground truth from /home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json
# Each entry: (volume_size, expected_shard_bits, expected_minishard_bits, expected_preshift_bits)
MCNS_GROUND_TRUTH = [
    ((94088, 78317, 134576), 19, 6, 9),   # scale 0
    ((47044, 39159, 67288), 16, 6, 9),    # scale 1
    ((23522, 19580, 33644), 13, 6, 9),    # scale 2
    ((11761, 9790, 16822), 10, 6, 9),     # scale 3
    ((5881, 4895, 8411), 7, 6, 9),        # scale 4
    ((2941, 2448, 4206), 4, 6, 9),        # scale 5
    ((1471, 1224, 2103), 1, 6, 9),        # scale 6
    ((736, 612, 1052), 0, 4, 9),          # scale 7
    ((368, 306, 526), 0, 1, 9),           # scale 8
    ((184, 153, 263), 0, 0, 7),           # scale 9
    ((92, 77, 132), 0, 0, 4),             # scale 10
]


class TestMcnsGroundTruth:
    """Verify sharding params match DVID's known-good spec for all 11 scales."""

    def test_all_scales(self):
        for i, (size, exp_shard, exp_mini, exp_pre) in enumerate(MCNS_GROUND_TRUTH):
            result = compute_sharding_params(size)
            assert result["shard_bits"] == exp_shard, (
                f"Scale {i}: shard_bits {result['shard_bits']} != {exp_shard}"
            )
            assert result["minishard_bits"] == exp_mini, (
                f"Scale {i}: minishard_bits {result['minishard_bits']} != {exp_mini}"
            )
            assert result["preshift_bits"] == exp_pre, (
                f"Scale {i}: preshift_bits {result['preshift_bits']} != {exp_pre}"
            )

    def test_bits_sum_equals_total(self):
        """shard + minishard + preshift must always equal total_chunk_bits."""
        for size, _, _, _ in MCNS_GROUND_TRUTH:
            result = compute_sharding_params(size)
            total = result["shard_bits"] + result["minishard_bits"] + result["preshift_bits"]
            assert total == result["total_chunk_bits"], (
                f"Size {size}: {total} != {result['total_chunk_bits']}"
            )


class TestEdgeCases:
    def test_single_voxel(self):
        result = compute_sharding_params((1, 1, 1))
        assert result["total_chunk_bits"] == 0
        assert result["shard_bits"] == 0
        assert result["minishard_bits"] == 0
        assert result["preshift_bits"] == 0

    def test_single_chunk(self):
        result = compute_sharding_params((64, 64, 64))
        # grid = (1,1,1), all bits = 0
        assert result["total_chunk_bits"] == 0
        assert result["num_shards"] == 1

    def test_two_chunks_per_dim(self):
        result = compute_sharding_params((128, 128, 128))
        # grid = (2,2,2), bits per dim = 1, total = 3
        assert result["total_chunk_bits"] == 3
        assert result["preshift_bits"] == 3
        assert result["minishard_bits"] == 0
        assert result["shard_bits"] == 0

    def test_very_asymmetric(self):
        # One dimension much larger than others
        result = compute_sharding_params((64, 64, 65536))
        # grid = (1, 1, 1024), bits = 0 + 0 + 10 = 10
        assert result["total_chunk_bits"] == 10
        assert result["preshift_bits"] == 9
        assert result["minishard_bits"] == 1
        assert result["shard_bits"] == 0

    def test_exact_power_of_two_grid(self):
        """Grid sizes that are exact powers of 2 use fewer bits than bit_length."""
        # volume 256x256x256 with chunk 64 -> grid 4x4x4
        # grid 4 -> (4-1).bit_length() = 2 bits per dim, total 6
        result = compute_sharding_params((256, 256, 256))
        assert result["total_chunk_bits"] == 6
        assert result["preshift_bits"] == 6
        assert result["minishard_bits"] == 0
        assert result["shard_bits"] == 0

    def test_custom_targets(self):
        result = compute_sharding_params(
            (94088, 78317, 134576),
            target_preshift=6,
            target_minishard=6,
        )
        assert result["preshift_bits"] == 6
        assert result["minishard_bits"] == 6
        assert result["shard_bits"] == 22  # 34 - 6 - 6


class TestGroundTruthFile:
    """Load and verify against the actual JSON spec file."""

    def test_against_dvid_spec_file(self):
        spec_path = "/home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json"
        try:
            with open(spec_path) as f:
                spec = json.load(f)
        except FileNotFoundError:
            import pytest
            pytest.skip("DVID test data not available")

        for i, scale in enumerate(spec["scales"]):
            size = tuple(scale["size"])
            result = compute_sharding_params(size)
            sharding = scale["sharding"]
            assert result["shard_bits"] == sharding["shard_bits"], f"Scale {i}"
            assert result["minishard_bits"] == sharding["minishard_bits"], f"Scale {i}"
            assert result["preshift_bits"] == sharding["preshift_bits"], f"Scale {i}"
