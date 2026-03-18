"""Tests for JSON spec generation.

Validates structure and correctness against the DVID ground truth spec.
"""

import json
import math

from ngspec.spec_generator import compute_num_scales, generate_spec


class TestSpecStructure:
    def test_top_level_fields(self):
        spec = generate_spec((1000, 1000, 1000), num_scales=1)
        assert spec["@type"] == "neuroglancer_multiscale_volume"
        assert spec["data_type"] == "uint8"
        assert spec["num_channels"] == 1
        assert spec["type"] == "image"
        assert len(spec["scales"]) == 1

    def test_scale_fields(self):
        spec = generate_spec((1000, 1000, 1000), num_scales=1)
        scale = spec["scales"][0]
        assert "chunk_sizes" in scale
        assert "encoding" in scale
        assert "key" in scale
        assert "resolution" in scale
        assert "sharding" in scale
        assert "size" in scale

    def test_sharding_fields(self):
        spec = generate_spec((1000, 1000, 1000), num_scales=1)
        sharding = spec["scales"][0]["sharding"]
        assert sharding["@type"] == "neuroglancer_uint64_sharded_v1"
        assert sharding["hash"] == "identity"
        assert "minishard_bits" in sharding
        assert "minishard_index_encoding" in sharding
        assert "preshift_bits" in sharding
        assert "shard_bits" in sharding

    def test_json_serializable(self):
        spec = generate_spec((94088, 78317, 134576), num_scales=11)
        json_str = json.dumps(spec, indent=2)
        roundtrip = json.loads(json_str)
        assert roundtrip == spec


class TestScaleProgression:
    def test_sizes_halve(self):
        spec = generate_spec((94088, 78317, 134576), num_scales=11)
        for i in range(1, len(spec["scales"])):
            prev = spec["scales"][i - 1]["size"]
            curr = spec["scales"][i]["size"]
            for d in range(3):
                assert curr[d] == math.ceil(prev[d] / 2)

    def test_resolutions_double(self):
        spec = generate_spec((1000, 1000, 1000), num_scales=5,
                             voxel_resolution=(8.0, 8.0, 8.0))
        for i, scale in enumerate(spec["scales"]):
            expected = 8.0 * (2 ** i)
            assert scale["resolution"] == [expected, expected, expected]

    def test_keys_match_resolution(self):
        spec = generate_spec((1000, 1000, 1000), num_scales=3,
                             voxel_resolution=(8.0, 8.0, 8.0))
        assert spec["scales"][0]["key"] == "8x8x8"
        assert spec["scales"][1]["key"] == "16x16x16"
        assert spec["scales"][2]["key"] == "32x32x32"

    def test_shard_bits_decrease(self):
        spec = generate_spec((94088, 78317, 134576), num_scales=11)
        shard_bits = [s["sharding"]["shard_bits"] for s in spec["scales"]]
        # Shard bits should be non-increasing
        for i in range(1, len(shard_bits)):
            assert shard_bits[i] <= shard_bits[i - 1]

    def test_bits_always_sum_correctly(self):
        """shard + mini + pre should equal total chunk bits at each scale."""
        from ngspec.sharding import compute_sharding_params
        spec = generate_spec((94088, 78317, 134576), num_scales=11)
        size = [94088, 78317, 134576]
        for scale in spec["scales"]:
            params = compute_sharding_params(tuple(scale["size"]))
            sh = scale["sharding"]
            total = sh["shard_bits"] + sh["minishard_bits"] + sh["preshift_bits"]
            assert total == params["total_chunk_bits"]
            size = [math.ceil(s / 2) for s in size]


class TestDvidGroundTruth:
    """Validate against the DVID mcns-ng-specs.json."""

    def test_matches_dvid_spec(self):
        spec = generate_spec(
            (94088, 78317, 134576),
            num_scales=11,
            voxel_resolution=(8.0, 8.0, 8.0),
            data_type="uint8",
            volume_type="image",
            encoding="jpeg",
        )

        spec_path = "/home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json"
        try:
            with open(spec_path) as f:
                expected = json.load(f)
        except FileNotFoundError:
            import pytest
            pytest.skip("DVID test data not available")

        assert len(spec["scales"]) == len(expected["scales"])

        for i, (gen, exp) in enumerate(zip(spec["scales"], expected["scales"])):
            assert gen["size"] == exp["size"], f"Scale {i} size mismatch"
            assert gen["sharding"]["shard_bits"] == exp["sharding"]["shard_bits"], (
                f"Scale {i} shard_bits"
            )
            assert gen["sharding"]["minishard_bits"] == exp["sharding"]["minishard_bits"], (
                f"Scale {i} minishard_bits"
            )
            assert gen["sharding"]["preshift_bits"] == exp["sharding"]["preshift_bits"], (
                f"Scale {i} preshift_bits"
            )
            assert gen["key"] == exp["key"], f"Scale {i} key"
            assert gen["resolution"] == exp["resolution"], f"Scale {i} resolution"


class TestAutoScales:
    def test_mcns_auto_gives_12(self):
        """The mcns volume auto-computes to 12 scales (last multi-chunk grid is 1x1x2)."""
        assert compute_num_scales((94088, 78317, 134576)) == 12

    def test_clamps_excessive_scales(self):
        spec = generate_spec((94088, 78317, 134576), num_scales=20)
        assert len(spec["scales"]) == 12

    def test_respects_fewer_scales(self):
        spec = generate_spec((94088, 78317, 134576), num_scales=5)
        assert len(spec["scales"]) == 5

    def test_single_chunk_volume(self):
        """A volume that fits in one chunk should produce exactly 1 scale."""
        assert compute_num_scales((64, 64, 64)) == 1
        spec = generate_spec((64, 64, 64))
        assert len(spec["scales"]) == 1

    def test_small_volume(self):
        """128^3 -> grid 2x2x2 at scale 0 only (scale 1 would be 1x1x1)."""
        assert compute_num_scales((128, 128, 128)) == 1

    def test_last_scale_has_multi_chunk_grid(self):
        """The final auto-computed scale should still have at least one grid dim > 1."""
        spec = generate_spec((94088, 78317, 134576))
        last = spec["scales"][-1]
        grid = [math.ceil(s / 64) for s in last["size"]]
        assert any(g > 1 for g in grid)

    def test_no_zero_bit_scales(self):
        """Auto-computed specs should never include a 0-total-bits scale."""
        spec = generate_spec((94088, 78317, 134576))
        for i, scale in enumerate(spec["scales"]):
            sh = scale["sharding"]
            total = sh["shard_bits"] + sh["minishard_bits"] + sh["preshift_bits"]
            assert total > 0, f"Scale {i} has 0 total bits"

    def test_dvid_ground_truth_subset(self):
        """Auto scales should be a superset of DVID's 11 — first 11 must match."""
        spec = generate_spec(
            (94088, 78317, 134576),
            encoding="jpeg",
        )
        assert len(spec["scales"]) >= 11
        spec_path = "/home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json"
        try:
            with open(spec_path) as f:
                expected = json.load(f)
        except FileNotFoundError:
            import pytest
            pytest.skip("DVID test data not available")
        for i in range(11):
            gen_sh = spec["scales"][i]["sharding"]
            exp_sh = expected["scales"][i]["sharding"]
            for field in ("shard_bits", "minishard_bits", "preshift_bits"):
                assert gen_sh[field] == exp_sh[field], f"Scale {i} {field}"


class TestVolumeTypes:
    def test_segmentation_encoding(self):
        spec = generate_spec((1000, 1000, 1000), num_scales=1,
                             volume_type="segmentation", data_type="uint64")
        assert spec["type"] == "segmentation"
        assert spec["scales"][0]["encoding"] == "compressed_segmentation"

    def test_image_encoding(self):
        spec = generate_spec((1000, 1000, 1000), num_scales=1,
                             volume_type="image")
        assert spec["scales"][0]["encoding"] == "jpeg"

    def test_custom_encoding(self):
        spec = generate_spec((1000, 1000, 1000), num_scales=1,
                             encoding="raw")
        assert spec["scales"][0]["encoding"] == "raw"
