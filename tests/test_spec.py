"""Tests for JSON spec generation.

Validates structure and correctness against tensorstore invariants.
The DVID mcns-ng-specs.json is used for sharding parameter comparison only;
tensorstore (~/tensorstore) is the authoritative reference for the
neuroglancer precomputed format.
"""

import json
import math

from ngspec.morton import bits_per_dimension, compressed_morton_code, total_chunk_bits
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


class TestDvidShardingComparison:
    """Compare sharding parameters against DVID mcns-ng-specs.json.

    NOTE: The DVID file uses uint8/jpeg/image which is incorrect for
    segmentation data.  Only sharding params, sizes, keys, and resolutions
    are compared here.  Tensorstore is the authoritative reference for the
    neuroglancer precomputed format — see TestTensorstoreInvariants.
    """

    def test_sharding_params_match_dvid(self):
        spec_path = "/home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json"
        try:
            with open(spec_path) as f:
                expected = json.load(f)
        except FileNotFoundError:
            import pytest
            pytest.skip("DVID test data not available")

        spec = generate_spec(
            (94088, 78317, 134576),
            num_scales=len(expected["scales"]),
            voxel_resolution=(8.0, 8.0, 8.0),
        )

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


class TestTensorstoreInvariants:
    """Validate against tensorstore's neuroglancer precomputed format rules.

    Reference: tensorstore/driver/neuroglancer_precomputed/metadata.cc
    - GetCompressedZIndexBits: bit_width(ceil(shape/chunk) - 1)
    - GetShardChunkHierarchy: total_z_index_bits <= shard + minishard + preshift
    - GetChunksPerVolumeShardFunction: all chunks map to valid shard IDs
    """

    def test_bits_per_dim_matches_tensorstore(self):
        """Our bits_per_dimension matches tensorstore's GetCompressedZIndexBits.

        tensorstore: bit_width(max(0, ceil(shape/chunk) - 1))
        ours: (max(g, 1) - 1).bit_length() where g = ceil(shape/chunk)
        """
        cases = [
            # (grid_size, expected_bits) — verified against tensorstore
            ((1, 1, 1), (0, 0, 0)),
            ((2, 2, 2), (1, 1, 1)),
            ((3, 3, 5), (2, 2, 3)),
            ((4, 4, 4), (2, 2, 2)),  # power-of-2: bit_width(3)=2, not 3
            ((8, 8, 8), (3, 3, 3)),  # power-of-2: bit_width(7)=3, not 4
            ((1471, 1224, 2103), (11, 11, 12)),
        ]
        for grid, expected in cases:
            assert bits_per_dimension(grid) == expected, f"grid={grid}"

    def test_total_bits_cover_morton_space(self):
        """shard + minishard + preshift >= total_z_index_bits at every scale.

        This is tensorstore's validation at metadata.cc:1483-1488.
        Our allocator sets them equal, satisfying this invariant.
        """
        spec = generate_spec(
            (94088, 78317, 134576),
            data_type="uint64",
            volume_type="segmentation",
        )
        for i, scale in enumerate(spec["scales"]):
            size = tuple(scale["size"])
            grid = tuple(math.ceil(s / 64) for s in size)
            z_index_bits = sum(bits_per_dimension(grid))
            sh = scale["sharding"]
            allocated = sh["shard_bits"] + sh["minishard_bits"] + sh["preshift_bits"]
            assert allocated >= z_index_bits, (
                f"Scale {i}: allocated {allocated} < z_index_bits {z_index_bits}"
            )

    def test_all_morton_codes_fit_in_single_shard_when_shard_bits_zero(self):
        """When shard_bits=0, every chunk's morton code must map to shard 0.

        Simulates tensorstore's shard ID derivation (identity hash):
          hash_input = morton_code >> preshift_bits
          shard = (hash_input >> minishard_bits) & ((1 << shard_bits) - 1)
        """
        spec = generate_spec(
            (94088, 78317, 134576),
            data_type="uint64",
            volume_type="segmentation",
        )
        for i, scale in enumerate(spec["scales"]):
            sh = scale["sharding"]
            if sh["shard_bits"] != 0:
                continue
            size = tuple(scale["size"])
            grid = tuple(math.ceil(s / 64) for s in size)
            # Check every chunk coordinate at this scale
            for x in range(grid[0]):
                for y in range(grid[1]):
                    for z in range(grid[2]):
                        morton = compressed_morton_code((x, y, z), grid)
                        hash_input = morton >> sh["preshift_bits"]
                        shard_mask = (1 << sh["shard_bits"]) - 1
                        shard_id = (hash_input >> sh["minishard_bits"]) & shard_mask
                        assert shard_id == 0, (
                            f"Scale {i} chunk ({x},{y},{z}): morton={morton}, "
                            f"shard_id={shard_id} != 0 with shard_bits=0"
                        )

    def test_max_morton_fits_allocated_bits(self):
        """The maximum morton code at each scale fits within allocated bits.

        Ensures 2^total_allocated_bits > max_morton for all chunk coordinates.
        """
        spec = generate_spec(
            (94088, 78317, 134576),
            data_type="uint64",
            volume_type="segmentation",
        )
        for i, scale in enumerate(spec["scales"]):
            size = tuple(scale["size"])
            grid = tuple(math.ceil(s / 64) for s in size)
            sh = scale["sharding"]
            allocated = sh["shard_bits"] + sh["minishard_bits"] + sh["preshift_bits"]
            # Max coordinate is (grid[d]-1) for each dimension
            max_coord = tuple(g - 1 for g in grid)
            max_morton = compressed_morton_code(max_coord, grid)
            assert max_morton < (1 << allocated), (
                f"Scale {i}: max_morton {max_morton} >= 2^{allocated}"
            )


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

    def test_dvid_sharding_params_subset(self):
        """Auto scales should be a superset of DVID's 11 — first 11 sharding params must match."""
        spec = generate_spec((94088, 78317, 134576))
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


class TestAnisotropicDownsampling:
    """Validate anisotropic downsampling for volumes with non-uniform resolution.

    When voxel resolution differs across dimensions, only the finer dimensions
    are downsampled until they catch up to the coarsest, then all dimensions
    downsample together.
    """

    def test_fish2_anisotropic_sizes(self):
        """Fish2 EM grayscale: Z is not halved until X,Y catch up to 30nm."""
        spec = generate_spec(
            (204800, 114688, 10254),
            voxel_resolution=(8.0, 8.0, 30.0),
        )
        # Scales 0-2: only X,Y halve, Z stays at 10254
        for i in range(3):
            assert spec["scales"][i]["size"][2] == 10254, f"Scale {i}: Z should not halve"
        # Scale 3: all halve — Z becomes ceil(10254/2) = 5127
        assert spec["scales"][3]["size"][2] == 5127

    def test_fish2_anisotropic_resolutions(self):
        """Resolution only doubles for halved dimensions."""
        spec = generate_spec(
            (204800, 114688, 10254),
            voxel_resolution=(8.0, 8.0, 30.0),
        )
        assert spec["scales"][0]["resolution"] == [8.0, 8.0, 30.0]
        assert spec["scales"][1]["resolution"] == [16.0, 16.0, 30.0]
        assert spec["scales"][2]["resolution"] == [32.0, 32.0, 30.0]
        assert spec["scales"][3]["resolution"] == [64.0, 64.0, 60.0]
        assert spec["scales"][4]["resolution"] == [128.0, 128.0, 120.0]

    def test_fish2_sharding_matches_em_info_from_scale3(self):
        """With 64^3 chunks, scales 3+ match the EM info.json sharding exactly.

        Scales 0-2 of the ground truth EM info use different chunk sizes
        ([128,128,32] and [128,128,64]) so their sharding params differ.
        """
        # Ground truth from the EM info.json, scales 3-10 (all use 64^3 chunks)
        expected_sharding = [
            # (size, shard_bits, minishard_bits, preshift_bits)
            ([25600, 14336, 5127], 9, 6, 9),
            ([12800, 7168, 2564], 6, 6, 9),
            ([6400, 3584, 1282], 3, 6, 9),
            ([3200, 1792, 641], 0, 6, 9),
            ([1600, 896, 321], 0, 3, 9),
            ([800, 448, 161], 0, 0, 9),
            ([400, 224, 81], 0, 0, 6),
            ([200, 112, 41], 0, 0, 3),
        ]
        spec = generate_spec(
            (204800, 114688, 10254),
            voxel_resolution=(8.0, 8.0, 30.0),
        )
        for j, (exp_size, exp_shard, exp_mini, exp_pre) in enumerate(expected_sharding):
            i = j + 3  # offset to scale 3
            scale = spec["scales"][i]
            sh = scale["sharding"]
            assert scale["size"] == exp_size, f"Scale {i} size"
            assert sh["shard_bits"] == exp_shard, f"Scale {i} shard_bits"
            assert sh["minishard_bits"] == exp_mini, f"Scale {i} minishard_bits"
            assert sh["preshift_bits"] == exp_pre, f"Scale {i} preshift_bits"

    def test_isotropic_unchanged(self):
        """Isotropic resolution should halve all dimensions every scale (unchanged behavior)."""
        spec = generate_spec(
            (94088, 78317, 134576),
            num_scales=3,
            voxel_resolution=(8.0, 8.0, 8.0),
        )
        for i in range(1, 3):
            prev = spec["scales"][i - 1]["size"]
            curr = spec["scales"][i]["size"]
            for d in range(3):
                assert curr[d] == math.ceil(prev[d] / 2)

    def test_anisotropic_keys(self):
        """Keys reflect the anisotropic resolution."""
        spec = generate_spec(
            (204800, 114688, 10254),
            voxel_resolution=(8.0, 8.0, 30.0),
        )
        assert spec["scales"][0]["key"] == "8x8x30"
        assert spec["scales"][1]["key"] == "16x16x30"
        assert spec["scales"][2]["key"] == "32x32x30"
        assert spec["scales"][3]["key"] == "64x64x60"

    def test_decimal_keys(self):
        """decimal_keys=True produces keys like '8.0x8.0x30.0'."""
        spec = generate_spec(
            (204800, 114688, 10254),
            num_scales=2,
            voxel_resolution=(8.0, 8.0, 30.0),
            decimal_keys=True,
        )
        assert spec["scales"][0]["key"] == "8.0x8.0x30.0"
        assert spec["scales"][1]["key"] == "16.0x16.0x30.0"

    def test_per_dim_chunk_size(self):
        """Per-dimension chunk sizes produce correct grid and sharding."""
        spec = generate_spec(
            (204800, 114688, 10254),
            num_scales=1,
            voxel_resolution=(8.0, 8.0, 30.0),
            chunk_size=(128, 128, 32),
        )
        scale = spec["scales"][0]
        assert scale["chunk_sizes"] == [[128, 128, 32]]
        # Grid: ceil(204800/128)=1600, ceil(114688/128)=896, ceil(10254/32)=321
        sh = scale["sharding"]
        # bits: (1600-1).bit_length()=11, (896-1).bit_length()=10, (321-1).bit_length()=9
        # total=30, pre=9, mini=6, shard=15
        assert sh["shard_bits"] == 15
        assert sh["minishard_bits"] == 6
        assert sh["preshift_bits"] == 9

    def test_bits_always_sum_with_anisotropic(self):
        """shard + mini + pre should equal total chunk bits for anisotropic volumes."""
        from ngspec.sharding import compute_sharding_params
        spec = generate_spec(
            (204800, 114688, 10254),
            voxel_resolution=(8.0, 8.0, 30.0),
        )
        for i, scale in enumerate(spec["scales"]):
            params = compute_sharding_params(tuple(scale["size"]))
            sh = scale["sharding"]
            total = sh["shard_bits"] + sh["minishard_bits"] + sh["preshift_bits"]
            assert total == params["total_chunk_bits"], f"Scale {i} bits don't sum"


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
