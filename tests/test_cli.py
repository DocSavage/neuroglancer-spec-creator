"""End-to-end tests for the CLI generate command.

Runs the CLI and validates the JSON output (stdout) and summary table (stderr).
"""

import json

from click.testing import CliRunner

from cli import cli

# Ground truth: expected sharding params for each scale, derived from
# /home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json
# Volume: 94088 x 78317 x 134576 (image, uint8, chunk_size=64, 11 scales)
MCNS_EXPECTED = [
    # (size, key, resolution, shard_bits, minishard_bits, preshift_bits)
    ([94088, 78317, 134576], "8x8x8",       [8.0, 8.0, 8.0],         19, 6, 9),
    ([47044, 39159, 67288],  "16x16x16",     [16.0, 16.0, 16.0],      16, 6, 9),
    ([23522, 19580, 33644],  "32x32x32",     [32.0, 32.0, 32.0],      13, 6, 9),
    ([11761, 9790, 16822],   "64x64x64",     [64.0, 64.0, 64.0],      10, 6, 9),
    ([5881, 4895, 8411],     "128x128x128",  [128.0, 128.0, 128.0],    7, 6, 9),
    ([2941, 2448, 4206],     "256x256x256",  [256.0, 256.0, 256.0],    4, 6, 9),
    ([1471, 1224, 2103],     "512x512x512",  [512.0, 512.0, 512.0],    1, 6, 9),
    ([736, 612, 1052],       "1024x1024x1024", [1024.0, 1024.0, 1024.0], 0, 4, 9),
    ([368, 306, 526],        "2048x2048x2048", [2048.0, 2048.0, 2048.0], 0, 1, 9),
    ([184, 153, 263],        "4096x4096x4096", [4096.0, 4096.0, 4096.0], 0, 0, 7),
    ([92, 77, 132],          "8192x8192x8192", [8192.0, 8192.0, 8192.0], 0, 0, 4),
]


def run_generate(args):
    """Run the generate command and return (spec, full_output).

    JSON goes to stdout, summary table to stderr.  CliRunner mixes both
    into result.output, so we extract the JSON by finding the opening brace.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["generate"] + args)
    assert result.exit_code == 0, result.output
    output = result.output
    json_start = output.index("{")
    spec = json.loads(output[json_start:])
    return spec, output[:json_start]


class TestCliGenerate:
    def test_generates_valid_json(self):
        spec, _ = run_generate([
            "--em", "--size", "94088,78317,134576", "--scales", "11",
        ])
        assert spec["@type"] == "neuroglancer_multiscale_volume"
        assert len(spec["scales"]) == 11

    def test_mcns_ground_truth(self):
        """Run the CLI with DVID mcns volume params and check every scale."""
        spec, _ = run_generate([
            "--em", "--size", "94088,78317,134576", "--scales", "11",
        ])
        assert len(spec["scales"]) == len(MCNS_EXPECTED)

        for i, (exp_size, exp_key, exp_res, exp_shard, exp_mini, exp_pre) in enumerate(MCNS_EXPECTED):
            scale = spec["scales"][i]
            sh = scale["sharding"]
            assert scale["size"] == exp_size, f"Scale {i}: size"
            assert scale["key"] == exp_key, f"Scale {i}: key"
            assert scale["resolution"] == exp_res, f"Scale {i}: resolution"
            assert sh["shard_bits"] == exp_shard, f"Scale {i}: shard_bits"
            assert sh["minishard_bits"] == exp_mini, f"Scale {i}: minishard_bits"
            assert sh["preshift_bits"] == exp_pre, f"Scale {i}: preshift_bits"

    def test_mcns_against_spec_file(self):
        """Cross-check CLI output against the actual DVID spec file on disk."""
        spec_path = "/home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json"
        try:
            with open(spec_path) as f:
                expected = json.load(f)
        except FileNotFoundError:
            import pytest
            pytest.skip("DVID test data not available")

        spec, _ = run_generate([
            "--em", "--size", "94088,78317,134576", "--scales", "11",
        ])

        for i, (gen, exp) in enumerate(zip(spec["scales"], expected["scales"])):
            assert gen["size"] == exp["size"], f"Scale {i} size"
            assert gen["key"] == exp["key"], f"Scale {i} key"
            assert gen["resolution"] == exp["resolution"], f"Scale {i} resolution"
            for field in ("shard_bits", "minishard_bits", "preshift_bits"):
                assert gen["sharding"][field] == exp["sharding"][field], (
                    f"Scale {i} {field}: {gen['sharding'][field]} != {exp['sharding'][field]}"
                )

    def test_table_output_shows_all_scales(self):
        """The stderr table should have a line for each scale."""
        _, stderr = run_generate([
            "--size", "94088,78317,134576", "--scales", "11",
        ])
        for i in range(11):
            assert f"\n{i}" in stderr or stderr.startswith(f"{i}"), (
                f"Scale {i} not found in table output"
            )

    def test_auto_scales_default(self):
        """Omitting --scales should auto-compute to 12 for the mcns volume."""
        spec, _ = run_generate(["--size", "94088,78317,134576"])
        assert len(spec["scales"]) == 12

    def test_auto_scales_includes_ground_truth(self):
        """Auto-computed scales should include all 11 DVID ground truth scales."""
        spec, _ = run_generate([
            "--em", "--size", "94088,78317,134576",
        ])
        # First 11 scales must match DVID ground truth exactly
        for i, (exp_size, exp_key, exp_res, exp_shard, exp_mini, exp_pre) in enumerate(MCNS_EXPECTED):
            sh = spec["scales"][i]["sharding"]
            assert sh["shard_bits"] == exp_shard, f"Scale {i}: shard_bits"
            assert sh["minishard_bits"] == exp_mini, f"Scale {i}: minishard_bits"
            assert sh["preshift_bits"] == exp_pre, f"Scale {i}: preshift_bits"

    def test_excessive_scales_clamped(self):
        """Requesting more scales than useful should clamp and warn."""
        spec, stderr = run_generate([
            "--size", "94088,78317,134576", "--scales", "14",
        ])
        assert "clamping" in stderr.lower()
        assert len(spec["scales"]) == 12


class TestCliFish2Anisotropic:
    """Test CLI with fish2 anisotropic EM grayscale data."""

    def test_anisotropic_resolution(self):
        """Fish2 EM with --resolution 8,8,30 should produce correct aniso scales."""
        spec, _ = run_generate([
            "--em", "--size", "204800,114688,10254", "--resolution", "8,8,30",
        ])
        # Z should not halve for first 3 scales
        for i in range(3):
            assert spec["scales"][i]["size"][2] == 10254, f"Scale {i}: Z should stay"
            assert spec["scales"][i]["resolution"][2] == 30.0, f"Scale {i}: Z res should stay"
        # Scale 3: all halve
        assert spec["scales"][3]["size"] == [25600, 14336, 5127]
        assert spec["scales"][3]["resolution"] == [64.0, 64.0, 60.0]

    def test_per_dim_chunk_size(self):
        """Fish2 with per-dimension chunk size like the EM info."""
        spec, _ = run_generate([
            "--em", "--size", "204800,114688,10254",
            "--resolution", "8,8,30", "--chunk-size", "128,128,32", "--scales", "1",
        ])
        assert spec["scales"][0]["chunk_sizes"] == [[128, 128, 32]]
        sh = spec["scales"][0]["sharding"]
        assert sh["shard_bits"] == 15
        assert sh["minishard_bits"] == 6
        assert sh["preshift_bits"] == 9

    def test_decimal_keys_flag(self):
        """--decimal-keys should produce keys like 8.0x8.0x30.0."""
        spec, _ = run_generate([
            "--em", "--size", "204800,114688,10254",
            "--resolution", "8,8,30", "--scales", "2", "--decimal-keys",
        ])
        assert spec["scales"][0]["key"] == "8.0x8.0x30.0"
        assert spec["scales"][1]["key"] == "16.0x16.0x30.0"


class TestCliPresets:
    """Test --seg and --em preset flags."""

    def test_seg_default(self):
        """Default (--seg) should produce segmentation spec with all required fields."""
        spec, _ = run_generate([
            "--size", "102400,57344,20508", "--resolution", "16,16,15", "--scales", "2",
        ])
        assert spec["data_type"] == "uint64"
        assert spec["type"] == "segmentation"
        for scale in spec["scales"]:
            assert scale["encoding"] == "compressed_segmentation"
            assert scale["compressed_segmentation_block_size"] == [8, 8, 8]
            assert scale["voxel_offset"] == [0, 0, 0]
            assert scale["sharding"]["data_encoding"] == "gzip"

    def test_em_flag(self):
        """--em should produce EM spec without seg-specific fields."""
        spec, _ = run_generate([
            "--em", "--size", "204800,114688,10254",
            "--resolution", "8,8,30", "--scales", "2",
        ])
        assert spec["data_type"] == "uint8"
        assert spec["type"] == "image"
        for scale in spec["scales"]:
            assert scale["encoding"] == "jpeg"
            assert "compressed_segmentation_block_size" not in scale
            assert "voxel_offset" not in scale
            assert "data_encoding" not in scale["sharding"]

    def test_seg_explicit(self):
        """Explicit --seg should behave same as default."""
        spec, _ = run_generate([
            "--seg", "--size", "102400,57344,20508",
            "--resolution", "16,16,15", "--scales", "1",
        ])
        assert spec["data_type"] == "uint64"
        assert spec["type"] == "segmentation"
