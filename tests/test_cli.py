"""End-to-end tests for the CLI generate command.

Runs the CLI and validates the output JSON against DVID ground truth.
"""

import json
import os
import tempfile

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


class TestCliGenerate:
    def test_generates_valid_json(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            outpath = f.name
        try:
            result = runner.invoke(cli, [
                "generate",
                "--size", "94088,78317,134576",
                "--scales", "11",
                "--output", outpath,
            ])
            assert result.exit_code == 0, result.output
            with open(outpath) as f:
                spec = json.load(f)
            assert spec["@type"] == "neuroglancer_multiscale_volume"
            assert len(spec["scales"]) == 11
        finally:
            os.unlink(outpath)

    def test_mcns_ground_truth(self):
        """Run the CLI with DVID mcns volume params and check every scale."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            outpath = f.name
        try:
            result = runner.invoke(cli, [
                "generate",
                "--size", "94088,78317,134576",
                "--scales", "11",
                "--data-type", "uint8",
                "--volume-type", "image",
                "--encoding", "jpeg",
                "--output", outpath,
            ])
            assert result.exit_code == 0, result.output

            with open(outpath) as f:
                spec = json.load(f)

            assert len(spec["scales"]) == len(MCNS_EXPECTED)

            for i, (exp_size, exp_key, exp_res, exp_shard, exp_mini, exp_pre) in enumerate(MCNS_EXPECTED):
                scale = spec["scales"][i]
                sh = scale["sharding"]
                assert scale["size"] == exp_size, f"Scale {i}: size {scale['size']} != {exp_size}"
                assert scale["key"] == exp_key, f"Scale {i}: key {scale['key']} != {exp_key}"
                assert scale["resolution"] == exp_res, f"Scale {i}: resolution mismatch"
                assert sh["shard_bits"] == exp_shard, f"Scale {i}: shard_bits {sh['shard_bits']} != {exp_shard}"
                assert sh["minishard_bits"] == exp_mini, f"Scale {i}: minishard_bits {sh['minishard_bits']} != {exp_mini}"
                assert sh["preshift_bits"] == exp_pre, f"Scale {i}: preshift_bits {sh['preshift_bits']} != {exp_pre}"
        finally:
            os.unlink(outpath)

    def test_mcns_against_spec_file(self):
        """Cross-check CLI output against the actual DVID spec file on disk."""
        spec_path = "/home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json"
        try:
            with open(spec_path) as f:
                expected = json.load(f)
        except FileNotFoundError:
            import pytest
            pytest.skip("DVID test data not available")

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            outpath = f.name
        try:
            result = runner.invoke(cli, [
                "generate",
                "--size", "94088,78317,134576",
                "--scales", "11",
                "--data-type", "uint8",
                "--volume-type", "image",
                "--encoding", "jpeg",
                "--output", outpath,
            ])
            assert result.exit_code == 0, result.output

            with open(outpath) as f:
                generated = json.load(f)

            for i, (gen, exp) in enumerate(zip(generated["scales"], expected["scales"])):
                assert gen["size"] == exp["size"], f"Scale {i} size"
                assert gen["key"] == exp["key"], f"Scale {i} key"
                assert gen["resolution"] == exp["resolution"], f"Scale {i} resolution"
                for field in ("shard_bits", "minishard_bits", "preshift_bits"):
                    assert gen["sharding"][field] == exp["sharding"][field], (
                        f"Scale {i} {field}: {gen['sharding'][field]} != {exp['sharding'][field]}"
                    )
        finally:
            os.unlink(outpath)

    def test_table_output_shows_all_scales(self):
        """The stdout table should have a line for each scale."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            outpath = f.name
        try:
            result = runner.invoke(cli, [
                "generate",
                "--size", "94088,78317,134576",
                "--scales", "11",
                "--output", outpath,
            ])
            assert result.exit_code == 0
            # Each scale prints a line starting with the scale index
            for i in range(11):
                assert f"\n{i}" in result.output or result.output.startswith(f"{i}"), (
                    f"Scale {i} not found in table output"
                )
        finally:
            os.unlink(outpath)

    def test_auto_scales_default(self):
        """Omitting --scales should auto-compute to 12 for the mcns volume."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            outpath = f.name
        try:
            result = runner.invoke(cli, [
                "generate",
                "--size", "94088,78317,134576",
                "--output", outpath,
            ])
            assert result.exit_code == 0, result.output
            with open(outpath) as f:
                spec = json.load(f)
            assert len(spec["scales"]) == 12
        finally:
            os.unlink(outpath)

    def test_auto_scales_includes_ground_truth(self):
        """Auto-computed scales should include all 11 DVID ground truth scales."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            outpath = f.name
        try:
            result = runner.invoke(cli, [
                "generate",
                "--size", "94088,78317,134576",
                "--encoding", "jpeg",
                "--output", outpath,
            ])
            assert result.exit_code == 0, result.output
            with open(outpath) as f:
                spec = json.load(f)

            # First 11 scales must match DVID ground truth exactly
            for i, (exp_size, exp_key, exp_res, exp_shard, exp_mini, exp_pre) in enumerate(MCNS_EXPECTED):
                scale = spec["scales"][i]
                sh = scale["sharding"]
                assert sh["shard_bits"] == exp_shard, f"Scale {i}: shard_bits"
                assert sh["minishard_bits"] == exp_mini, f"Scale {i}: minishard_bits"
                assert sh["preshift_bits"] == exp_pre, f"Scale {i}: preshift_bits"
        finally:
            os.unlink(outpath)

    def test_excessive_scales_clamped(self):
        """Requesting more scales than useful should clamp and warn."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            outpath = f.name
        try:
            result = runner.invoke(cli, [
                "generate",
                "--size", "94088,78317,134576",
                "--scales", "14",
                "--output", outpath,
            ])
            assert result.exit_code == 0, result.output
            assert "clamping" in result.output.lower()
            with open(outpath) as f:
                spec = json.load(f)
            assert len(spec["scales"]) == 12
        finally:
            os.unlink(outpath)
