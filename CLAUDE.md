# neuroglancer-spec-creator

An educational Manim-based Python application that computes neuroglancer precomputed sharding parameters and generates multiscale volume spec JSON files.

## Project structure

- `ngspec/` — Core computation (no Manim dependency): morton codes, sharding params, spec generation
- `scenes/` — Manim scene definitions for animated visualizations
- `cli.py` — CLI entry point (`pixi run generate` / `pixi run animate`)
- `tests/` — pytest tests validating against known-good DVID specs

## Development

```bash
pixi install          # Install dependencies
pixi run test         # Run tests
pixi run generate     # Generate a spec JSON
pixi run animate      # Render Manim scenes
```

## Key algorithms

- **Compressed morton code**: Bits per dimension = `(grid_size - 1).bit_length()` (equivalent to `ceil(log2(grid_size))` for grid_size > 0). The interleaving skips dimensions that have exhausted their bits.
- **Sharding params**: `total_bits = sum(bits_per_dim)`, then allocate preshift (target 9), minishard (target 6), remainder to shard_bits.
- **Scale halving**: Each scale halves the volume with ceiling division: `ceil(size / 2)`.

## Ground truth test data

- Correct spec: `/home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json`
- Incorrect spec (negative test): `/home/katzw/tensorstore-export/mcns-v0.9-ng-specs.json`
- Go reference implementation: `/home/katzw/go-code/src/github.com/janelia-flyem/dvid/datatype/labelmap/export.go`
