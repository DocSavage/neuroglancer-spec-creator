# neuroglancer-spec-creator

Generate correct [neuroglancer](https://github.com/google/neuroglancer) precomputed multiscale volume spec JSON files, with optional Manim animations that visually explain the sharding math.

## The problem

Getting neuroglancer sharding parameters right is hard. A precomputed sharded volume needs three bit parameters — `preshift_bits`, `minishard_bits`, and `shard_bits` — that control how chunks are organized into shard files. These depend on the compressed morton code interleaving of chunk coordinates, and they change at every resolution scale. Misconfigure them and chunks that should be co-located scatter across shard files, or the shard count explodes.

This tool computes the correct parameters from volume dimensions and optionally renders educational animations showing how the math works.

## What it does

1. **Computes compressed morton codes** — the neuroglancer algorithm that interleaves chunk coordinate bits, skipping dimensions that run out of bits early
2. **Derives sharding parameters** — allocates `preshift_bits` (spatial locality), `minishard_bits` (intra-shard organization), and `shard_bits` (number of shard files) for each resolution scale
3. **Generates spec JSON** — a complete `info` file ready for neuroglancer, with correct per-scale sharding that decreases as the volume halves at each scale
4. **Renders animations** (optional) — four Manim scenes that visually walk through the morton code interleaving, bit allocation, 3D shard layout, and multi-scale progression

## Install

Requires [pixi](https://pixi.sh/).

```bash
# Core functionality (spec generation, no Manim)
pixi install

# With Manim for animations (needs pango, cairo, ffmpeg)
pixi install -e viz
```

## Usage

### Generate a spec

```bash
pixi run generate --size 94088,78317,134576 --scales 11
```

Output:

```
Scale  Size                         Grid                   Bits             Shard  Mini   Pre  Total    #Shards
---------------------------------------------------------------------------------------------------------
0      94088x78317x134576           1471x1224x2103         11+11+12=34         19     6     9     34     524288
1      47044x39159x67288            736x612x1052           10+10+11=31         16     6     9     31      65536
...
10     92x77x132                    2x2x3                  1+1+2=4              0     0     4      4          1

Written: neuroglancer_spec.json
```

Options:

```
--size X,Y,Z          Volume dimensions in voxels (required)
--scales N            Number of resolution scales (default: 11)
--resolution R        Base voxel resolution, 1 or 3 values (default: 8)
--chunk-size N        Chunk size in voxels (default: 64)
--data-type TYPE      uint8, uint16, uint32, uint64, float32 (default: uint8)
--volume-type TYPE    image or segmentation (default: image)
--encoding ENC        jpeg, compressed_segmentation, raw (default: auto)
--output PATH         Output file (default: neuroglancer_spec.json)
--target-preshift N   Target preshift_bits (default: 9)
--target-minishard N  Target minishard_bits (default: 6)
```

### Render animations

Requires the `viz` environment.

```bash
# All scenes
pixi run -e viz animate --size 94088,78317,134576 --scales 3 --scene all

# Just the morton code explanation
pixi run -e viz animate --size 94088,78317,134576 --scene morton

# Bit allocation for a specific scale
pixi run -e viz animate --size 94088,78317,134576 --scene bits --scale 0

# Interactive preview (OpenGL)
pixi run -e viz animate --size 94088,78317,134576 --scene morton --preview
```

Scenes:
- **morton** — How chunk coordinates become a compressed morton code
- **bits** — How morton code bits split into preshift/minishard/shard regions
- **shards** — 3D visualization of shard spatial extent
- **multiscale** — Full walkthrough across all resolution scales

### Run tests

```bash
pixi run test
```

## How the math works

Given a volume of size `(X, Y, Z)` with chunk size 64:

1. **Grid size**: `grid[d] = ceil(size[d] / 64)` for each dimension
2. **Bits per dimension**: `bits[d] = ceil(log2(grid[d]))` — how many bits each dimension contributes to the morton code
3. **Total bits**: `total = bits[0] + bits[1] + bits[2]`
4. **Bit allocation**:
   - `preshift_bits = min(9, total)` — controls spatial locality
   - `minishard_bits = min(6, total - preshift_bits)` — controls intra-shard organization
   - `shard_bits = total - preshift_bits - minishard_bits` — number of shard files = 2^shard_bits
5. **Per scale**: the volume halves (with ceiling) at each scale, so bit counts decrease by ~3 per scale

The compressed morton code interleaves bits from all three dimensions, but unlike standard Z-order interleaving, it skips a dimension once that dimension has contributed all its bits. This means the interleaving pattern is asymmetric when dimensions have different sizes.

## Project structure

```
ngspec/                  Core computation (no Manim dependency)
  morton.py              Compressed morton code
  sharding.py            Bit parameter computation
  spec_generator.py      JSON spec output
scenes/                  Manim scene definitions
  scene_morton.py        Compressed morton code animation
  scene_bits.py          Bit allocation visualization
  scene_shards.py        3D shard visualization
  scene_multiscale.py    Multi-scale walkthrough
cli.py                   CLI entry point
tests/                   Tests (validated against DVID ground truth)
```
