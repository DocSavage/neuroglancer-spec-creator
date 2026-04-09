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
# Segmentation spec (default) — pipe JSON to a file
pixi run generate --size 102400,57344,20508 --resolution 16,16,15 > seg_spec.json

# EM grayscale spec
pixi run generate --em --size 204800,114688,10254 --resolution 8,8,30 > em_spec.json
```

The number of scales is auto-computed — it stops at the last scale where the chunk grid has more than one chunk in any dimension. You can override with `--scales N`.

For anisotropic resolutions, only the finer dimensions are downsampled until they catch up to the coarsest, then all dimensions downsample together.

JSON is written to stdout; the summary table goes to stderr so you always see it:

```
Mode: seg (uint64, compressed_segmentation)

Scale  Size                   Grid               Bits          Shard  Mini   Pre  Total    #Shards
--------------------------------------------------------------------------------------------------
0      102400x57344x20508     1600x896x321       11+10+9=30       15     6     9     30      32768
1      51200x28672x10254      800x448x161        10+9+8=27        12     6     9     27       4096
...
```

Presets:

| | `--seg` (default) | `--em` |
|---|---|---|
| `data_type` | `uint64` | `uint8` |
| `type` | `segmentation` | `image` |
| `encoding` | `compressed_segmentation` | `jpeg` |
| `data_encoding` | `gzip` | _(none)_ |
| `compressed_segmentation_block_size` | `[8, 8, 8]` | _(none)_ |
| `voxel_offset` | `[0, 0, 0]` | _(none)_ |

Options:

```
--size X,Y,Z          Volume dimensions in voxels (required)
--seg / --em          Segmentation [default] or EM grayscale preset
--scales N            Number of resolution scales (default: auto)
--resolution R        Base voxel resolution, 1 or 3 values (default: 8)
--chunk-size N        Chunk size, 1 or 3 values (default: 64, e.g., 128,128,32)
--decimal-keys        Use decimal key format (e.g., 8.0x8.0x30.0)
--target-preshift N   Target preshift_bits (default: 9)
--target-minishard N  Target minishard_bits (default: 6)
```

See `spec_examples/` for complete example specs (male CNS and fish2 volumes).

### Render animations

Requires the `viz` environment.

```bash
# All scenes (rendered in parallel via multiprocessing)
pixi run -e viz animate --size 94088,78317,134576 --scene all

# Just the morton code explanation
pixi run -e viz animate --size 94088,78317,134576 --scene morton

# Bit allocation for a specific scale
pixi run -e viz animate --size 94088,78317,134576 --scene bits --scale 0

# Interactive preview (OpenGL)
pixi run -e viz animate --size 94088,78317,134576 --scene morton --preview
```

Each scene produces its own mp4 in `media/videos/`. When `--scene all` is used, all four scenes render in parallel.

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
spec_examples/           Example spec JSON files
cli.py                   CLI entry point
tests/                   Tests (validated against DVID ground truth)
```
