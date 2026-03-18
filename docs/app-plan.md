# neuroglancer-spec-creator: Application Plan

An educational Manim-based Python application that visually walks users through the neuroglancer precomputed sharding scheme — from volume dimensions through compressed morton codes to shard bit allocation — and outputs a correct neuroglancer multiscale volume spec JSON.

## Motivation

Getting neuroglancer sharding parameters right is hard. The compressed morton code interleaving is non-obvious, the three bit parameters (`preshift_bits`, `minishard_bits`, `shard_bits`) interact in subtle ways, and misconfiguring them causes chunks that should be co-located to scatter across shard files. This tool makes the math visible and produces correct specs.

---

## Architecture Overview

The app has two layers:

1. **Core logic** (`ngspec/`) — Pure Python computation of sharding parameters, compressed morton codes, and spec generation. No Manim dependency. Testable independently.

2. **Manim scenes** (`scenes/`) — Animated visualizations that import from the core logic. Rendered to video files (MP4/GIF) or previewed interactively via OpenGL mode.

A CLI (`cli.py`) ties them together: the user provides volume dimensions, and the tool either generates a spec JSON directly or renders educational animations explaining each scale's sharding.

```
neuroglancer-spec-creator/
├── cli.py                    # Entry point: pixi run generate / pixi run animate
├── ngspec/                   # Core computation (no Manim dependency)
│   ├── __init__.py
│   ├── morton.py             # Compressed morton code implementation
│   ├── sharding.py           # Bit parameter computation
│   └── spec_generator.py     # JSON spec output
├── scenes/                   # Manim scene definitions
│   ├── __init__.py
│   ├── scene_morton.py       # Scene 1: Compressed morton code explained
│   ├── scene_bits.py         # Scene 2: Bit allocation breakdown
│   ├── scene_shards.py       # Scene 3: 3D shard visualization
│   └── scene_multiscale.py   # Scene 4: Full multi-scale walkthrough
├── tests/
│   ├── test_morton.py        # Verify against neuroglancer reference
│   ├── test_sharding.py      # Verify bit allocation algorithm
│   └── test_spec.py          # Verify JSON output matches known-good specs
├── docs/
│   └── app-plan.md           # This file
├── pixi.toml
└── README.md
```

---

## Core Logic (`ngspec/`)

### `morton.py` — Compressed Morton Code

Implements the algorithm from the [neuroglancer spec](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#compressed-morton-code):

```python
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
    max_bits = max(grid_size).bit_length()
    for i in range(max_bits):
        for dim in range(3):
            if (1 << i) < grid_size[dim]:
                bit = (coord[dim] >> i) & 1
                output |= bit << j
                j += 1
    return output
```

Key helper functions:
- `bits_per_dimension(grid_size)` — returns how many bits each dimension contributes
- `interleave_table(grid_size)` — returns the full mapping showing which output bit comes from which dimension and input bit position (this is what the animation visualizes)
- `total_chunk_bits(grid_size)` — sum of bits across dimensions

### `sharding.py` — Bit Parameter Computation

Implements the algorithm derived from analyzing known-good specs:

```python
def compute_sharding_params(volume_size: tuple[int, int, int],
                             chunk_size: int = 64,
                             target_preshift: int = 9,
                             target_minishard: int = 6) -> dict:
    """Compute optimal shard_bits, minishard_bits, preshift_bits.

    The total bits must equal the sum of chunk coordinate bits across
    all dimensions (so the morton code covers the full coordinate space).

    Priority: preshift_bits >= minishard_bits >= shard_bits
    - preshift controls spatial locality within minishards
    - minishard controls internal shard organization
    - shard controls number of shard files (2^shard_bits)
    """
    grid_size = tuple(math.ceil(s / chunk_size) for s in volume_size)
    total = total_chunk_bits(grid_size)

    preshift = min(target_preshift, total)
    remaining = total - preshift
    minishard = min(target_minishard, remaining)
    shard = remaining - minishard

    return {
        "shard_bits": shard,
        "minishard_bits": minishard,
        "preshift_bits": preshift,
        "total_chunk_bits": total,
        "grid_size": grid_size,
        "num_shards": 2 ** shard,
        "num_minishards_per_shard": 2 ** minishard,
        "chunks_per_minishard": 2 ** preshift,
    }
```

### `spec_generator.py` — JSON Output

```python
def generate_spec(volume_size: tuple[int, int, int],
                  num_scales: int,
                  voxel_resolution: tuple[float, float, float] = (8.0, 8.0, 8.0),
                  data_type: str = "uint64",
                  volume_type: str = "segmentation") -> dict:
    """Generate a complete neuroglancer multiscale volume spec.

    Computes correct per-scale sharding parameters by halving the volume
    at each scale and recalculating bit allocations.
    """
```

---

## Manim Scenes (`scenes/`)

Each scene is a self-contained Manim `Scene` or `ThreeDScene` class. They can be rendered individually or composed into a full walkthrough.

### Scene 1: Compressed Morton Code (`scene_morton.py`)

**Goal**: Show how chunk coordinates become a single uint64 morton code, and why compressed interleaving differs from standard interleaving.

**Visual elements**:
- Three horizontal rows of colored bit cells representing X, Y, Z chunk coordinates (e.g., X=11 bits blue, Y=11 bits green, Z=12 bits red)
- An output row showing the interleaved morton code bits
- Arrows connecting each input bit to its output position
- Animation: bits fly from their source dimension into the output, one at a time, showing the interleaving pattern
- Side panel showing the algorithm pseudocode, highlighting the current step
- Comparison inset: standard morton (wastes bits) vs compressed morton (no waste)

**Key insight to convey**: In standard morton, you always cycle X→Y→Z→X→Y→Z. In compressed morton, when a dimension runs out of bits, it's skipped. So the pattern might be X→Y→Z→X→Y→Z→...→X→Z→X→Z→Z (if Z has more bits than X and Y).

**Animation sequence**:
1. Show volume dimensions (e.g., 94088 x 77248 x 134592 voxels)
2. Divide by chunk size (64) to get grid: 1471 x 1208 x 2103 chunks
3. Show bits needed per dimension: ceil(log2(1471))=11, ceil(log2(1208))=11, ceil(log2(2103))=12
4. Animate the interleaving: bit 0 from X, bit 0 from Y, bit 0 from Z, bit 1 from X, ... showing which dimensions contribute at each step
5. Highlight the moment Z continues contributing after X and Y are exhausted

### Scene 2: Bit Allocation Breakdown (`scene_bits.py`)

**Goal**: Show how the morton code bits are partitioned into preshift, minishard, and shard regions.

**Visual elements**:
- The morton code as a long horizontal strip of colored bits (from Scene 1)
- Three colored regions overlaid: preshift (bottom/rightmost bits, green), minishard (middle, yellow), shard (top/leftmost, red)
- Labeled brackets showing bit ranges
- Numeric annotations: "2^19 = 524,288 shards", "2^6 = 64 minishards/shard", "2^9 = 512 chunks/minishard"
- A pipeline diagram: `chunk_coord → morton_code → >>preshift → hash → extract_minishard → extract_shard`

**Animation sequence**:
1. Start with the morton code strip from Scene 1 (34 bits for scale 0)
2. Highlight the lowest 9 bits → "preshift_bits: these bits are discarded by right-shifting, grouping 2^9=512 nearby chunks"
3. Show the right-shift operation visually (bits slide right, lowest 9 fall off)
4. From the remaining 25 bits, highlight the lowest 6 → "minishard_bits: 2^6=64 minishards per shard"
5. The remaining 19 bits → "shard_bits: 2^19=524,288 shard files"
6. Show a chunk coordinate flowing through the full pipeline to produce a shard ID

### Scene 3: 3D Shard Visualization (`scene_shards.py`)

**Goal**: Show what shards look like in 3D space relative to the volume bounding box.

**Visual elements**:
- `ThreeDScene` with a wireframe bounding box showing volume dimensions (labeled edges)
- Colored sub-cubes representing a few shards
- Dimension labels along edges (in voxels and in chunks)
- Camera rotation to show the 3D structure

**Animation sequence**:
1. Draw the volume bounding box with dimension labels
2. Overlay the chunk grid (translucent lines showing 64-voxel divisions)
3. Highlight a single shard's spatial extent as a colored region
4. Show 2-3 neighboring shards in different colors
5. Show how morton ordering traces a Z-curve through the chunks within a shard
6. Optionally: show what happens with wrong bit params — chunks from distant regions ending up in the same shard

**Design note**: For a 94088^3 volume with 2^19 shards, each shard covers roughly a (94088/64)^(1/3) ≈ 12-chunk-wide region per dimension. The visualization can show a simplified smaller volume (e.g., 8x8x8 chunks) to keep it legible.

### Scene 4: Multi-Scale Walkthrough (`scene_multiscale.py`)

**Goal**: Tie it all together — show how the same logic applies at each resolution scale with decreasing bit counts.

**Visual elements**:
- A table/grid showing all scales side by side
- Per-scale: volume size, grid size, bits per dimension, total bits, shard/minishard/preshift allocation
- The 3D visualization shrinking at each scale
- Running total of shard files across all scales

**Animation sequence**:
1. Start at scale 0: full resolution, show the bit breakdown (recap from Scene 2)
2. Transition: "At scale 1, volume halves in each dimension..."
3. Show the grid shrinking, bits per dimension decreasing by ~1 each
4. Show the new bit allocation: shard_bits decreases by 3 (one per dimension)
5. Repeat for scales 2-3, then fast-forward through remaining scales
6. At the lowest scales, show preshift and minishard also shrinking
7. Final: display the complete table and the output JSON spec

---

## CLI (`cli.py`)

Two main commands:

### `pixi run generate` — Quick spec generation (no Manim)

```
$ pixi run generate --size 94088,77248,134592 --scales 11 --resolution 8

Scale  Size                  Grid           Bits    Shard  Mini  Pre  Total  #Shards
0      94088x77248x134592    1471x1208x2103 11+11+12=34   19     6    9     524288
1      47044x38624x67296      736x604x1052  10+10+11=31   16     6    9     65536
...
10     92x76x132                2x2x3         1+1+2=4      0     0    4     1

Written: neuroglancer_spec.json
```

### `pixi run animate` — Render Manim scenes

```
$ pixi run animate --size 94088,77248,134592 --scales 3 --scene all
# Renders Scene 1-4 as MP4 files in media/

$ pixi run animate --size 94088,77248,134592 --scene morton
# Render just the compressed morton code explanation

$ pixi run animate --size 94088,77248,134592 --scene bits --scale 0
# Render the bit allocation breakdown for scale 0

$ pixi run animate --preview  # OpenGL interactive preview
```

---

## Testing (`tests/`)

### `test_morton.py`

- Verify compressed morton code against known values from the neuroglancer reference implementation
- Test edge cases: dimensions that are powers of 2, one dimension much larger than others, single-chunk dimensions
- Test that `interleave_table` correctly reports which output bits come from which dimensions

### `test_sharding.py`

- **Ground truth test**: Compute sharding params for the DVID repo's `mcns-ng-specs.json` volume sizes and assert they match the known-good values exactly:
  ```
  Scale 0: 94088x78317x134576 → shard=19, mini=6, pre=9
  Scale 1: 47044x39159x67288  → shard=16, mini=6, pre=9
  ...
  Scale 10: 92x77x132         → shard=0,  mini=0, pre=4
  ```
- Test that `total_chunk_bits == shard_bits + minishard_bits + preshift_bits` for all scales
- Test degenerate cases: 1-voxel volume, single-chunk volume, very asymmetric volumes

### `test_spec.py`

- Generate a full spec and validate JSON structure matches neuroglancer schema
- Round-trip test: generate spec → feed to DVID's export.go `initialize()` logic (conceptually — verify the bit math matches)

---

## Dependencies

```toml
# pixi.toml
[workspace]
name = "neuroglancer-spec-creator"
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = ">=3.10,<3.13"

[pypi-dependencies]
manim = ">=0.18.0"
click = ">=8.1.0"
pytest = ">=8.0"

[tasks]
generate = "python cli.py generate"
animate = "python cli.py animate"
test = "pytest tests/ -v"
```

Manim pulls in numpy, scipy, Pillow, pycairo, manimpango, and moderngl. No other heavy dependencies needed.

---

## Implementation Order

1. **`ngspec/morton.py`** + `tests/test_morton.py` — Get the compressed morton code right first. This is the mathematical foundation.
2. **`ngspec/sharding.py`** + `tests/test_sharding.py` — Bit allocation algorithm, validated against DVID's known-good specs.
3. **`ngspec/spec_generator.py`** + `tests/test_spec.py` — JSON output.
4. **`cli.py generate`** — Quick spec generation without Manim.
5. **`scenes/scene_morton.py`** — The most educational and novel scene.
6. **`scenes/scene_bits.py`** — Bit allocation visualization.
7. **`scenes/scene_shards.py`** — 3D shard visualization.
8. **`scenes/scene_multiscale.py`** — Putting it all together.
9. **`cli.py animate`** — Wire up the Manim rendering commands.

Steps 1-4 can be completed and useful without any Manim work. Steps 5-9 add the educational visualization layer.

---

## References

### Neuroglancer precomputed format

- **Compressed Morton Code specification**: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#compressed-morton-code
  — The authoritative definition of the compressed morton interleaving algorithm. The pseudocode in the "Compressed Morton Code" section is what `ngspec/morton.py` must implement exactly.

- **Sharded format specification**: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#sharded-chunk-storage
  — Defines `shard_bits`, `minishard_bits`, `preshift_bits`, the hash function, and the binary shard file layout (shard index → minishard index → chunk data).

- **Full precomputed volume spec**: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md
  — Top-level reference for the `info` JSON file format, multiscale metadata, and all encoding types.

### Known-good neuroglancer spec JSON (ground truth for tests)

- **DVID repo mcns-ng-specs.json**: `/home/katzw/go-code/src/github.com/janelia-flyem/dvid/test_data/mcns-ng-specs.json`
  — A correct 11-scale spec for the male CNS dataset (94088 x 78317 x 134576, grayscale). Sharding params decrease properly across scales: shard_bits=19→0, minishard_bits=6→0, preshift_bits=9→4. Use this as the primary test fixture.

- **tensorstore-export mcns-v0.9-ng-specs.json**: `/home/katzw/tensorstore-export/mcns-v0.9-ng-specs.json`
  — A **known-incorrect** spec (shard_bits=21 at all scales, does not decrease). Useful as a negative test case showing what happens when bits are wrong.

### DVID export-shards implementation

- **export.go**: `/home/katzw/go-code/src/github.com/janelia-flyem/dvid/datatype/labelmap/export.go`
  — The Go implementation that consumes the neuroglancer spec JSON. Key functions:
  - `ngScale.initialize()` (line 84): computes chunk coordinate bits, grid sizes, and shard/minishard masks
  - `ngScale.mortonCode()` (line 126): Go implementation of compressed morton code
  - `ngScale.computeShardID()` (line 138): applies preshift, hash, and bit extraction
  - `shardHandler.Initialize()` (line 435): derives shard voxel dimensions from the bit parameters

- **compressed_test.go**: `/home/katzw/go-code/src/github.com/janelia-flyem/dvid/datatype/common/labels/compressed_test.go`
  — Go unit tests for block compression roundtrips using real test data. Shows how `MakeBlock` + `MarshalBinary` + `UnmarshalBinary` + `MakeLabelVolume` are verified.

### Manim

- **Manim Community Edition**: https://www.manim.community/
- **GitHub repo**: https://github.com/ManimCommunity/manim
- **3D scene reference**: https://docs.manim.community/en/stable/reference/manim.mobject.three_d.three_dimensions.html
  — `Cube`, `Prism`, `ThreeDAxes`, `Line3D` for volume/shard visualization
- **OpenGL interactivity**: https://slama.dev/manim/opengl-and-interactivity/
  — How to use `--renderer=opengl` for interactive preview with keyboard/mouse handlers
- **Manim DSA plugin** (data structure animations): https://github.com/F4bbi/manim-dsa
  — Pre-built array/tree visualization components that may be useful for bit-level displays

### Bit allocation algorithm (derived from analysis)

The relationship between volume size and sharding params, derived from the DVID repo's known-good spec:

```
grid_size[d] = ceil(volume_size[d] / chunk_size)        for each dimension d
bits_per_dim[d] = ceil(log2(grid_size[d]))               bits needed per dimension
total_chunk_bits = sum(bits_per_dim)                      total bits in compressed morton code

preshift_bits    = min(target_preshift, total_chunk_bits) target_preshift=9 (spatial locality)
remaining        = total_chunk_bits - preshift_bits
minishard_bits   = min(target_minishard, remaining)       target_minishard=6 (intra-shard org)
shard_bits       = remaining - minishard_bits              number of shard files = 2^shard_bits
```

This was verified to reproduce all 11 scales of `mcns-ng-specs.json` exactly.
