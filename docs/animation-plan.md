# Animation Plan

## Global Rules

- **No fade-out at end**: The final frame should remain static. These videos will be embedded in slides where the presenter talks over the last frame.
- **Split at transitions**: Before any major visual transition, output the current frame as the end of one video. The next video in the sequence begins with the transition. This gives the presenter control over pacing.
- **Text within frame**: All text lines must fit well within the video dimensions. Never let text extend near the edges.
- **Titles appear instantly**: Don't animate titles letter by letter — just pop them up immediately.
- **Little-endian bit ordering**: MSB on left, LSB on right. Label the ends "MSB" and "LSB".

## Scene Sequence

The scenes are numbered in the order the viewer should watch them:

### 1 — Shard Visualization (`1-ShardVisualizationScene`)

**Goal**: Show the hierarchy: volume → shards → minishards → chunks. This is the first thing the viewer sees — a concrete, spatial understanding of what these terms mean before any math.

**Animation sequence**:
1. Show the full volume as a 3D bounding box with labeled dimensions.
2. Overlay the chunk grid (translucent lines showing 64-voxel divisions).
3. Color-code a few shards as groups of chunks within the volume.
4. **Zoom in** on one shard while the others fade out.
5. Within that shard, show the minishards as colored sub-regions.
6. **Zoom in** on one minishard while the others fade out.
7. Within that minishard, show the individual chunks.
8. End frame: the full hierarchy is labeled — volume → shard → minishard → chunks.

Each zoom-in is a split point for a separate video.

### 2 — Compressed Morton Code (`2-CompressedMortonScene`)

**Goal**: Show how 3D chunk coordinates become a single uint64 morton code, and why the compressed interleaving is asymmetric.

**Animation sequence**:
1. Show volume dimensions, divide by chunk size to get grid, show bits per dimension.
2. Display three input bit rows (X, Y, Z) color-coded, and an output row.
3. **Animate with arrows**: Show arrows from each input bit to its output position (X0 → 0, Y0 → 1, Z0 → 2, X1 → 3, ...). Keep arrows visible so the viewer can see the pattern building up. Animate the first few cycles one-by-one.
4. Batch-fill the uniform middle section (where all 3 dims still contribute).
5. **Endgame**: Animate slowly when dimensions drop out. Announce "X exhausted" etc. Show consecutive bits from remaining dimensions — this is the key insight.
6. End frame: complete interleave table with arrows visible, summary pattern shown.

The arrows from input to output are critical — don't remove them.

### 3 — Bit Allocation (`3-BitAllocationScene`)

**Goal**: Show how the morton code bits are partitioned into preshift, minishard, and shard regions, and connect it back to a concrete 3D chunk.

**Bit ordering**: Little-endian. Shard bits on the left (MSB), preshift bits on the right (LSB). Label "MSB" on left and "LSB" on right above the bit strip.

**Layout for labels**: To avoid text overlap:
- `minishard_bits` label and brace go **above** the bit strip.
- `preshift_bits` and `shard_bits` labels and braces go **below** the bit strip.

**Visual connection to 3D**: Instead of the text pipeline `chunk_coord → morton_code → ...`, show it visually:
- Below the bit strip, display a small 3D chunk box with its (x, y, z) coordinate.
- An arrow from the chunk coordinate to the morton code bit strip.
- Highlight the bit regions to show how the shard ID and minishard ID are extracted.

**Marry to morton scene**: This scene should feel like a direct continuation of Scene 2. The bit strip from the morton interleaving becomes the input to the bit allocation.

### 4 — Multi-Scale Bit Comparison (replaces MultiscaleWalkthroughScene)

**Goal**: Show how the bit allocation changes across scales by aligning the colored bit strips.

**Layout**:
- Stack bit strips vertically, one per scale (or a subset of scales).
- **Right-align (LSB-aligned)**: The preshift and minishard regions stay the same width until the total bits are too few. This makes it visually obvious that as resolution decreases, shard bits shrink from the left while the right side stays stable.
- Label each row with the scale number and grid size.
- Color-code the three regions (shard, minishard, preshift) consistently with Scene 3.

**Animation**:
1. Show Scale 0's bit strip (from Scene 3).
2. Transition: "At Scale 1, volume halves..." and show the next strip appearing below, right-aligned, with fewer shard bits on the left.
3. Continue for several scales, showing the shard region shrinking.
4. At the lower scales, show minishard and preshift also shrinking.
5. End frame: all strips stacked, right-aligned, showing the clear pattern of decreasing bits from the left.

## Output Naming

Videos are named with their sequence number prefix:
- `1-ShardVisualizationScene.mp4`
- `2-CompressedMortonScene.mp4`
- `3-BitAllocationScene.mp4`
- `4-MultiscaleBitComparison.mp4`

Within each scene, sub-videos from transition splits are numbered:
- `1-ShardVisualizationScene-1.mp4` (full volume with shards)
- `1-ShardVisualizationScene-2.mp4` (zoom into one shard, show minishards)
- `1-ShardVisualizationScene-3.mp4` (zoom into one minishard, show chunks)
