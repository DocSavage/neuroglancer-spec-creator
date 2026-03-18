# Animation Plan

## Global Rules

- **No fade-out at end**: The final frame should remain static. These videos will be embedded in slides where the presenter talks over the last frame.
- **Split at transitions**: Before any major visual transition, output the current frame as the end of one video. The next video in the sequence begins with the transition. This gives the presenter control over pacing.
- **Text within frame**: All text lines must fit well within the video dimensions. Never let text extend near the edges.
- **Titles appear instantly**: Don't animate titles letter by letter — just pop them up immediately.
- **Little-endian bit ordering**: MSB on left, LSB on right. Label the ends "MSB" and "LSB".
- **No bottom subtitles/commentary**: Don't add explanatory text at the bottom of the screen. The animations should speak for themselves and the presenter will narrate.

## Scene Sequence

The scenes are numbered in the order the viewer should watch them:

### 1 — Shard Visualization (`1-ShardVisualizationScene`)

**Goal**: Show the hierarchy: volume → shards → minishards → chunks. This is the first thing the viewer sees — a concrete, spatial understanding of what these terms mean before any math.

**Animation sequence**:
1. Show the full volume as a 3D bounding box with labeled dimensions.
2. Overlay the "shard" grid (translucent lines showing 8 x 8 x 8 divisions).
3. Rotate the full volume with the 8 x 8 x 8 shards within.
4. **Zoom in** on one shard so it occupies most of the screen. The shard stays as-is in its color. Then the OTHER shards fade out, leaving only the zoomed-in shard visible. Note that only ONE shard remains after the other shards fade out.
5. Within the remaining shard, show the "minishards" as smaller colored blocks distinct from the color of the shard. The minishard subdivision should reflect the actual `minishard_bits` — e.g., if 6 bits, that means 2^6 = 64 minishards, roughly 4×4×4 within the shard. 
6. **Zoom in** on one minishard so it occupies most of the screen. The OTHER minishards fade out, leaving only the zoomed-in minishard. So at the end of this subscene, only one zoomed-in minishard (represented as a colored block) is visible.
7. Then, within that minishard, reveal smaller "chunks" with different colors corresponding to the `preshift_bits`.
8. End frame: the full hierarchy is labeled — volume → shard → minishard → chunks.

Each zoom-in is a split point for a separate video.

### 2 — Compressed Morton Code (`2-CompressedMortonScene`)

**Goal**: Show how 3D chunk coordinates become a single uint64 morton code, and why the compressed interleaving is asymmetric.

**Animation sequence**:
1. Show volume dimensions, divide by chunk size to get grid, show bits per dimension. **Keep this text visible** throughout — don't fade it out before showing the bit rows.
2. Display three input bit rows (X, Y, Z) color-coded, and a **single output row**. Use an ellipsis "..." to skip the uniform middle bits (e.g., show bits 0-11, then "...", then bits 24-33). This keeps the output to one line.
3. **Animate with arrows**: For each visible bit, show an arrow from the source input cell to the output cell, color the output cell, then fade the arrow out. One arrow at a time.
4. The middle bits (covered by "...") are implied — no animation needed for them.
5. **Endgame**: Continue showing arrows for ALL bits after the ellipsis, especially as dimensions drop out. This is the key visual — viewers see consecutive same-color bits appearing.
6. End frame: complete output row with colored cells and the pattern visible. **No subtitle text** at the bottom — let the animation speak for itself.

### 3 — Bit Allocation (`3-BitAllocationScene`)

**Goal**: Show how the morton code bits are partitioned into preshift, minishard, and shard regions, and connect it back to a concrete 3D chunk.

**Bit ordering**: Little-endian. Shard bits on the left (MSB), preshift bits on the right (LSB). Label "MSB" on left and "LSB" on right above the bit strip.

**Layout for labels**: To avoid text overlap:
- `minishard_bits` label and brace go **above** the bit strip.
- `preshift_bits` and `shard_bits` labels and braces go **below** the bit strip.

**Pacing**: Each region highlight (shard, minishard, preshift) should be in its own sub-video via `next_section()`. The transitions between highlights are too fast for human comprehension if done in one continuous video. Split so each sub-video ends on the frame showing that region highlighted, giving the presenter time to explain.

**Visual connection to 3D**: Instead of the text pipeline `chunk_coord → morton_code → ...`, show it visually:
- Below the bit strip, display a small 3D chunk box with its (x, y, z) coordinate.
- An arrow from the chunk coordinate to the morton code bit strip.
- Highlight the bit regions to show how the shard ID and minishard ID are extracted.

**Marry to morton scene**: This scene should feel like a direct continuation of Scene 2. The bit strip from the morton interleaving becomes the input to the bit allocation.

### 4 — Multi-Scale Bit Comparison (replaces MultiscaleWalkthroughScene)

**Goal**: Show how the bit allocation changes across scales by aligning the colored bit strips.

**Layout**:
- Stack bit strips vertically, one per scale (or a subset of scales).
- **Right-justify strips close to the right edge** of the video screen. The LSB end should be near the right edge so the strips use the available width well.
- **Right-align (LSB-aligned)**: The preshift and minishard regions stay the same width until the total bits are too few. This makes it visually obvious that as resolution decreases, shard bits shrink from the left while the right side stays stable.
- **Left-align all row labels**: Each row's label (scale number, grid size, bit count) should be aligned to a fixed left edge with padding from the left side of the video. Don't place labels immediately next to each strip's left edge — that creates a staircase look as strips get shorter.
- **MSB/LSB labels**: Show with the very first scale strip, not after all strips are displayed.
- Color-code the three regions (shard, minishard, preshift) consistently with Scene 3.

**Animation**:
1. Show Scale 0's bit strip with MSB/LSB labels and row label.
2. For each subsequent scale, show the next strip appearing below, right-aligned, with fewer shard bits on the left.
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
