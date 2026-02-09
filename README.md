# 3D Voxel Merge

Multiresolution blending of voxel volumes -- the 3D extension of Burt & Adelson's
Laplacian pyramid image blending, applied to MagicaVoxel assets.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start -- build the oraple

```bash
source .venv/bin/activate

# Smooth blend (rounded shapes, recommended)
python main.py --skip-stacks --smooth

# Blocky blend (preserves voxel aesthetic)
python main.py --skip-stacks
```

Results land in `output/blend/Apple-0+Orange-0/`.

## Interactive 3D viewer

After running the blend, explore the results interactively with pyvista.
Each command opens a 3D window -- **left-drag** to orbit, **scroll** to zoom,
**right-drag** to pan, **q** or close the window to exit.

```bash
source .venv/bin/activate

# View the apple
python viewer.py output/blend/Apple-0+Orange-0/input_a.npy --title Apple

# View the orange
python viewer.py output/blend/Apple-0+Orange-0/input_b.npy --title Orange

# View the oraple (multiresolution blend)
python viewer.py output/blend/Apple-0+Orange-0/result.npy --title Oraple

# View the naive blend (hard seam, for comparison)
python viewer.py output/blend/Apple-0+Orange-0/naive.npy --title Naive
```

Or launch the viewer directly after blending:

```bash
python main.py --skip-stacks --smooth --view
```

## Full pipeline (stacks + blend)

```bash
# Everything: Laplacian stacks for apple & orange, then blend
python main.py --smooth

# Custom assets
python main.py --skip-stacks --blend-a banana --blend-b carrot --smooth

# Custom camera angle
python main.py --skip-stacks --smooth --elev 25 --azim -60

# Blocky style, 8x upsample
python main.py --skip-stacks --upsample 8
```

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--smooth` | off | Trilinear upsampling (rounded shapes) |
| `--upsample N` | 4 | Spatial upsampling factor |
| `--levels N` | 5 | Number of Laplacian detail levels |
| `--blend-axis` | 0 (X) | Split axis: 0=X, 1=Y, 2=Z |
| `--blend-a` / `--blend-b` | apple / orange | Assets to blend |
| `--elev` / `--azim` | 30 / -135 | Camera angle for static renders |
| `--alpha-thresh` | 0.5 | Surface threshold (lower = rounder) |
| `--skip-stacks` | off | Skip Laplacian stack visualization |
| `--view` | off | Open pyvista viewer after blending |

## Project structure

```
voxel_io.py      Parse OBJ -> RGBA grids, upsample, fill interiors
visualize.py     Matplotlib 3D renders, slice montages, stack layouts
laplacian3d.py   3D volumetric Gaussian/Laplacian stacks
laplacian2d.py   Per-slice 2D Gaussian/Laplacian stacks
blend.py         Multiresolution blending (Burt & Adelson in 3D)
viewer.py        Interactive pyvista 3D voxel viewer
main.py          CLI driving the full pipeline
```

## How the blend works

1. **Fill interiors** -- the OBJ models are hollow shells; we seal gaps and flood-fill
2. **Upsample** -- nearest-neighbor (blocky) or trilinear (smooth) to increase resolution
3. **Laplacian stack** -- decompose each volume into frequency bands (fine detail to coarse blob)
4. **Gaussian mask stack** -- blur the binary split-mask at each frequency level
5. **Blend per level** -- `blended[i] = mask[i] * A[i] + (1-mask[i]) * B[i]`
6. **Reconstruct** -- sum all blended layers + residual
7. **Alpha threshold** -- snap to fully opaque, discard ghost voxels
