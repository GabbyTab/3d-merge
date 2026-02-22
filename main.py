#!/usr/bin/env python3
"""
main.py – Voxel data extraction, Laplacian stacks & multiresolution blending.

Parts:
  1. Data extraction & visualisation
  2. 2-D per-slice & 3-D volumetric Laplacian stacks
  3. Multiresolution blending (the 3-D oraple!)

Usage:
    python main.py                                  # full pipeline
    python main.py --skip-stacks                    # blend only (faster)
    python main.py --skip-stacks --upsample 4       # blend at 4x resolution
    python main.py --skip-stacks --view             # blend + open viewer
    python main.py --elev 25 --azim -60             # custom camera
"""

import argparse
import os
import numpy as np

from voxel_io import (find_asset, load_voxel_obj, list_assets,
                      upsample_nn, upsample_smooth, fill_interior)
from visualize import (
    render_voxel_3d,
    render_slices,
    render_full_stack_2d,
    render_volumetric_stack_3d,
    render_blend_process,
    _render_voxel_to_array,
    DEFAULT_ELEV,
    DEFAULT_AZIM,
)
from laplacian3d import (
    build_laplacian_stack as build_laplacian_stack_3d,
    reconstruct_from_stack as reconstruct_3d,
    default_sigmas,
    to_float,
    to_uint8,
)
from laplacian2d import (
    build_slice_laplacian_stack,
    reconstruct_slice_stack,
)
from blend import (
    multiresolution_blend,
    naive_blend,
    make_half_mask,
    pad_to_common,
)

ASSETS_DIR = "assets/VoxelFruitsAndVegetables"


# ── inspection ───────────────────────────────────────────────────────────────

def inspect_grid(grid: np.ndarray, info: dict) -> None:
    print(f"\n  {info['name']}")
    print(f"    grid shape      : {grid.shape[:3]}")
    print(f"    filled voxels   : {info['n_filled']}")
    print(f"    world origin    : {info['origin']}")
    filled = grid[grid[..., 3] > 0].reshape(-1, 4)
    unique = np.unique(filled, axis=0)
    print(f"    unique colours  : {len(unique)}")
    for c in unique:
        print(f"      RGBA({c[0]:3d}, {c[1]:3d}, {c[2]:3d}, {c[3]:3d})")
    f = filled.astype(np.float32) / 255.0
    print(f"    R [{f[:,0].min():.2f}, {f[:,0].max():.2f}]  "
          f"G [{f[:,1].min():.2f}, {f[:,1].max():.2f}]  "
          f"B [{f[:,2].min():.2f}, {f[:,2].max():.2f}]")


# ── basic renders ────────────────────────────────────────────────────────────

def run_renders(grid: np.ndarray, info: dict,
                elev: float, azim: float) -> None:
    out = "output/renders"
    label = info["name"]
    print(f"\n  Rendering {label} …")
    render_voxel_3d(grid, title=label,
                    save_path=os.path.join(out, f"{label}_3d.png"),
                    elev=elev, azim=azim)
    for ax in range(3):
        render_slices(grid, axis=ax,
                      title=f"{label} – {'XYZ'[ax]} slices",
                      save_path=os.path.join(out, f"{label}_slices_{'XYZ'[ax]}.png"))


# ── 2-D per-slice Laplacian stack ────────────────────────────────────────────

def run_laplacian_2d(grid: np.ndarray, info: dict,
                     sigmas: list[float], axis: int = 1) -> None:
    label = info["name"]
    outdir = os.path.join("output", "laplacian_2d", label)
    axis_name = "XYZ"[axis]
    print(f"\n  2-D slice Laplacian stack  ({label}, axis={axis_name}, "
          f"{len(sigmas)-1} detail levels)")
    print(f"    σ schedule: {[f'{s:.2f}' for s in sigmas]}")

    L, residuals, _ = build_slice_laplacian_stack(grid, axis=axis, sigmas=sigmas)

    for i, level in enumerate(L):
        vals = np.stack(level)
        print(f"    L[{i}] σ {sigmas[i]:.2f}→{sigmas[i+1]:.2f}  "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]")

    render_full_stack_2d(L, residuals, sigmas, axis=axis,
                         title=f"{label} – 2D Laplacian stack ({axis_name} slices)",
                         save_path=os.path.join(outdir, "full_stack.png"))

    recon_slices = reconstruct_slice_stack(L, residuals)
    orig = to_float(grid)
    max_err = 0.0
    for si, rslc in enumerate(recon_slices):
        orig_slc = np.take(orig, si, axis=axis)
        max_err = max(max_err, np.abs(orig_slc - rslc).max())
    print(f"    reconstruction max-error = {max_err:.6f}")


# ── 3-D volumetric Laplacian stack ───────────────────────────────────────────

def run_laplacian_3d(grid: np.ndarray, info: dict,
                     sigmas: list[float],
                     elev: float, azim: float) -> None:
    label = info["name"]
    outdir = os.path.join("output", "laplacian_3d", label)
    print(f"\n  3-D volumetric Laplacian stack  ({label}, "
          f"{len(sigmas)-1} detail levels)")
    print(f"    σ schedule: {[f'{s:.2f}' for s in sigmas]}")

    L, residual, _ = build_laplacian_stack_3d(grid, sigmas=sigmas)

    for i, lap in enumerate(L):
        print(f"    L[{i}] σ {sigmas[i]:.2f}→{sigmas[i+1]:.2f}  "
              f"range=[{lap.min():.3f}, {lap.max():.3f}]")

    render_volumetric_stack_3d(
        L, residual, sigmas,
        title=f"{label} – 3D volumetric Laplacian stack",
        save_path=os.path.join(outdir, "full_stack.png"),
        elev=elev, azim=azim,
    )

    recon = reconstruct_3d(L, residual)
    err = np.abs(to_float(grid) - recon).max()
    print(f"    reconstruction max-error = {err:.6f}")


# ── multiresolution blend (the 3-D oraple) ──────────────────────────────────

def run_blend(grid_a: np.ndarray, info_a: dict,
              grid_b: np.ndarray, info_b: dict,
              sigmas: list[float],
              blend_axis: int,
              elev: float, azim: float,
              upsample_factor: int = 1,
              alpha_threshold: float = 0.5,
              shape_sigma: float | None = None,
              smooth: bool = False,
              save_npy: bool = True) -> np.ndarray:
    """Run the full blend pipeline and return the result grid."""
    label_a = info_a["name"]
    label_b = info_b["name"]
    outdir = os.path.join("output", "blend", f"{label_a}+{label_b}")
    axis_name = "XYZ"[blend_axis]

    print(f"\n{'='*60}")
    print(f" Multiresolution blend: {label_a} + {label_b}")
    print(f"   split axis={axis_name}, {len(sigmas)-1} levels")
    if upsample_factor > 1:
        print(f"   upsample {upsample_factor}x: "
              f"{grid_a.shape[:3]} -> "
              f"{tuple(s*upsample_factor for s in grid_a.shape[:3])}")
    print(f"   alpha threshold: {alpha_threshold}")
    print(f"{'='*60}")

    # ── Pad to common size if shapes differ ─────────────────────────────
    if grid_a.shape != grid_b.shape:
        print(f"  padding {grid_a.shape[:3]} + {grid_b.shape[:3]} → ", end="")
        grid_a, grid_b = pad_to_common(grid_a, grid_b)
        print(f"{grid_a.shape[:3]}")

    # ── Fill hollow interiors (before upsampling so fill works on shell) ──
    print(f"  filling interiors …")
    grid_a = fill_interior(grid_a)
    grid_b = fill_interior(grid_b)

    # ── Upsample if requested ─────────────────────────────────────────────
    if upsample_factor > 1:
        if smooth:
            grid_a = upsample_smooth(grid_a, upsample_factor,
                                     alpha_threshold=alpha_threshold)
            grid_b = upsample_smooth(grid_b, upsample_factor,
                                     alpha_threshold=alpha_threshold)
            print(f"  smooth-upsampled to {grid_a.shape[:3]}")
        else:
            grid_a = upsample_nn(grid_a, upsample_factor)
            grid_b = upsample_nn(grid_b, upsample_factor)
            print(f"  nn-upsampled to {grid_a.shape[:3]}")

    # Build the mask
    spatial = grid_a.shape[:3]
    mask = make_half_mask(spatial, axis=blend_axis)
    mid = spatial[blend_axis] // 2
    print(f"  mask: {axis_name}<{mid} = {label_a},  {axis_name}>={mid} = {label_b}")

    # ── Naive blend (for comparison) ──────────────────────────────────────
    naive = naive_blend(grid_a, grid_b, mask)
    print(f"\n  Naive blend (hard mask):")
    render_voxel_3d(naive, title=f"naive: {label_a}|{label_b}",
                    save_path=os.path.join(outdir, "naive_3d.png"),
                    elev=elev, azim=azim)
    render_slices(naive, axis=blend_axis,
                  title=f"naive blend – {axis_name} slices",
                  save_path=os.path.join(outdir, "naive_slices.png"))

    # ── Multiresolution blend ─────────────────────────────────────────────
    result, details = multiresolution_blend(
        grid_a, grid_b, mask, sigmas=sigmas,
        alpha_threshold=alpha_threshold,
        shape_sigma=shape_sigma,
    )
    n_filled = int((result[..., 3] > 0).sum())
    print(f"\n  Multiresolution blend: {n_filled} filled voxels")
    render_voxel_3d(result, title=f"blend: {label_a}|{label_b}",
                    save_path=os.path.join(outdir, "blend_3d.png"),
                    elev=elev, azim=azim)
    render_slices(result, axis=blend_axis,
                  title=f"multiresolution blend – {axis_name} slices",
                  save_path=os.path.join(outdir, "blend_slices.png"))

    # ── Save .npy for interactive viewer ──────────────────────────────────
    if save_npy:
        npy_path = os.path.join(outdir, "result.npy")
        np.save(npy_path, result)
        print(f"  saved → {npy_path}")
        # Also save inputs for comparison viewing
        np.save(os.path.join(outdir, "input_a.npy"), grid_a)
        np.save(os.path.join(outdir, "input_b.npy"), grid_b)
        np.save(os.path.join(outdir, "naive.npy"), naive)

    # ── Side-by-side comparison ───────────────────────────────────────────
    print(f"\n  Rendering comparison (A | naive | multires | B) …")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for vol, lbl in [(grid_a, label_a), (naive, "naive"),
                     (result, "multires"), (grid_b, label_b)]:
        panels.append(_render_voxel_to_array(vol, title=lbl,
                                             elev=elev, azim=azim,
                                             figsize=(3, 3), dpi=100))
    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        c = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        oh, ow = p.shape[:2]
        c[(max_h-oh)//2:(max_h-oh)//2+oh,
          (max_w-ow)//2:(max_w-ow)//2+ow] = p
        padded.append(c)
    montage = np.hstack(padded)
    fig, ax = plt.subplots(figsize=(montage.shape[1]/100,
                                    montage.shape[0]/100 + 0.4))
    ax.imshow(montage); ax.axis("off")
    fig.suptitle(f"{label_a} + {label_b}:  A | naive | multires | B",
                 fontsize=11, y=1.0)
    plt.tight_layout()
    cmp_path = os.path.join(outdir, "comparison.png")
    fig.savefig(cmp_path, dpi=100, bbox_inches="tight")
    print(f"  saved → {cmp_path}")
    plt.close(fig)

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Voxel Laplacian stacks & multiresolution blending")
    parser.add_argument("--names", nargs="+", default=["apple", "orange"],
                        help="asset names for stacks (case-insensitive)")
    parser.add_argument("--all", action="store_true",
                        help="process every asset for stacks")
    parser.add_argument("--levels", type=int, default=5,
                        help="number of Laplacian detail levels (default 5)")
    parser.add_argument("--axis", type=int, default=1,
                        help="slicing axis for 2D stack montage (0=X,1=Y,2=Z)")
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEV,
                        help=f"camera elevation (default {DEFAULT_ELEV})")
    parser.add_argument("--azim", type=float, default=DEFAULT_AZIM,
                        help=f"camera azimuth (default {DEFAULT_AZIM})")
    parser.add_argument("--skip-stacks", action="store_true",
                        help="skip individual stack visualisation, blend only")
    parser.add_argument("--blend-axis", type=int, default=0,
                        help="split axis for blending (0=X,1=Y,2=Z, default 0)")
    parser.add_argument("--blend-a", type=str, default="apple",
                        help="first asset for blend (default apple)")
    parser.add_argument("--blend-b", type=str, default="orange",
                        help="second asset for blend (default orange)")
    parser.add_argument("--upsample", type=int, default=4,
                        help="upsample factor before blend (default 4)")
    parser.add_argument("--smooth", action="store_true",
                        help="use trilinear upsampling for rounded shapes "
                             "(default: nearest-neighbour / blocky)")
    parser.add_argument("--alpha-thresh", type=float, default=0.5,
                        help="alpha threshold to remove ghost voxels (default 0.5)")
    parser.add_argument("--shape-sigma", type=float, default=None,
                        help="Gaussian sigma for shape blending (default: same "
                             "as coarsest colour sigma). Larger = rounder morph.")
    parser.add_argument("--view", action="store_true",
                        help="open interactive pyvista viewer after blending")
    args = parser.parse_args()

    sigmas = default_sigmas(args.levels)

    print("=" * 60)
    print(" 3-D Voxel Merge – Laplacian stacks & blending")
    print(f"   camera: elev={args.elev}  azim={args.azim}")
    print(f"   levels: {args.levels}  σ: {[f'{s:.2f}' for s in sigmas]}")
    if args.upsample > 1:
        print(f"   upsample: {args.upsample}x")
    print(f"   alpha threshold: {args.alpha_thresh}")
    print("=" * 60)

    # ── Part 1 & 2: Individual stacks ─────────────────────────────────────
    if not args.skip_stacks:
        paths = (list_assets(ASSETS_DIR) if args.all
                 else [find_asset(ASSETS_DIR, n) for n in args.names])

        for path in paths:
            grid, info = load_voxel_obj(path)
            inspect_grid(grid, info)
            run_renders(grid, info, elev=args.elev, azim=args.azim)
            run_laplacian_2d(grid, info, sigmas, axis=args.axis)
            run_laplacian_3d(grid, info, sigmas,
                             elev=args.elev, azim=args.azim)

    # ── Part 3: Multiresolution blend ─────────────────────────────────────
    grid_a, info_a = load_voxel_obj(find_asset(ASSETS_DIR, args.blend_a))
    grid_b, info_b = load_voxel_obj(find_asset(ASSETS_DIR, args.blend_b))

    result = run_blend(
        grid_a, info_a, grid_b, info_b, sigmas,
        blend_axis=args.blend_axis,
        elev=args.elev, azim=args.azim,
        upsample_factor=args.upsample,
        alpha_threshold=args.alpha_thresh,
        shape_sigma=args.shape_sigma,
        smooth=args.smooth,
    )

    # ── Interactive viewer ────────────────────────────────────────────────
    outdir = os.path.join("output", "blend",
                          f"{info_a['name']}+{info_b['name']}")
    npy_path = os.path.join(outdir, "result.npy")

    if args.view:
        from viewer import view_voxels
        print("\n  Opening interactive viewer …")
        view_voxels(result, title=f"Blend: {info_a['name']}|{info_b['name']}")
    else:
        print(f"\n  To explore interactively:")
        print(f"    python viewer.py {npy_path}")

    print("\nDone. Check output/ for results.")


if __name__ == "__main__":
    main()
