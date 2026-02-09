"""
visualize.py – Render RGBA voxel grids to images using matplotlib.

Rendering modes:
  1. 3D isometric view             (Axes3D voxels)
  2. Slice montage                  (all slices along one axis, in a row)
  3. 2D full-stack montage          (rows = pyramid levels, cols = slices)
  4. 3D volumetric stack montage    (each level is a projected 3D voxel render)

Camera angle (elev, azim) is parametrised everywhere.
"""

import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Default camera: shows body contour + stem/leaf clearly
DEFAULT_ELEV = 30
DEFAULT_AZIM = -135


# ── colour helpers ───────────────────────────────────────────────────────────

def _rgba_to_float(grid: np.ndarray) -> np.ndarray:
    if grid.dtype == np.uint8:
        return grid.astype(np.float32) / 255.0
    return grid.astype(np.float32)


def _composite_on_white(rgba: np.ndarray) -> np.ndarray:
    """RGBA float [0,1] → RGB composited onto white."""
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + 1.0 * (1 - a)


# ── render a single 3-D view to an in-memory image ─────────────────────────

def _render_voxel_to_array(grid: np.ndarray,
                           title: str = "",
                           elev: float = DEFAULT_ELEV,
                           azim: float = DEFAULT_AZIM,
                           figsize: tuple = (4, 4),
                           dpi: int = 100,
                           alpha_threshold: int = 0) -> np.ndarray:
    """Render a 3D voxel view and return it as an RGB numpy array."""
    filled = grid[..., 3] > alpha_threshold
    colors = _rgba_to_float(grid)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled, facecolors=colors, edgecolor=(0, 0, 0, 0.08))
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=9, pad=2)

    shape = np.array(grid.shape[:3])
    mx = shape.max()
    ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_zlim(0, mx)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)

    buf.seek(0)
    img = PILImage.open(buf).convert("RGB")
    return np.array(img)


# ── 3-D voxel rendering (single view, saved to file) ────────────────────────

def render_voxel_3d(grid: np.ndarray,
                    title: str = "",
                    save_path: str | None = None,
                    elev: float = DEFAULT_ELEV,
                    azim: float = DEFAULT_AZIM,
                    figsize: tuple = (8, 8),
                    alpha_threshold: int = 0) -> None:
    """Render and save a 3D isometric-style view of the voxel grid."""
    filled = grid[..., 3] > alpha_threshold
    colors = _rgba_to_float(grid)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled, facecolors=colors, edgecolor=(0, 0, 0, 0.08))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=14)
    shape = np.array(grid.shape[:3])
    mx = shape.max()
    ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_zlim(0, mx)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.close(fig)


# ── single-row slice montage ────────────────────────────────────────────────

def render_slices(grid: np.ndarray,
                  axis: int = 0,
                  title: str = "",
                  save_path: str | None = None,
                  figsize_per_slice: float = 1.5,
                  show_all: bool = False) -> None:
    """Montage of 2-D slices along one axis (skips empty unless show_all)."""
    n_slices = grid.shape[axis]
    colors_f = _rgba_to_float(grid)

    slices = []
    for i in range(n_slices):
        s = np.take(colors_f, i, axis=axis)
        has_content = np.any(np.take(grid[..., 3], i, axis=axis) > 0)
        if show_all or has_content:
            slices.append((i, s))

    if not slices:
        return

    n = len(slices)
    fig, axes = plt.subplots(1, n,
                             figsize=(figsize_per_slice * n,
                                      figsize_per_slice * 1.3))
    if n == 1:
        axes = [axes]

    axis_names = ["X", "Y", "Z"]
    for ax, (idx, s) in zip(axes, slices):
        rgb = _composite_on_white(s)
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title(f"{axis_names[axis]}={idx}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.close(fig)


# ── 2-D full-stack montage (rows = levels, cols = slices) ────────────────────

def render_full_stack_2d(layers_2d: list[list[np.ndarray]],
                         residuals_2d: list[np.ndarray],
                         sigmas: list[float],
                         axis: int = 1,
                         title: str = "",
                         save_path: str | None = None,
                         figsize_per_cell: float = 1.2) -> None:
    """Render the 2D per-slice stack as a grid: rows = levels, cols = slices.

    Last row is the residual.
    """
    n_levels = len(layers_2d)
    n_slices = len(residuals_2d)
    n_rows = n_levels + 1
    n_cols = n_slices
    axis_names = ["X", "Y", "Z"]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_cell * n_cols,
                                      figsize_per_cell * n_rows * 1.1),
                             squeeze=False)

    for li in range(n_levels):
        for si in range(n_slices):
            ax = axes[li][si]
            img = layers_2d[li][si]
            # Normalise signed detail for display
            rgb = img[..., :3]
            amax = np.abs(rgb).max()
            if amax > 0:
                disp_rgb = rgb / (2 * amax) + 0.5
            else:
                disp_rgb = np.full_like(rgb, 0.5)
            ax.imshow(disp_rgb, origin="lower", interpolation="nearest",
                      vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if si == 0:
                s_lo, s_hi = sigmas[li], sigmas[li + 1]
                ax.set_ylabel(f"L{li}\n\u03c3 {s_lo:.2f}\u2192{s_hi:.2f}",
                              fontsize=7, rotation=0, labelpad=40, va="center")
            if li == 0:
                ax.set_title(f"{axis_names[axis]}={si}", fontsize=7)

    for si in range(n_slices):
        ax = axes[n_levels][si]
        img = residuals_2d[si]
        rgb = _composite_on_white(img)
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if si == 0:
            ax.set_ylabel(f"residual\n\u03c3={sigmas[-1]:.2f}",
                          fontsize=7, rotation=0, labelpad=40, va="center")

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.close(fig)


# ── 3-D volumetric stack montage (projected 3D views) ───────────────────────

def render_volumetric_stack_3d(detail_vols: list[np.ndarray],
                               residual_vol: np.ndarray,
                               sigmas: list[float],
                               title: str = "",
                               save_path: str | None = None,
                               elev: float = DEFAULT_ELEV,
                               azim: float = DEFAULT_AZIM,
                               cell_figsize: tuple = (3, 3),
                               dpi: int = 100) -> None:
    """Render the 3-D volumetric Laplacian stack as a column of projected
    3-D voxel views.

    Each row is one level rendered from the same camera angle:
      L0, L1, …, L_{n-1}, residual

    Parameters
    ----------
    detail_vols : list of float32 (X, Y, Z, 4) signed detail volumes
    residual_vol : float32 (X, Y, Z, 4) coarsest Gaussian
    sigmas : σ schedule (len = n_detail + 1)
    elev, azim : camera angle for the 3-D projection
    """
    from laplacian3d import detail_to_viewable, to_uint8

    n_levels = len(detail_vols)

    # Render each level to an image array
    panels = []
    for i, vol in enumerate(detail_vols):
        viewable = detail_to_viewable(vol)
        s_lo, s_hi = sigmas[i], sigmas[i + 1]
        label = f"L{i}  \u03c3 {s_lo:.2f}\u2192{s_hi:.2f}"
        img = _render_voxel_to_array(viewable, title=label,
                                     elev=elev, azim=azim,
                                     figsize=cell_figsize, dpi=dpi)
        panels.append(img)

    # Residual
    res_u8 = to_uint8(residual_vol)
    label = f"residual  \u03c3={sigmas[-1]:.2f}"
    img = _render_voxel_to_array(res_u8, title=label,
                                 elev=elev, azim=azim,
                                 figsize=cell_figsize, dpi=dpi)
    panels.append(img)

    # Make all panels the same size (pad to max)
    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        oh, ow = p.shape[:2]
        y0 = (max_h - oh) // 2
        x0 = (max_w - ow) // 2
        canvas[y0:y0 + oh, x0:x0 + ow] = p
        padded.append(canvas)

    # Stack vertically into one tall image
    montage = np.vstack(padded)

    # Add overall title
    fig, ax = plt.subplots(figsize=(montage.shape[1] / dpi,
                                    montage.shape[0] / dpi + 0.4))
    ax.imshow(montage)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.0)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.close(fig)


# ── blend process montage (Figure-10 style: rows=levels, cols=A/B/blend) ────

def render_blend_process(details: dict,
                         title: str = "",
                         save_path: str | None = None,
                         elev: float = DEFAULT_ELEV,
                         azim: float = DEFAULT_AZIM,
                         cell_figsize: tuple = (2.5, 2.5),
                         dpi: int = 80) -> None:
    """Visualise the multiresolution blend as a grid of 3D projected views.

    Columns: masked-A | masked-B | blended
    Rows:    L0, L1, …, residual

    Parameters
    ----------
    details : dict returned by blend.multiresolution_blend
    """
    from laplacian3d import detail_to_viewable, to_uint8

    L_a = details["L_a"]
    L_b = details["L_b"]
    L_blend = details["L_blend"]
    res_a = details["res_a"]
    res_b = details["res_b"]
    res_blend = details["res_blend"]
    sigmas = details["sigmas"]
    mask_stack = details["mask_stack"]

    n_levels = len(L_a)
    n_rows = n_levels + 1  # detail levels + residual
    n_cols = 3             # A, B, blend

    def _render_detail(vol, label):
        v = detail_to_viewable(vol)
        return _render_voxel_to_array(v, title=label, elev=elev, azim=azim,
                                      figsize=cell_figsize, dpi=dpi)

    def _render_residual(vol, label):
        v = to_uint8(np.clip(vol, 0, 1))
        return _render_voxel_to_array(v, title=label, elev=elev, azim=azim,
                                      figsize=cell_figsize, dpi=dpi)

    # Build grid of rendered images
    grid_imgs = []  # grid_imgs[row][col] = RGB numpy array
    for i in range(n_levels):
        s_lo, s_hi = sigmas[i], sigmas[i + 1]
        row = [
            _render_detail(L_a[i], f"A  L{i}"),
            _render_detail(L_b[i], f"B  L{i}"),
            _render_detail(L_blend[i], f"blend  L{i}"),
        ]
        grid_imgs.append(row)

    # Residual row
    grid_imgs.append([
        _render_residual(res_a, "A residual"),
        _render_residual(res_b, "B residual"),
        _render_residual(res_blend, "blend residual"),
    ])

    # Pad all cells to same size and compose
    all_panels = [img for row in grid_imgs for img in row]
    max_h = max(p.shape[0] for p in all_panels)
    max_w = max(p.shape[1] for p in all_panels)

    rows_composed = []
    for row in grid_imgs:
        padded_row = []
        for p in row:
            canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            oh, ow = p.shape[:2]
            y0 = (max_h - oh) // 2
            x0 = (max_w - ow) // 2
            canvas[y0:y0 + oh, x0:x0 + ow] = p
            padded_row.append(canvas)
        rows_composed.append(np.hstack(padded_row))
    montage = np.vstack(rows_composed)

    fig, ax = plt.subplots(figsize=(montage.shape[1] / dpi,
                                    montage.shape[0] / dpi + 0.5))
    ax.imshow(montage)
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.0)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.close(fig)


# ── self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from voxel_io import find_asset, load_voxel_obj

    assets = "assets/VoxelFruitsAndVegetables"
    out = "output/renders"

    for name in ("apple", "orange"):
        path = find_asset(assets, name)
        grid, info = load_voxel_obj(path)
        print(f"{info['name']}: shape={info['shape']}  filled={info['n_filled']}")
        render_voxel_3d(grid, title=info["name"],
                        save_path=os.path.join(out, f"{info['name']}_3d.png"))
        for ax in range(3):
            render_slices(grid, axis=ax,
                          title=f"{info['name']} – {'XYZ'[ax]} slices",
                          save_path=os.path.join(out, f"{info['name']}_slices_{'XYZ'[ax]}.png"))
