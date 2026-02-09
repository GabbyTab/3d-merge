"""
laplacian2d.py – Per-slice 2-D Laplacian stack (regular image Laplacian).

For each 2-D slice of the volume along a chosen axis, build a standard
2-D Gaussian/Laplacian stack at increasing σ.  No downsampling – every
level stays at the original slice resolution.

This is deliberately kept separate from the volumetric 3-D stack in
``laplacian3d.py`` so the two decompositions can be compared and used
independently.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from laplacian3d import to_float, to_uint8, default_sigmas


# ── 2-D blur ────────────────────────────────────────────────────────────────

def blur2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 2-D Gaussian blur to each channel of a (H, W, C) image."""
    if sigma <= 0:
        return img.copy()
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        out[..., c] = gaussian_filter(img[..., c], sigma=sigma)
    return out


# ── per-slice stacks ────────────────────────────────────────────────────────

def build_slice_gaussian_stack(grid: np.ndarray,
                               axis: int = 1,
                               sigmas: list[float] | None = None,
                               n_levels: int = 5
                               ) -> tuple[list[list[np.ndarray]], list[float]]:
    """Gaussian stack applied independently to every 2-D slice.

    Parameters
    ----------
    grid : (X, Y, Z, 4) uint8 or float32 RGBA volume
    axis : slicing axis (0=X, 1=Y, 2=Z)
    sigmas : explicit σ schedule, or None for default_sigmas(n_levels)

    Returns
    -------
    G : list of lists – G[level][slice_idx] is a float32 (H, W, 4) image.
    sigmas : σ values used
    """
    if sigmas is None:
        sigmas = default_sigmas(n_levels)
    vol = to_float(grid)

    n_slices = vol.shape[axis]
    # G[level][slice]
    G = [[] for _ in sigmas]
    for i in range(n_slices):
        slc = np.take(vol, i, axis=axis)  # (A, B, 4)
        for li, s in enumerate(sigmas):
            G[li].append(blur2d(slc, s))
    return G, sigmas


def build_slice_laplacian_stack(grid: np.ndarray,
                                axis: int = 1,
                                sigmas: list[float] | None = None,
                                n_levels: int = 5
                                ) -> tuple[list[list[np.ndarray]],
                                           list[np.ndarray],
                                           list[float]]:
    """Laplacian (detail) stack applied independently to every 2-D slice.

    Returns
    -------
    L : list of lists – L[level][slice_idx] is a float32 (H, W, 4) detail.
    residuals : list of float32 slices – the most-blurred Gaussian for each slice.
    sigmas : σ values used
    """
    G, sigmas = build_slice_gaussian_stack(grid, axis, sigmas, n_levels)
    n_detail = len(sigmas) - 1
    n_slices = len(G[0])

    L = [[] for _ in range(n_detail)]
    for si in range(n_slices):
        for li in range(n_detail):
            L[li].append(G[li][si] - G[li + 1][si])

    residuals = G[-1]  # list of slices at the coarsest blur
    return L, residuals, sigmas


def reconstruct_slice_stack(L: list[list[np.ndarray]],
                            residuals: list[np.ndarray]) -> list[np.ndarray]:
    """Reconstruct original slices from Laplacian stack + residuals."""
    n_slices = len(residuals)
    out = [r.copy() for r in residuals]
    for level in L:
        for si in range(n_slices):
            out[si] = out[si] + level[si]
    return out
