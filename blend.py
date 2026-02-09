"""
blend.py – Multiresolution blending of two voxel volumes (3-D oraple).

Implements the Burt & Adelson (1983) algorithm extended to 3-D:

    For each Laplacian level i:
        blended_L[i] = mask_G[i] * A_L[i] + (1 - mask_G[i]) * B_L[i]

    blended_residual  = mask_G[-1] * A_res + (1 - mask_G[-1]) * B_res

    result = reconstruct(blended_L, blended_residual)

The mask is a single-channel (X, Y, Z) volume in [0, 1].  Its Gaussian
stack is built at the same σ schedule as the images, giving progressively
smoother transitions at coarser frequency bands.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from laplacian3d import (
    to_float, to_uint8,
    build_laplacian_stack,
    reconstruct_from_stack,
    default_sigmas,
)


# ── padding to common size ───────────────────────────────────────────────────

def pad_to_common(grid_a: np.ndarray,
                  grid_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre-pad two RGBA grids so they share the same spatial shape.

    Each grid is padded with zeros (empty voxels) on both sides of each axis.
    """
    sa = np.array(grid_a.shape[:3])
    sb = np.array(grid_b.shape[:3])
    target = np.maximum(sa, sb)

    def _pad(grid, shape):
        s = np.array(grid.shape[:3])
        diff = target - s
        before = diff // 2
        after = diff - before
        pads = [(b, a) for b, a in zip(before, after)] + [(0, 0)]
        return np.pad(grid, pads, mode="constant", constant_values=0)

    return _pad(grid_a, sa), _pad(grid_b, sb)


# ── mask helpers ─────────────────────────────────────────────────────────────

def make_half_mask(shape: tuple[int, int, int],
                   axis: int = 0,
                   smooth: bool = False) -> np.ndarray:
    """Create a binary half-split mask.

    Parameters
    ----------
    shape : (X, Y, Z) spatial dimensions
    axis : split axis (0=X, 1=Y, 2=Z)
    smooth : if True, use a linear ramp over the middle 2 voxels
             instead of a hard step

    Returns
    -------
    mask : float32 (X, Y, Z) with values in [0, 1]
           1 = image A, 0 = image B
    """
    mask = np.zeros(shape, dtype=np.float32)
    mid = shape[axis] // 2
    # Set the first half to 1
    slc = [slice(None)] * 3
    slc[axis] = slice(0, mid)
    mask[tuple(slc)] = 1.0
    if smooth and shape[axis] > 2:
        # Linear ramp over the boundary voxel
        slc[axis] = mid
        mask[tuple(slc)] = 0.5
    return mask


def build_mask_gaussian_stack(mask: np.ndarray,
                              sigmas: list[float]) -> list[np.ndarray]:
    """Build a Gaussian stack of the mask volume.

    Parameters
    ----------
    mask : (X, Y, Z) float32, values in [0, 1]
    sigmas : σ schedule (same as used for the image stacks)

    Returns
    -------
    M_G : list of (X, Y, Z) float32 blurred masks, one per sigma level
    """
    M_G = []
    for s in sigmas:
        if s <= 0:
            M_G.append(mask.copy())
        else:
            M_G.append(gaussian_filter(mask, sigma=s))
    return M_G


# ── core blending ────────────────────────────────────────────────────────────

def multiresolution_blend(grid_a: np.ndarray,
                          grid_b: np.ndarray,
                          mask: np.ndarray,
                          sigmas: list[float] | None = None,
                          n_levels: int = 5,
                          alpha_threshold: float = 0.5,
                          ) -> tuple[np.ndarray, dict]:
    """Blend two voxel volumes using multiresolution blending.

    Shape and colour are blended separately:
      - **Colour** (RGB) goes through the full Laplacian multiresolution
        pipeline for a seamless frequency-aware transition.
      - **Shape** (alpha / occupancy) is blended directly using the most-
        blurred mask level, then thresholded.  This prevents the Gaussian
        blur in the Laplacian stack from inflating the shape far beyond
        the original volumes.

    Parameters
    ----------
    grid_a, grid_b : (X, Y, Z, 4) uint8 RGBA volumes (same shape)
    mask : (X, Y, Z) float32 in [0, 1].  1 = take from A, 0 = take from B.
    sigmas : σ schedule, or None for default
    n_levels : used only when sigmas is None
    alpha_threshold : controls the blended shape boundary.

    Returns
    -------
    result : (X, Y, Z, 4) uint8 RGBA blended volume
    details : dict with intermediate results for visualisation
    """
    assert grid_a.shape == grid_b.shape, \
        f"Shape mismatch: {grid_a.shape} vs {grid_b.shape}"
    assert mask.shape == grid_a.shape[:3], \
        f"Mask shape {mask.shape} != spatial shape {grid_a.shape[:3]}"

    if sigmas is None:
        sigmas = default_sigmas(n_levels)

    vol_a = to_float(grid_a)
    vol_b = to_float(grid_b)

    # ── Shape blending (alpha channel) ────────────────────────────────────
    # Blend occupancy directly with the most-blurred mask.  This keeps the
    # shape close to the union of both inputs instead of inflating it.
    shape_a = (vol_a[..., 3] > 0).astype(np.float32)
    shape_b = (vol_b[..., 3] > 0).astype(np.float32)
    mask_coarse = gaussian_filter(mask, sigma=sigmas[-1])
    blended_shape = mask_coarse * shape_a + (1 - mask_coarse) * shape_b
    solid_mask = blended_shape >= alpha_threshold

    # ── Colour blending (RGB channels only) ───────────────────────────────
    # Build RGB-only volumes (3 channels), using the Laplacian pipeline
    rgb_a = vol_a[..., :3]
    rgb_b = vol_b[..., :3]

    # Build Laplacian stacks for RGB
    from laplacian3d import blur3d
    def _build_rgb_stack(rgb, sigmas):
        G = [blur3d(np.concatenate([rgb, np.ones(rgb.shape[:3]+(1,), dtype=np.float32)], axis=-1), s)[..., :3]
             if s > 0 else rgb.copy()
             for s in sigmas]
        # Actually, just blur RGB directly
        G2 = []
        for s in sigmas:
            if s <= 0:
                G2.append(rgb.copy())
            else:
                out = np.empty_like(rgb)
                for c in range(3):
                    out[..., c] = gaussian_filter(rgb[..., c], sigma=s)
                G2.append(out)
        L = [G2[i] - G2[i+1] for i in range(len(G2)-1)]
        return L, G2[-1], G2

    L_a, res_a_rgb, _ = _build_rgb_stack(rgb_a, sigmas)
    L_b, res_b_rgb, _ = _build_rgb_stack(rgb_b, sigmas)

    # Build Gaussian stack of the mask
    M_G = build_mask_gaussian_stack(mask, sigmas)

    # Blend each Laplacian level (RGB only)
    L_blend = []
    for i in range(len(L_a)):
        m = M_G[i][..., None]  # (X, Y, Z, 1)
        L_blend.append(m * L_a[i] + (1 - m) * L_b[i])

    # Blend residual
    m_last = M_G[-1][..., None]
    res_blend_rgb = m_last * res_a_rgb + (1 - m_last) * res_b_rgb

    # Reconstruct RGB
    rgb_result = res_blend_rgb.copy()
    for L in L_blend:
        rgb_result = rgb_result + L

    # ── Combine: blended RGB + blended shape ──────────────────────────────
    result_float = np.zeros((*vol_a.shape[:3], 4), dtype=np.float32)
    result_float[..., :3] = np.clip(rgb_result, 0, 1)
    result_float[solid_mask, 3] = 1.0
    result_float[~solid_mask] = 0.0

    result = to_uint8(result_float)

    # Build details dict for visualisation (wrap RGB layers as RGBA)
    def _rgb_to_rgba(rgb_layer):
        alpha = np.ones(rgb_layer.shape[:3] + (1,), dtype=np.float32)
        return np.concatenate([rgb_layer, alpha], axis=-1)

    L_a_rgba = [_rgb_to_rgba(l) for l in L_a]
    L_b_rgba = [_rgb_to_rgba(l) for l in L_b]
    L_blend_rgba = [_rgb_to_rgba(l) for l in L_blend]

    details = {
        "sigmas": sigmas,
        "mask_stack": M_G,
        "L_a": L_a_rgba, "L_b": L_b_rgba, "L_blend": L_blend_rgba,
        "res_a": _rgb_to_rgba(res_a_rgb),
        "res_b": _rgb_to_rgba(res_b_rgb),
        "res_blend": _rgb_to_rgba(res_blend_rgb),
        "result_float": result_float,
    }
    return result, details


# ── naive blend for comparison ───────────────────────────────────────────────

def naive_blend(grid_a: np.ndarray,
                grid_b: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Simple alpha-blend with the raw (un-blurred) mask, for comparison."""
    a = to_float(grid_a)
    b = to_float(grid_b)
    m = mask[..., None]
    result = m * a + (1 - m) * b
    return to_uint8(np.clip(result, 0, 1))
