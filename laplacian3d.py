"""
laplacian3d.py – 3-D Laplacian volume **stack**.

Unlike a pyramid (blur + downsample), a stack keeps every level at the
original resolution and just increases the blur σ.  This gives as many
fine detail layers as we want regardless of grid size.

    G_i  =  gaussian_filter(original, σ_i)      Gaussian stack
    L_i  =  G_i - G_{i+1}                       Laplacian (detail) stack
    residual = G_{n}                             coarsest Gaussian

Reconstruction:  sum(L_i) + residual == original  (exact).

The volume is treated as float32 RGBA (X, Y, Z, 4) in [0, 1].
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ── helpers (shared with other modules) ─────────────────────────────────────

def to_float(grid: np.ndarray) -> np.ndarray:
    """uint8 RGBA → float32 [0, 1]."""
    if grid.dtype == np.uint8:
        return grid.astype(np.float32) / 255.0
    return grid.astype(np.float32)


def to_uint8(grid: np.ndarray) -> np.ndarray:
    """float32 → uint8, clipped to [0, 255]."""
    return np.clip(grid * 255, 0, 255).astype(np.uint8)


def default_sigmas(n_levels: int = 5) -> list[float]:
    """Return a default σ schedule: doubling from 0.25.

    Produces n_levels+1 values (including σ=0 for the original).
    Example (n_levels=5):  [0, 0.25, 0.5, 1.0, 2.0, 4.0]
    """
    sigmas = [0.0]
    s = 0.25
    for _ in range(n_levels):
        sigmas.append(s)
        s *= 2.0
    return sigmas


# ── 3-D blur ────────────────────────────────────────────────────────────────

def blur3d(vol: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 3-D Gaussian blur to each channel independently.

    Parameters
    ----------
    vol : (X, Y, Z, C) float32 – typically C=4 RGBA
    sigma : standard deviation of the Gaussian kernel
    """
    if sigma <= 0:
        return vol.copy()
    out = np.empty_like(vol)
    for c in range(vol.shape[-1]):
        out[..., c] = gaussian_filter(vol[..., c], sigma=sigma)
    return out


# ── Gaussian & Laplacian stacks ─────────────────────────────────────────────

def build_gaussian_stack(grid: np.ndarray,
                         sigmas: list[float] | None = None,
                         n_levels: int = 5) -> tuple[list[np.ndarray], list[float]]:
    """Build a Gaussian stack at full resolution.

    Parameters
    ----------
    grid : (X, Y, Z, 4) uint8 or float32 RGBA
    sigmas : explicit list of σ values (including 0 for the original).
             If None, uses ``default_sigmas(n_levels)``.
    n_levels : used only when sigmas is None

    Returns
    -------
    G : list of float32 volumes [G_σ₀, G_σ₁, …]
    sigmas : the σ values used
    """
    if sigmas is None:
        sigmas = default_sigmas(n_levels)
    vol = to_float(grid)
    G = [blur3d(vol, s) for s in sigmas]
    return G, sigmas


def build_laplacian_stack(grid: np.ndarray,
                          sigmas: list[float] | None = None,
                          n_levels: int = 5
                          ) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    """Build a 3-D Laplacian (detail) stack.

    Parameters
    ----------
    grid : (X, Y, Z, 4) uint8 or float32 RGBA
    sigmas : explicit list of σ values (including 0)
    n_levels : used only when sigmas is None

    Returns
    -------
    L : list of float32 detail volumes [L₀, L₁, …]
        L_i = G_σᵢ - G_σᵢ₊₁.  Values can be negative.
    residual : float32 – the most-blurred Gaussian level G_σₙ
    sigmas : σ values used
    """
    G, sigmas = build_gaussian_stack(grid, sigmas, n_levels)
    L = [G[i] - G[i + 1] for i in range(len(G) - 1)]
    residual = G[-1]
    return L, residual, sigmas


def reconstruct_from_stack(L: list[np.ndarray],
                           residual: np.ndarray) -> np.ndarray:
    """Reconstruct original from Laplacian stack + residual (exact)."""
    return sum(L, residual.copy())


# ── visualisation helpers ───────────────────────────────────────────────────

def detail_to_viewable(detail: np.ndarray) -> np.ndarray:
    """Convert a signed detail layer → viewable uint8 RGBA.

    RGB: maps [-max, +max] → [0, 255] (0.5 = zero detail).
    Alpha: proportional to |detail| magnitude.
    """
    rgb = detail[..., :3]
    amax = np.abs(rgb).max()
    if amax > 0:
        rgb_norm = rgb / (2 * amax) + 0.5
    else:
        rgb_norm = np.full_like(rgb, 0.5)

    mag = np.abs(detail).max(axis=-1)
    alpha_max = mag.max()
    alpha = mag / alpha_max if alpha_max > 0 else np.zeros_like(mag)

    out = np.concatenate([rgb_norm, alpha[..., None]], axis=-1)
    return to_uint8(out)
