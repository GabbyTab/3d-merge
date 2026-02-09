"""
voxel_io.py – Load voxel OBJ assets into dense RGBA numpy grids.

Each voxel model is an axis-aligned cube mesh exported from MagicaVoxel.
The OBJ encodes only the *visible* faces; we reconstruct the filled voxel
grid by offsetting each face centroid inward along its normal.

Colour comes from the palette PNG referenced by the MTL file.
"""

import os
import numpy as np
from PIL import Image


# ── OBJ parsing helpers ─────────────────────────────────────────────────────

def _parse_obj(path: str):
    """Return (verts, vts, vns, faces) from an OBJ file.

    Each face entry is a list of (vertex_idx, vt_idx, vn_idx) tuples (0-based).
    """
    verts, vts, vns, faces = [], [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif tag == "vt":
                vts.append([float(x) for x in parts[1:3]])
            elif tag == "vn":
                vns.append([float(x) for x in parts[1:4]])
            elif tag == "f":
                tri = []
                for p in parts[1:]:
                    a, b, c = p.split("/")
                    tri.append((int(a) - 1, int(b) - 1, int(c) - 1))
                faces.append(tri)
    return np.array(verts), np.array(vts), np.array(vns), faces


def _load_palette(png_path: str) -> np.ndarray:
    """Load palette PNG → (N, 4) uint8 RGBA array."""
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)
    # Palette is a 1-pixel-tall strip (1, W, 4)
    assert arr.shape[0] == 1, f"Expected 1-row palette, got {arr.shape}"
    return arr[0]  # (W, 4)


def _vt_to_palette_index(vt_u: float, palette_width: int) -> int:
    """Map a texture-coordinate u ∈ [0,1] to a palette pixel index."""
    return min(int(round(vt_u * (palette_width - 1))), palette_width - 1)


# ── Public API ───────────────────────────────────────────────────────────────

GRID_SPACING = 0.1  # voxel edge length used in the OBJ exports


def load_voxel_obj(obj_path: str) -> tuple[np.ndarray, dict]:
    """Load a single voxel OBJ asset into a dense RGBA grid.

    Returns
    -------
    grid : np.ndarray, shape (X, Y, Z, 4), dtype uint8
        RGBA voxel volume.  Empty voxels have alpha = 0.
    info : dict
        Metadata: 'origin' (world-space min corner), 'spacing', 'shape',
        'name', 'n_filled'.
    """
    obj_dir = os.path.dirname(obj_path)
    name = os.path.splitext(os.path.basename(obj_path))[0]
    png_path = os.path.join(obj_dir, name + ".png")

    verts, vts, vns, faces = _parse_obj(obj_path)
    palette = _load_palette(png_path)
    pw = len(palette)

    sp = GRID_SPACING

    # --- Collect voxel centres and colours from faces -------------------------
    voxel_map = {}  # (ix, iy, iz) → RGBA
    for face in faces:
        vi = [f[0] for f in face]
        vti = face[0][1]
        vni = face[0][2]

        centroid = verts[vi].mean(axis=0)
        normal = vns[vni]

        # Offset inward by half a voxel to reach the voxel centre
        centre = centroid - 0.5 * sp * normal

        # Snap to grid indices (relative to an arbitrary origin we'll normalise)
        # We store raw rounded positions; normalisation happens below.
        ix = round(centre[0] / sp)
        iy = round(centre[1] / sp)
        iz = round(centre[2] / sp)

        if (ix, iy, iz) not in voxel_map:
            color_idx = _vt_to_palette_index(vts[vti, 0], pw)
            voxel_map[(ix, iy, iz)] = palette[color_idx]

    if not voxel_map:
        empty = np.zeros((1, 1, 1, 4), dtype=np.uint8)
        return empty, {"origin": np.zeros(3), "spacing": sp,
                       "shape": (1, 1, 1), "name": name, "n_filled": 0}

    keys = np.array(list(voxel_map.keys()))
    mins = keys.min(axis=0)
    maxs = keys.max(axis=0)
    shape = tuple((maxs - mins + 1).astype(int))

    grid = np.zeros((*shape, 4), dtype=np.uint8)
    for (ix, iy, iz), rgba in voxel_map.items():
        gi = int(ix - mins[0])
        gj = int(iy - mins[1])
        gk = int(iz - mins[2])
        grid[gi, gj, gk] = rgba

    origin = mins * sp
    info = {
        "origin": origin,
        "spacing": sp,
        "shape": shape,
        "name": name,
        "n_filled": len(voxel_map),
    }
    return grid, info


def upsample_nn(grid: np.ndarray, factor: int = 4) -> np.ndarray:
    """Nearest-neighbour upsample: repeat each voxel factor×factor×factor.

    Parameters
    ----------
    grid : (X, Y, Z, 4) uint8 RGBA
    factor : integer upsampling factor per spatial axis

    Returns
    -------
    upsampled : (X*f, Y*f, Z*f, 4) uint8 RGBA
    """
    return np.repeat(
        np.repeat(
            np.repeat(grid, factor, axis=0),
            factor, axis=1),
        factor, axis=2)


def upsample_smooth(grid: np.ndarray, factor: int = 4,
                     alpha_threshold: float = 0.5) -> np.ndarray:
    """Smooth (trilinear) upsample that rounds off blocky edges.

    Uses scipy.ndimage.zoom with order=1 (trilinear interpolation) on
    each RGBA channel.  The interpolation creates smooth gradients at
    voxel boundaries.  After interpolation, voxels with alpha above
    ``alpha_threshold`` are snapped to fully opaque; the rest are cleared.

    Parameters
    ----------
    grid : (X, Y, Z, 4) uint8 RGBA
    factor : upsampling factor per spatial axis
    alpha_threshold : controls where the rounded surface is carved.
                      Lower = larger/rounder, higher = tighter to original.

    Returns
    -------
    upsampled : (X*f, Y*f, Z*f, 4) uint8 RGBA – smooth, solid voxels
    """
    from scipy.ndimage import zoom

    # Interpolate in float space
    vol = grid.astype(np.float32) / 255.0
    zoomed = zoom(vol, [factor, factor, factor, 1], order=1)

    # Threshold alpha to carve the smooth surface
    solid = zoomed[..., 3] >= alpha_threshold
    zoomed[~solid] = 0.0
    zoomed[solid, 3] = 1.0  # snap to fully opaque

    return np.clip(zoomed * 255, 0, 255).astype(np.uint8)


def fill_interior(grid: np.ndarray) -> np.ndarray:
    """Fill the hollow interior of a voxel shell to make it solid.

    The OBJ exports from MagicaVoxel only contain visible surface faces,
    so parsed models are hollow shells.  This function:
      1. Dilates the shell by 1 voxel to close checkerboard gaps
      2. Fills enclosed regions slice-by-slice from all 3 axes
      3. Intersects the three fills (a voxel is "inside" only if enclosed
         from every viewing direction)
      4. Colours new interior voxels by nearest-surface-voxel lookup

    Parameters
    ----------
    grid : (X, Y, Z, 4) uint8 RGBA – the hollow shell

    Returns
    -------
    filled : (X, Y, Z, 4) uint8 RGBA – solid interior
    """
    from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt

    mask = grid[..., 3] > 0

    # 1. Dilate to close 1-voxel checkerboard gaps in the shell
    dilated = binary_dilation(mask, iterations=1)

    # 2. Fill slice-by-slice along each axis, then intersect
    fills = []
    for axis in range(3):
        filled_ax = np.zeros_like(dilated)
        for i in range(dilated.shape[axis]):
            slc = np.take(dilated, i, axis=axis)
            idx = [slice(None)] * 3
            idx[axis] = i
            filled_ax[tuple(idx)] = binary_fill_holes(slc)
        fills.append(filled_ax)
    solid = fills[0] & fills[1] & fills[2]

    # 3. Identify new interior voxels that need colouring
    new_interior = solid & ~mask
    if not new_interior.any():
        return grid.copy()

    # 4. Colour interior voxels from nearest surface voxel
    #    Use distance transform to find the index of the closest surface voxel
    out = grid.copy()
    # Invert mask: distance from each point to nearest surface voxel
    dist, indices = distance_transform_edt(~mask, return_indices=True)
    # indices[axis][x,y,z] gives the coordinate of the nearest surface voxel
    new_coords = np.argwhere(new_interior)
    for x, y, z in new_coords:
        sx, sy, sz = indices[0][x, y, z], indices[1][x, y, z], indices[2][x, y, z]
        out[x, y, z] = grid[sx, sy, sz]

    n_new = new_interior.sum()
    n_orig = mask.sum()
    print(f"  fill_interior: {n_orig} surface + {n_new} interior = {n_orig + n_new} total")
    return out


def list_assets(assets_dir: str) -> list[str]:
    """Return sorted list of OBJ paths under the assets directory."""
    objs_dir = os.path.join(assets_dir, "OBJs")
    paths = sorted(
        os.path.join(objs_dir, f)
        for f in os.listdir(objs_dir)
        if f.endswith(".obj")
    )
    return paths


def find_asset(assets_dir: str, name: str) -> str:
    """Find asset OBJ by (case-insensitive) name prefix, e.g. 'apple'."""
    name_lower = name.lower()
    for p in list_assets(assets_dir):
        basename = os.path.splitext(os.path.basename(p))[0].lower()
        if basename.startswith(name_lower):
            return p
    raise FileNotFoundError(f"No asset matching '{name}' in {assets_dir}")


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    assets = sys.argv[1] if len(sys.argv) > 1 else "assets/VoxelFruitsAndVegetables"

    print("Available assets:")
    for p in list_assets(assets):
        print(f"  {os.path.basename(p)}")

    for name in ("apple", "orange"):
        path = find_asset(assets, name)
        grid, info = load_voxel_obj(path)
        print(f"\n{info['name']}:")
        print(f"  grid shape : {info['shape']}")
        print(f"  filled     : {info['n_filled']}")
        print(f"  origin     : {info['origin']}")
        # Show unique colours (ignoring empty voxels)
        filled = grid[grid[..., 3] > 0].reshape(-1, 4)
        unique_colors = np.unique(filled, axis=0)
        print(f"  colours    : {len(unique_colors)}")
        for c in unique_colors:
            print(f"    RGBA {tuple(c)}")
