#!/usr/bin/env python3
"""
viewer.py – Interactive 3-D voxel viewer using pyvista.

Launch from CLI:
    python viewer.py output/blend/Apple-0+Orange-0/result.npy
    python viewer.py result.npy --alpha-thresh 128

Or call from Python:
    from viewer import view_voxels
    view_voxels(grid)           # (X, Y, Z, 4) uint8 RGBA
"""

import argparse
import sys
import numpy as np


def view_voxels(grid: np.ndarray,
                alpha_threshold: int = 0,
                title: str = "Voxel Viewer",
                window_size: tuple = (1024, 768)) -> None:
    """Open an interactive 3-D window showing the voxel grid.

    Parameters
    ----------
    grid : (X, Y, Z, 4) uint8 RGBA volume
    alpha_threshold : voxels with alpha <= this are hidden
    title : window title
    window_size : (width, height) in pixels
    """
    import pyvista as pv

    # Find filled voxels
    filled_mask = grid[..., 3] > alpha_threshold
    coords = np.argwhere(filled_mask)  # (N, 3) array of (x, y, z)

    if len(coords) == 0:
        print("No visible voxels above threshold.")
        return

    # Build one cube per voxel using pyvista
    colors = grid[filled_mask]  # (N, 4) uint8 RGBA

    # Create a point cloud at voxel centres, then use glyphs to make cubes
    points = coords.astype(np.float64) + 0.5  # centre of each voxel cell
    cloud = pv.PolyData(points)

    # Attach RGB colour (pyvista uses 0-255 uint8 RGB)
    cloud["rgba"] = colors

    # Create a unit cube glyph
    cube = pv.Cube(center=(0, 0, 0), x_length=0.95, y_length=0.95, z_length=0.95)

    # Glyph each point with the cube
    voxels = cloud.glyph(geom=cube, orient=False, scale=False)

    # Transfer colours: glyphing repeats point data per glyph vertex
    # We need to map the per-point rgba to per-cell colours
    n_cells_per_cube = cube.n_cells
    n_points_total = len(coords)
    cell_colors = np.repeat(colors[:, :3], n_cells_per_cube, axis=0)
    voxels.cell_data["RGB"] = cell_colors

    # Set up the plotter
    pl = pv.Plotter(title=title, window_size=window_size)
    pl.add_mesh(voxels, scalars="RGB", rgb=True,
                show_edges=True, edge_color=(0.2, 0.2, 0.2),
                line_width=0.5)
    pl.set_background("white")

    # Set nice initial camera
    shape = np.array(grid.shape[:3], dtype=float)
    centre = shape / 2
    pl.camera.focal_point = tuple(centre)
    pl.camera.position = tuple(centre + shape.max() * np.array([1.2, -0.8, 0.9]))
    pl.camera.up = (0, 0, 1)

    pl.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D voxel viewer")
    parser.add_argument("path", help="path to .npy file (X, Y, Z, 4) uint8 RGBA")
    parser.add_argument("--alpha-thresh", type=int, default=0,
                        help="hide voxels with alpha <= this (default 0)")
    parser.add_argument("--title", type=str, default="Voxel Viewer")
    args = parser.parse_args()

    grid = np.load(args.path)
    print(f"Loaded {args.path}: shape={grid.shape}, dtype={grid.dtype}")
    print(f"  filled (alpha>{args.alpha_thresh}): "
          f"{(grid[..., 3] > args.alpha_thresh).sum()}")

    view_voxels(grid, alpha_threshold=args.alpha_thresh, title=args.title)


if __name__ == "__main__":
    main()
