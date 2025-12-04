#!/usr/bin/env python3
"""
PLY Height Adjustment and Visualization Tool

This script adjusts the z-coordinates of points within specified rectangular
regions and provides visualization with height-difference coloring and a
reference plane.

Usage:
    python adjust_height.py --config config.yaml
    python adjust_height.py --config config.yaml --no-save
    python adjust_height.py --config config.yaml --no-vis
"""

import argparse
import yaml
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import sys
from pathlib import Path


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AdjustmentRegion:
    """Defines a rectangular region with height adjustment."""
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    z_offset: float


@dataclass
class PgmConfig:
    """PGM image projection settings."""
    enabled: bool = False
    yaml_path: str = ""  # Path to ROS map.yaml (auto-loads image, resolution, origin)
    path: str = ""  # Direct path to PGM (overridden if yaml_path is set)
    resolution: float = 0.05  # meters per pixel
    origin_x: float = 0.0  # world X coordinate of image origin
    origin_y: float = 0.0  # world Y coordinate of image origin
    auto_align: bool = True  # Auto-align PGM center to PLY center
    rotate_180: bool = False  # Rotate 180 degrees (auto-detected if auto_align)
    offset_x: float = 0.0  # Additional X offset (auto-computed if auto_align)
    offset_y: float = 0.0  # Additional Y offset (auto-computed if auto_align)
    alpha: float = 1.0  # Opacity (0.0 = transparent/white, 1.0 = fully visible)


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    enabled: bool = True
    colormap: str = "coolwarm"
    color_range: Tuple[float, float] = (-2.0, 2.0)
    downsample_voxel_size: Optional[float] = 0.05
    show_region_bounds: bool = True
    pgm: PgmConfig = field(default_factory=PgmConfig)


@dataclass
class Config:
    """Main configuration."""
    input_ply: str
    output_ply: str
    reference_height: float = 0.0
    regions: List[AdjustmentRegion] = field(default_factory=list)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# ============================================================================
# MAP.YAML LOADING AND AUTO-ALIGNMENT
# ============================================================================

def load_map_yaml(yaml_path: str) -> dict:
    """
    Load ROS map.yaml and return PGM configuration.

    Args:
        yaml_path: Path to the map.yaml file

    Returns:
        Dictionary with image_path, resolution, origin_x, origin_y, origin_yaw
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    yaml_dir = Path(yaml_path).parent
    image_name = data['image']

    # Handle relative or absolute image path
    if Path(image_name).is_absolute():
        image_path = image_name
    else:
        image_path = str(yaml_dir / image_name)

    origin = data.get('origin', [0.0, 0.0, 0.0])

    return {
        'image_path': image_path,
        'resolution': data['resolution'],
        'origin_x': origin[0],
        'origin_y': origin[1],
        'origin_yaw': origin[2] if len(origin) > 2 else 0.0
    }


def compute_pgm_bounds(pgm_config: PgmConfig, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Compute PGM world coordinate bounds.

    Returns:
        (xmin, xmax, ymin, ymax) in world coordinates
    """
    img_width_m = img_width * pgm_config.resolution
    img_height_m = img_height * pgm_config.resolution

    xmin = pgm_config.origin_x
    xmax = pgm_config.origin_x + img_width_m
    ymin = pgm_config.origin_y
    ymax = pgm_config.origin_y + img_height_m

    return xmin, xmax, ymin, ymax


def compute_overlap_area(bounds1: Tuple[float, float, float, float],
                         bounds2: Tuple[float, float, float, float]) -> float:
    """
    Compute overlap area between two bounding boxes.

    Args:
        bounds1, bounds2: (xmin, xmax, ymin, ymax)

    Returns:
        Overlap area (0 if no overlap)
    """
    x_overlap = max(0, min(bounds1[1], bounds2[1]) - max(bounds1[0], bounds2[0]))
    y_overlap = max(0, min(bounds1[3], bounds2[3]) - max(bounds1[2], bounds2[2]))
    return x_overlap * y_overlap


def detect_rotation_needed(ply_bounds: Tuple[float, float, float, float],
                           pgm_bounds: Tuple[float, float, float, float]) -> bool:
    """
    Detect if 180-degree rotation is needed by comparing overlap areas.

    Args:
        ply_bounds: PLY bounding box (xmin, xmax, ymin, ymax)
        pgm_bounds: PGM bounding box (xmin, xmax, ymin, ymax)

    Returns:
        True if 180-degree rotation improves alignment
    """
    # Compute centers
    ply_cx = (ply_bounds[0] + ply_bounds[1]) / 2
    ply_cy = (ply_bounds[2] + ply_bounds[3]) / 2
    pgm_cx = (pgm_bounds[0] + pgm_bounds[1]) / 2
    pgm_cy = (pgm_bounds[2] + pgm_bounds[3]) / 2
    pgm_w = pgm_bounds[1] - pgm_bounds[0]
    pgm_h = pgm_bounds[3] - pgm_bounds[2]

    # Offset to align centers (no rotation)
    offset_x_no_rot = ply_cx - pgm_cx
    offset_y_no_rot = ply_cy - pgm_cy
    aligned_no_rot = (
        pgm_bounds[0] + offset_x_no_rot,
        pgm_bounds[1] + offset_x_no_rot,
        pgm_bounds[2] + offset_y_no_rot,
        pgm_bounds[3] + offset_y_no_rot
    )
    overlap_no_rot = compute_overlap_area(ply_bounds, aligned_no_rot)

    # 180-degree rotated PGM bounds (rotate around origin, then align centers)
    # After 180 rotation: new_xmin = -old_xmax, new_xmax = -old_xmin, etc.
    pgm_rotated = (
        -pgm_bounds[1],  # new xmin = -old xmax
        -pgm_bounds[0],  # new xmax = -old xmin
        -pgm_bounds[3],  # new ymin = -old ymax
        -pgm_bounds[2]   # new ymax = -old ymin
    )
    pgm_rot_cx = (pgm_rotated[0] + pgm_rotated[1]) / 2
    pgm_rot_cy = (pgm_rotated[2] + pgm_rotated[3]) / 2
    offset_x_rot = ply_cx - pgm_rot_cx
    offset_y_rot = ply_cy - pgm_rot_cy
    aligned_rot = (
        pgm_rotated[0] + offset_x_rot,
        pgm_rotated[1] + offset_x_rot,
        pgm_rotated[2] + offset_y_rot,
        pgm_rotated[3] + offset_y_rot
    )
    overlap_rot = compute_overlap_area(ply_bounds, aligned_rot)

    print(f"  Rotation detection:")
    print(f"    No rotation overlap: {overlap_no_rot:.1f} m^2")
    print(f"    180-deg rotation overlap: {overlap_rot:.1f} m^2")

    return overlap_rot > overlap_no_rot


def compute_auto_alignment(ply_bounds: Tuple[float, float, float, float],
                           pgm_center: Tuple[float, float],
                           rotate_180: bool) -> Tuple[float, float]:
    """
    Compute offset to align PGM center to PLY center.

    Args:
        ply_bounds: PLY bounding box (xmin, xmax, ymin, ymax)
        pgm_center: PGM center (cx, cy) - can be occupied cell center
        rotate_180: Whether 180-degree rotation is applied

    Returns:
        (offset_x, offset_y) to apply to PGM coordinates
    """
    ply_cx = (ply_bounds[0] + ply_bounds[1]) / 2
    ply_cy = (ply_bounds[2] + ply_bounds[3]) / 2

    if rotate_180:
        # After rotation, coordinates are negated
        pgm_cx = -pgm_center[0]
        pgm_cy = -pgm_center[1]
    else:
        pgm_cx = pgm_center[0]
        pgm_cy = pgm_center[1]

    offset_x = ply_cx - pgm_cx
    offset_y = ply_cy - pgm_cy

    return offset_x, offset_y


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    regions = [
        AdjustmentRegion(**r) for r in data.get('regions', [])
    ]

    vis_data = data.get('visualization', {})

    # Load PGM config
    pgm_data = vis_data.get('pgm', {})

    # Check if yaml_path is specified (auto-load from ROS map.yaml)
    yaml_path = pgm_data.get('yaml_path', '')
    if yaml_path:
        print(f"Loading map settings from: {yaml_path}")
        map_info = load_map_yaml(yaml_path)
        pgm_path = map_info['image_path']
        resolution = map_info['resolution']
        origin_x = map_info['origin_x']
        origin_y = map_info['origin_y']
        print(f"  Image: {pgm_path}")
        print(f"  Resolution: {resolution} m/pixel")
        print(f"  Origin: ({origin_x:.2f}, {origin_y:.2f})")
    else:
        pgm_path = pgm_data.get('path', '')
        resolution = pgm_data.get('resolution', 0.05)
        origin_x = pgm_data.get('origin_x', 0.0)
        origin_y = pgm_data.get('origin_y', 0.0)

    pgm_config = PgmConfig(
        enabled=pgm_data.get('enabled', False),
        yaml_path=yaml_path,
        path=pgm_path,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        auto_align=pgm_data.get('auto_align', True),
        rotate_180=pgm_data.get('rotate_180', False),
        offset_x=pgm_data.get('offset_x', 0.0),
        offset_y=pgm_data.get('offset_y', 0.0),
        alpha=pgm_data.get('alpha', 1.0)
    )

    vis_config = VisualizationConfig(
        enabled=vis_data.get('enabled', True),
        colormap=vis_data.get('colormap', 'coolwarm'),
        color_range=tuple(vis_data.get('color_range', [-2.0, 2.0])),
        downsample_voxel_size=vis_data.get('downsample_voxel_size', 0.05),
        show_region_bounds=vis_data.get('show_region_bounds', True),
        pgm=pgm_config
    )

    return Config(
        input_ply=data['input_ply'],
        output_ply=data['output_ply'],
        reference_height=data.get('reference_height', 0.0),
        regions=regions,
        visualization=vis_config
    )


# ============================================================================
# POINT CLOUD I/O
# ============================================================================

def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load PLY point cloud file."""
    print(f"Loading: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud from {filepath}")
    return pcd


def save_point_cloud(pcd: o3d.geometry.PointCloud, filepath: str) -> None:
    """Save point cloud to PLY file (binary format)."""
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_point_cloud(filepath, pcd, write_ascii=False)
    if not success:
        raise IOError(f"Failed to save point cloud to {filepath}")
    print(f"Saved: {filepath}")


# ============================================================================
# HEIGHT ADJUSTMENT
# ============================================================================

def get_region_mask(points: np.ndarray, region: AdjustmentRegion) -> np.ndarray:
    """
    Create boolean mask for points within rectangular region.

    Uses vectorized NumPy operations for efficiency with large point clouds.
    """
    return (
        (points[:, 0] >= region.xmin) & (points[:, 0] <= region.xmax) &
        (points[:, 1] >= region.ymin) & (points[:, 1] <= region.ymax)
    )


def adjust_heights(points: np.ndarray,
                   regions: List[AdjustmentRegion],
                   verbose: bool = True) -> np.ndarray:
    """
    Apply z-offset adjustments to points within specified regions.

    Args:
        points: Nx3 array of point coordinates
        regions: List of AdjustmentRegion objects
        verbose: Print progress information

    Returns:
        Adjusted points array (copy of original with modifications)
    """
    adjusted = points.copy()
    total_adjusted = 0

    for region in regions:
        mask = get_region_mask(points, region)
        count = np.sum(mask)
        adjusted[mask, 2] += region.z_offset
        total_adjusted += count

        if verbose:
            print(f"  Region '{region.name}': {count:,} points, "
                  f"offset {region.z_offset:+.3f}m")

    if verbose:
        print(f"Total points adjusted: {total_adjusted:,}")

    return adjusted


# ============================================================================
# VISUALIZATION
# ============================================================================

def compute_height_colors(points: np.ndarray,
                          reference_height: float,
                          color_range: Tuple[float, float],
                          colormap_name: str) -> np.ndarray:
    """
    Compute colors based on height difference from reference plane.

    Args:
        points: Nx3 array of point coordinates
        reference_height: Z value of reference plane
        color_range: (min, max) height difference for color mapping
        colormap_name: Name of matplotlib colormap

    Returns:
        Nx3 array of RGB colors (0-1 range)
    """
    z_values = points[:, 2]
    height_diff = z_values - reference_height

    vmin, vmax = color_range
    normalized = (height_diff - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    cmap = cm.get_cmap(colormap_name)
    colors = cmap(normalized)[:, :3]

    return colors


def load_pgm_as_pointcloud(pgm_config: PgmConfig,
                           reference_height: float,
                           ply_bounds: Optional[Tuple[float, float, float, float]] = None
                           ) -> Optional[o3d.geometry.PointCloud]:
    """
    Load PGM image and create a point cloud projected at reference height.

    Args:
        pgm_config: PGM configuration with path, resolution, and origin
        reference_height: Z coordinate to project the image
        ply_bounds: Optional PLY bounding box (xmin, xmax, ymin, ymax) for auto-alignment

    Returns:
        PointCloud with grayscale colors, or None if loading fails
    """
    if not pgm_config.enabled or not pgm_config.path:
        return None

    try:
        from PIL import Image
        print(f"Loading PGM: {pgm_config.path}")

        # Load PGM image
        img = Image.open(pgm_config.path)
        img_array = np.array(img)

        # Get image dimensions
        height, width = img_array.shape[:2]
        print(f"  Image size: {width} x {height} pixels")
        print(f"  Resolution: {pgm_config.resolution} m/pixel")

        # Compute image extent in world coordinates
        img_width_m = width * pgm_config.resolution
        img_height_m = height * pgm_config.resolution

        print(f"  World extent: {img_width_m:.1f} x {img_height_m:.1f} meters")
        print(f"  Origin: ({pgm_config.origin_x:.2f}, {pgm_config.origin_y:.2f})")

        # Compute PGM bounds
        pgm_bounds = compute_pgm_bounds(pgm_config, width, height)

        # Find occupied cells center (more accurate than image bounds)
        occupied_mask = img_array < 100  # Occupied cells (dark pixels)
        occupied_rows, occupied_cols = np.where(occupied_mask)

        if len(occupied_rows) > 0:
            # Convert occupied pixels to world coordinates
            occ_x = pgm_config.origin_x + occupied_cols * pgm_config.resolution
            occ_y = pgm_config.origin_y + (height - 1 - occupied_rows) * pgm_config.resolution
            pgm_center = ((occ_x.min() + occ_x.max()) / 2, (occ_y.min() + occ_y.max()) / 2)
            print(f"  Occupied cells: {len(occupied_rows):,}")
            print(f"  Occupied center: ({pgm_center[0]:.2f}, {pgm_center[1]:.2f})")
        else:
            # Fallback to image bounds center
            pgm_center = ((pgm_bounds[0] + pgm_bounds[1]) / 2, (pgm_bounds[2] + pgm_bounds[3]) / 2)
            print(f"  No occupied cells found, using bounds center")

        # Auto-alignment if enabled and PLY bounds provided
        rotate_180 = pgm_config.rotate_180
        offset_x = pgm_config.offset_x
        offset_y = pgm_config.offset_y

        if pgm_config.auto_align and ply_bounds is not None:
            print(f"  Auto-alignment enabled")
            print(f"    Using rotate_180: {rotate_180} (from config)")

            # Compute offset to align centers using occupied cell center
            offset_x, offset_y = compute_auto_alignment(ply_bounds, pgm_center, rotate_180)
            print(f"    Computed offset: ({offset_x:.2f}, {offset_y:.2f})")
        else:
            print(f"  Manual alignment: offset=({offset_x:.2f}, {offset_y:.2f}), rotate_180={rotate_180}")

        # Create points for each pixel (vectorized for speed)
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        rows = rows.flatten()
        cols = cols.flatten()

        if rotate_180:
            # 180 degree rotation: negate coordinates
            x = -(pgm_config.origin_x + cols * pgm_config.resolution)
            y = -(pgm_config.origin_y + (height - 1 - rows) * pgm_config.resolution)
        else:
            # Standard ROS map format: origin at bottom-left
            x = pgm_config.origin_x + cols * pgm_config.resolution
            y = pgm_config.origin_y + (height - 1 - rows) * pgm_config.resolution

        # Apply offset
        x = x + offset_x
        y = y + offset_y

        z = np.full_like(x, reference_height)

        points = np.stack([x, y, z], axis=1)

        # Get grayscale values
        if len(img_array.shape) == 2:
            gray = img_array[rows, cols] / 255.0
        else:
            gray = img_array[rows, cols, 0] / 255.0

        # Filter out white/free-space pixels (keep only occupied cells)
        # PGM: 0=occupied(black), 205=unknown(gray), 254=free(white)
        # Keep pixels darker than threshold (occupied and unknown)
        occupied_threshold = 250 / 255.0  # Exclude pure white (free space)
        keep_mask = gray < occupied_threshold

        points = points[keep_mask]
        gray = gray[keep_mask]

        # Apply alpha (blend toward white: lower alpha = more transparent/lighter)
        alpha = pgm_config.alpha
        gray_blended = gray * alpha + (1.0 - alpha)  # Blend toward white (1.0)

        colors = np.stack([gray_blended, gray_blended, gray_blended], axis=1)

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        print(f"  Created {len(points):,} points for PGM projection (filtered from {len(rows):,} pixels)")
        return pcd

    except ImportError:
        print("Warning: PIL not installed. Cannot load PGM. Install with: pip install pillow")
        return None
    except Exception as e:
        print(f"Warning: Failed to load PGM: {e}")
        return None


def create_reference_plane(bounds: Tuple[float, float, float, float],
                           reference_height: float,
                           grid_spacing: float = 10.0) -> o3d.geometry.LineSet:
    """
    Create reference plane as grid lines at specified height.

    Args:
        bounds: (xmin, xmax, ymin, ymax) from point cloud
        reference_height: Z coordinate of the plane
        grid_spacing: Spacing between grid lines

    Returns:
        LineSet geometry for the reference grid
    """
    xmin, xmax, ymin, ymax = bounds

    # Add margin
    margin = 5.0
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin

    # Create grid lines for reference plane
    grid_points = []
    grid_lines = []
    point_idx = 0

    # Vertical lines (along Y)
    x = xmin
    while x <= xmax:
        grid_points.append([x, ymin, reference_height])
        grid_points.append([x, ymax, reference_height])
        grid_lines.append([point_idx, point_idx + 1])
        point_idx += 2
        x += grid_spacing

    # Horizontal lines (along X)
    y = ymin
    while y <= ymax:
        grid_points.append([xmin, y, reference_height])
        grid_points.append([xmax, y, reference_height])
        grid_lines.append([point_idx, point_idx + 1])
        point_idx += 2
        y += grid_spacing

    grid_line_set = o3d.geometry.LineSet()
    grid_line_set.points = o3d.utility.Vector3dVector(np.array(grid_points))
    grid_line_set.lines = o3d.utility.Vector2iVector(np.array(grid_lines))
    grid_line_set.paint_uniform_color([0.5, 0.5, 0.5])  # Gray

    return grid_line_set


def create_region_wireframe(region: AdjustmentRegion,
                            zmin: float, zmax: float) -> o3d.geometry.LineSet:
    """
    Create wireframe box for region visualization.

    Args:
        region: AdjustmentRegion object
        zmin, zmax: Z coordinate range for the box

    Returns:
        Open3D LineSet geometry
    """
    # 8 corners of the box
    corners = np.array([
        [region.xmin, region.ymin, zmin],
        [region.xmax, region.ymin, zmin],
        [region.xmax, region.ymax, zmin],
        [region.xmin, region.ymax, zmin],
        [region.xmin, region.ymin, zmax],
        [region.xmax, region.ymin, zmax],
        [region.xmax, region.ymax, zmax],
        [region.xmin, region.ymax, zmax],
    ])

    # 12 edges of the box
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # Red

    return line_set


def visualize(pcd: o3d.geometry.PointCloud,
              config: Config) -> None:
    """
    Interactive 3D visualization with height-based coloring and reference plane.

    Args:
        pcd: Point cloud to visualize
        config: Configuration object with visualization settings
    """
    vis_config = config.visualization

    # Downsample for visualization if configured
    display_pcd = pcd
    if vis_config.downsample_voxel_size:
        print(f"Downsampling for visualization (voxel size: "
              f"{vis_config.downsample_voxel_size}m)...")
        display_pcd = pcd.voxel_down_sample(vis_config.downsample_voxel_size)
        print(f"Display points: {len(display_pcd.points):,}")

    # Compute height-based colors
    points = np.asarray(display_pcd.points)
    colors = compute_height_colors(
        points,
        config.reference_height,
        vis_config.color_range,
        vis_config.colormap
    )
    display_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Build visualization geometry list
    geometries = [display_pcd]

    # Compute PLY bounds for auto-alignment
    ply_bounds = (points[:, 0].min(), points[:, 0].max(),
                  points[:, 1].min(), points[:, 1].max())

    # Add PGM projection if configured
    pgm_pcd = load_pgm_as_pointcloud(vis_config.pgm, config.reference_height, ply_bounds)
    if pgm_pcd is not None:
        geometries.append(pgm_pcd)
    else:
        # Add reference plane (grid lines) only if no PGM
        ref_grid = create_reference_plane(ply_bounds, config.reference_height)
        geometries.append(ref_grid)

    # Add region wireframes
    if vis_config.show_region_bounds and config.regions:
        zmin, zmax = points[:, 2].min(), points[:, 2].max()
        for region in config.regions:
            wireframe = create_region_wireframe(region, zmin, zmax)
            geometries.append(wireframe)

    # Print color legend
    vmin, vmax = vis_config.color_range
    print(f"\nColor Legend (reference height: {config.reference_height}m):")
    print(f"  Blue  = Below reference (down to {vmin:+.1f}m)")
    print(f"  White = At reference height")
    print(f"  Red   = Above reference (up to {vmax:+.1f}m)")
    print("\nControls:")
    print("  Left-click drag: Rotate (CAD-style)")
    print("  Scroll wheel: Zoom")
    print("  Middle-click drag / Ctrl+Left drag: Pan")
    print("  Press 'Q' or close window to exit")

    # Use standard draw_geometries for CAD-style rotation
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Height Adjustment Visualization (ref={config.reference_height}m)",
        width=1280,
        height=720,
        left=50,
        top=50
    )


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adjust height of points in PLY file within rectangular regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python adjust_height.py --config config.yaml

  # Visualize only (no save)
  python adjust_height.py --config config.yaml --no-save

  # Process and save only (no visualization)
  python adjust_height.py --config config.yaml --no-vis
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Skip saving the adjusted point cloud"
    )
    parser.add_argument(
        '--no-vis',
        action='store_true',
        help="Skip visualization"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    print("=" * 60)
    print("PLY Height Adjustment Tool")
    print("=" * 60)

    config = load_config(args.config)
    print(f"Input:  {config.input_ply}")
    print(f"Output: {config.output_ply}")
    print(f"Reference height: {config.reference_height}m")
    print(f"Regions to adjust: {len(config.regions)}")

    # Load point cloud
    print("\n" + "-" * 40)
    start_time = time.time()
    pcd = load_point_cloud(config.input_ply)
    load_time = time.time() - start_time
    print(f"Loaded {len(pcd.points):,} points in {load_time:.2f}s")

    # Show point cloud bounds
    points = np.asarray(pcd.points)
    print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # Apply adjustments
    if config.regions:
        print("\n" + "-" * 40)
        print(f"Applying {len(config.regions)} region adjustments:")
        adjusted_points = adjust_heights(points, config.regions)
        pcd.points = o3d.utility.Vector3dVector(adjusted_points)
    else:
        print("\nNo regions defined, skipping adjustment")

    # Save adjusted point cloud
    if not args.no_save:
        print("\n" + "-" * 40)
        print("Saving adjusted point cloud...")
        save_point_cloud(pcd, config.output_ply)

    # Visualization
    if not args.no_vis and config.visualization.enabled:
        print("\n" + "-" * 40)
        print("Preparing visualization...")
        visualize(pcd, config)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
