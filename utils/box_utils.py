"""3D bounding box geometry utilities."""

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation


def compute_box_3d_points(size):
    """Compute 8 corner points of an axis-aligned box centered at the origin.

    Args:
        size: (3,) array — [xl, yl, zl].

    Returns:
        np.ndarray of shape (8, 3).
    """
    hs = np.asarray(size) / 2
    return np.array([
        [-hs[0], -hs[1], -hs[2]],
        [-hs[0], -hs[1],  hs[2]],
        [-hs[0],  hs[1],  hs[2]],
        [-hs[0],  hs[1], -hs[2]],
        [ hs[0], -hs[1], -hs[2]],
        [ hs[0], -hs[1],  hs[2]],
        [ hs[0],  hs[1],  hs[2]],
        [ hs[0],  hs[1], -hs[2]],
    ])


def compute_box_3d_corners(center, size, rotation, euler_order='zxy'):
    """Compute 8 world-space corners of an oriented 3D bounding box.

    Args:
        center: (3,) — [x, y, z].
        size: (3,) — [xl, yl, zl].
        rotation: (3,) — euler angles in radians.
        euler_order: rotation convention (default 'zxy').

    Returns:
        np.ndarray of shape (8, 3).
    """
    center = np.asarray(center)
    rot_mat = SciRotation.from_euler(euler_order, list(rotation), degrees=False).as_matrix()
    corners = compute_box_3d_points(size) @ rot_mat.T + center
    return corners


def compute_box_3d_corners_from_params(box_params, euler_order='zxy'):
    """Compute 8 corners from a 9-value box parameter list.

    Args:
        box_params: [cx, cy, cz, xl, yl, zl, roll, pitch, yaw].

    Returns:
        np.ndarray of shape (8, 3).
    """
    return compute_box_3d_corners(
        box_params[:3], box_params[3:6], box_params[6:9], euler_order)


def convert_box_3d_world_to_camera(box_params, pose, euler_order='zxy'):
    """Convert a 9-param world-frame 3D box to camera frame.

    Args:
        box_params: [x, y, z, xl, yl, zl, roll, pitch, yaw].
        pose: 4x4 camera-to-world matrix.
        euler_order: Euler convention (default 'zxy').

    Returns:
        9-element list in camera frame, or None if input is invalid.
    """
    if box_params is None or len(box_params) < 9:
        return None
    center = np.array(box_params[:3])
    size = np.array(box_params[3:6])
    rotation = box_params[6:9]

    rot_mat = SciRotation.from_euler(euler_order, list(rotation), degrees=False).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = center

    cam_transform = np.linalg.inv(pose) @ transform
    cam_center = cam_transform[:3, 3]
    cam_euler = SciRotation.from_matrix(cam_transform[:3, :3]).as_euler(euler_order, degrees=False)
    return list(cam_center) + list(size) + list(cam_euler)


def check_box_2d_overlap(box1_xy, box2_xy):
    """Check if two 2D polygon projections overlap or are close.

    Args:
        box1_xy: Nx2 array of 2D polygon vertices.
        box2_xy: Mx2 array of 2D polygon vertices.

    Returns:
        True if polygons intersect or are within 50% of the larger box's longest edge.
    """
    from shapely.geometry import Polygon

    poly1 = Polygon(box1_xy)
    poly2 = Polygon(box2_xy)

    intersects_flag = poly1.intersects(poly2)

    distance = poly1.distance(poly2)
    max_size1 = np.max(np.linalg.norm(
        box1_xy - np.roll(box1_xy, shift=-1, axis=0), axis=1))
    max_size2 = np.max(np.linalg.norm(
        box2_xy - np.roll(box2_xy, shift=-1, axis=0), axis=1))
    max_size = max(max_size1, max_size2)
    distance_flag = distance < max_size * 0.5

    return intersects_flag or distance_flag


def check_box_3d_vertical_overlap(box_3d_world_list):
    """Check if any pair of 3D boxes has overlapping XY projections.

    Args:
        box_3d_world_list: list of 9-param boxes [cx,cy,cz,xl,yl,zl,r,p,y].

    Returns:
        True if ANY pair overlaps (has vertical overlap).
    """
    for i, box1 in enumerate(box_3d_world_list):
        for j in range(i + 1, len(box_3d_world_list)):
            corners1_xy = compute_box_3d_corners_from_params(box1)[:, :2]
            corners2_xy = compute_box_3d_corners_from_params(box_3d_world_list[j])[:, :2]
            if check_box_2d_overlap(corners1_xy, corners2_xy):
                return True
    return False
