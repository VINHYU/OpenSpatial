"""ScanNet++ dataset preprocessing pipeline.

Processes ScanNet++ scenes:
  - Reads RGB, depth, aligned poses, and intrinsics per frame
  - Extracts object annotations from mesh segmentation + OBB data
  - Outputs Parquet files for downstream consumption
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import trimesh
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Mesh & OBB utilities
# ---------------------------------------------------------------------------

def extract_sub_mesh(mesh: o3d.geometry.TriangleMesh, vertex_indices: list) -> o3d.geometry.TriangleMesh:
    """Extract a sub-mesh from an Open3D TriangleMesh by vertex indices."""
    sub_mesh = mesh.select_by_index(vertex_indices)
    if not sub_mesh.has_vertices():
        print("Warning: extracted sub-mesh has no vertices.")
    return sub_mesh


def open3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Convert an Open3D TriangleMesh to a Trimesh object."""
    v = np.asanyarray(o3d_mesh.vertices)
    f = np.asanyarray(o3d_mesh.triangles)
    t_mesh = trimesh.Trimesh(vertices=v, faces=f)

    if o3d_mesh.has_vertex_colors():
        colors = (np.asanyarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8)
        t_mesh.visual.vertex_colors = colors

    return t_mesh


def get_mesh_obb_params(mesh: trimesh.Trimesh) -> tuple | None:
    """Compute OBB parameters (x, y, z, xl, yl, zl, roll, pitch, yaw) from a Trimesh."""
    try:
        obb = mesh.bounding_box_oriented
    except Exception as e:
        print(f"Error computing OBB: {e}")
        return None

    transform_matrix = obb.primitive.transform
    extents = obb.primitive.extents

    center = transform_matrix[:3, 3]
    x, y, z = center
    xl, yl, zl = extents

    rotation_matrix = transform_matrix[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler("zxy", degrees=False)
    roll, pitch, yaw = euler_angles

    return (x, y, z, xl, yl, zl, roll, pitch, yaw)


def get_obb_from_annotation(obb_box: dict) -> list:
    """Extract OBB parameters from ScanNet++ annotation format."""
    obj_location_3d = obb_box["centroid"]
    bbox_rot = np.array(obb_box["normalizedAxes"]).reshape(3, 3).T
    obj_dims_3d = np.array(obb_box["axesLengths"])
    euler_angles = Rotation.from_matrix(bbox_rot).as_euler("zxy", degrees=False)
    return obj_location_3d + obj_dims_3d.tolist() + euler_angles.tolist()


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def json_to_4x4_txt(json_path: str) -> str:
    """Read a matrix from JSON and save as 4x4 txt in the same directory.

    If the JSON contains a 3x3 matrix, it is expanded to 4x4 (bottom-right = 1).
    Returns the absolute path to the saved txt file.
    """
    txt_path = json_path.replace(".json", ".txt")
    if os.path.exists(txt_path):
        return os.path.abspath(txt_path)

    with open(json_path, "r", encoding="utf-8") as f:
        mat = np.array(json.load(f))

    if mat.shape == (3, 3):
        mat_4x4 = np.eye(4)
        mat_4x4[:3, :3] = mat
    else:
        mat_4x4 = mat

    np.savetxt(txt_path, mat_4x4)
    return os.path.abspath(txt_path)

def process_single_scene(
    scene_id: str,
    input_root_dir: str,
    selected_obj_tags: list | None,
) -> dict | None:
    """Process a single ScanNet++ scene and return a record dict."""
    try:
        scene_path = os.path.join(input_root_dir, scene_id)
        base_sensor_path = os.path.join(scene_path, "iphone")
        rgb_dir = os.path.join(base_sensor_path, "rgb")
        depth_dir = os.path.join(base_sensor_path, "depth")
        pose_dir = os.path.join(base_sensor_path, "aligned_pose")
        intrinsic_dir = os.path.join(base_sensor_path, "intrinsic")
        mesh_path = os.path.join(scene_path, "scans", "mesh_aligned_0.05.ply")
        anno_path = os.path.join(scene_path, "scans", "segments_anno.json")

        if not os.path.exists(rgb_dir) or not os.path.exists(mesh_path) or not os.path.exists(anno_path):
            print(f"[scannetpp] Missing data for scene {scene_id}, skipping.")
            return None

        # Mesh & OBB processing
        input_ply = o3d.io.read_triangle_mesh(mesh_path)
        with open(anno_path, "r", encoding="utf-8") as f:
            objs = json.load(f)["segGroups"]

        obj_tags, obj_obbs = [], []
        for obj in objs:
            if selected_obj_tags is None or obj["label"] in selected_obj_tags:
                obj_tags.append(obj["label"])
                sub_mesh = extract_sub_mesh(input_ply, obj["segments"])
                sub_mesh_trimesh = open3d_to_trimesh(sub_mesh)
                params = get_obb_from_annotation(obj["obb"])
                obj_obbs.append(params)

        # Frame processing
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*0.jpg")))
        if not rgb_files:
            return None

        # Get RGB resolution from the first frame
        rgb_sample = cv2.imread(rgb_files[0])
        rgb_h, rgb_w = rgb_sample.shape[:2]

        # Directory for resized depth maps
        depth_resized_dir = os.path.join(base_sensor_path, "depth_resized")
        os.makedirs(depth_resized_dir, exist_ok=True)

        ids, images, poses, intrinsics, depth_maps = [], [], [], [], []
        for rgb_file in rgb_files:
            abs_rgb_path = os.path.abspath(rgb_file)
            frame_name = os.path.splitext(os.path.basename(rgb_file))[0]
            frame_idx_str = frame_name.split("_")[-1]

            ids.append(frame_idx_str)
            images.append(abs_rgb_path)

            expected_depth = os.path.join(depth_dir, f"{frame_name}.png")
            if os.path.exists(expected_depth):
                resized_depth_path = os.path.join(depth_resized_dir, f"{frame_name}.png")
                if not os.path.exists(resized_depth_path):
                    depth_raw = cv2.imread(expected_depth, cv2.IMREAD_UNCHANGED)
                    if depth_raw.shape[:2] != (rgb_h, rgb_w):
                        depth_resized = cv2.resize(depth_raw, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        depth_resized = depth_raw
                    cv2.imwrite(resized_depth_path, depth_resized)
                depth_maps.append(os.path.abspath(resized_depth_path))
            else:
                print(f"[scannetpp] Depth not found for {frame_name} in {scene_id}")
                continue

            expected_pose = os.path.join(pose_dir, f"{frame_name}.json")
            if os.path.exists(expected_pose):
                poses.append(json_to_4x4_txt(expected_pose))
            else:
                print(f"[scannetpp] Pose not found for {frame_name} in {scene_id}")
                continue

            expected_intr = os.path.join(intrinsic_dir, f"{frame_name}.json")
            if os.path.exists(expected_intr):
                intrinsics.append(json_to_4x4_txt(expected_intr))
            else:
                print(f"[scannetpp] Intrinsic not found for {frame_name} in {scene_id}")
                continue

        if (
            len(images) == 0
            or len(poses) != len(images)
            or len(intrinsics) != len(images)
            or len(depth_maps) != len(images)
        ):
            print(f"[scannetpp] Incomplete data for scene {scene_id}, skipping.")
            return None

        return {
            "scene_id": scene_id,
            "id": ids,
            "image": images,
            "pose": poses,
            "intrinsic": intrinsics,
            "obj_tags": obj_tags,
            "depth_map": depth_maps,
            "bboxes_3d_world_coords": obj_obbs,
            "axis_align_matrix": None,
            "depth_scale": 1000,
            "is_metric_depth": True,
        }
    except Exception as e:
        print(f"[scannetpp] Error processing {scene_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Parquet generation
# ---------------------------------------------------------------------------

def generate_parquet(
    input_root_dir: str,
    output_dir: str,
    selected_tags_file: str | None,
    chunk_size: int = 100,
    max_workers: int = 32,
) -> None:
    """Generate Parquet files from all ScanNet++ scenes."""
    os.makedirs(output_dir, exist_ok=True)

    scene_folders = sorted([
        f for f in os.listdir(input_root_dir)
        if os.path.isdir(os.path.join(input_root_dir, f))
    ])
    print(f"Found {len(scene_folders)} scenes, processing with {max_workers} workers ...")

    # Load selected object tags
    selected_obj_tags = None
    if selected_tags_file is not None:
        selected_obj_tags = []
        with open(selected_tags_file, "r") as f:
            for line in f:
                tag = line.strip()
                if tag:
                    selected_obj_tags.append(tag)

    all_results = []
    chunk_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_scene, sid, input_root_dir, selected_obj_tags): sid
            for sid in scene_folders
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="parquet"):
            result = future.result()
            if result:
                all_results.append(result)

            if len(all_results) >= chunk_size:
                save_path = os.path.join(output_dir, f"batch_{chunk_count}.parquet")
                pd.DataFrame(all_results).to_parquet(save_path, engine="pyarrow")
                all_results = []
                chunk_count += 1

    if all_results:
        save_path = os.path.join(output_dir, f"batch_{chunk_count}.parquet")
        pd.DataFrame(all_results).to_parquet(save_path, engine="pyarrow")

    

    print(f"Done. Generated {chunk_count + 1} Parquet file(s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ScanNet++ dataset preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_root", type=str, required=True,
        help="Root directory of ScanNet++ data (e.g. /data/scannetpp/data).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for Parquet files.",
    )
    parser.add_argument(
        "--selected_tags_file", type=str, default=None,
        help="Path to semantic_classes_selected_2.txt. If not provided, all object tags are kept.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100,
        help="Number of records per Parquet file (default: 100).",
    )
    parser.add_argument(
        "--max_workers", type=int, default=32,
        help="Max parallel workers for processing (default: 32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generate_parquet(
        args.input_root,
        args.output_dir,
        args.selected_tags_file,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
    )

    print("All steps completed successfully.")


if __name__ == "__main__":
    main()
