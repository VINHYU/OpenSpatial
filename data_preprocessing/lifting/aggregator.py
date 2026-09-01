import os
import json
import numpy as np
import open3d as o3d
from PIL import Image
from typing import List, Dict, Optional, Tuple

class SemanticMatcher:
    """Uses sentence embeddings to determine if two category names are semantically similar."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, similarity_threshold: float = 0.4,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self._embedding_cache = {}
        self.similarity_threshold = similarity_threshold  # Threshold for considering two labels as same category
        self.model_name = model_name

    def set_threshold(self, threshold: float):
        """Update the similarity threshold."""
        self.similarity_threshold = threshold

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a lightweight model for efficiency
                self.model = SentenceTransformer(self.model_name)
                print("    [SemanticMatcher] Loaded sentence transformer model")
            except ImportError:
                print("    [SemanticMatcher] sentence-transformers not installed, falling back to string matching")
                self.model = "fallback"

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text, with caching."""
        if text not in self._embedding_cache:
            self._load_model()
            if self.model == "fallback":
                return None
            # Expand underscores to spaces for better semantic matching
            clean_text = text.replace("_", " ").replace("-", " ")
            self._embedding_cache[text] = self.model.encode(clean_text, convert_to_numpy=True)
        return self._embedding_cache[text]

    def are_same_category(self, label_a: str, label_b: str) -> bool:
        """
        Determine if two labels refer to the same category using semantic similarity.

        Returns True if:
        1. Labels are exactly the same (string match)
        2. Semantic similarity is above threshold
        """
        # Exact match
        if label_a == label_b:
            return True

        # Normalize labels
        label_a_norm = label_a.lower().strip()
        label_b_norm = label_b.lower().strip()

        if label_a_norm == label_b_norm:
            return True

        # Check for substring containment (e.g., "chair" and "office_chair")
        if label_a_norm in label_b_norm or label_b_norm in label_a_norm:
            return True

        # Semantic similarity check
        emb_a = self._get_embedding(label_a)
        emb_b = self._get_embedding(label_b)

        if emb_a is None or emb_b is None:
            return False

        # Cosine similarity
        similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))

        return similarity >= self.similarity_threshold

# Global semantic matcher instance
_semantic_matcher = None

def get_semantic_matcher(similarity_threshold: float = 0.4,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SemanticMatcher:
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = SemanticMatcher(similarity_threshold, model_name)
    else:
        # Update threshold if already initialized
        _semantic_matcher.set_threshold(similarity_threshold)
    return _semantic_matcher

class TrackedInstance:
    """Represents a unique object instance in the 3D scene."""
    def __init__(self, instance_id: int, label: str, initial_points: np.ndarray, initial_colors: np.ndarray, description: str = ""):
        self.id = instance_id
        self.label = label
        self.description = description
        self.points = initial_points
        self.colors = initial_colors
        self.aabb = self._compute_aabb(initial_points)
        self.merged_into = None

    def merge(self, new_points: np.ndarray, new_colors: np.ndarray, new_label: str = None, new_description: str = None):
        """
        Adds new points to this instance and updates the bounding box.
        If new_label/new_description are provided and new cloud is larger, updates label/description.
        """
        # Determine if we should update label/description (use the one with more points)
        if new_label is not None and len(new_points) > len(self.points):
            self.label = new_label
            if new_description is not None:
                self.description = new_description

        self.points = np.vstack((self.points, new_points))
        self.colors = np.vstack((self.colors, new_colors))

        # Recompute AABB with the combined cloud
        # Optional: Downsample here if clouds get too massive for performance
        self.aabb = self._compute_aabb(self.points)

    def _compute_aabb(self, points: np.ndarray) -> o3d.geometry.AxisAlignedBoundingBox:
        """Computes AABB. Uses outlier removal first to ensure tight boxes."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Remove statistical outliers to prevent noise from expanding the box too much
        # nb_neighbors=20, std_ratio=2.0 are standard defaults
        clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(clean_pcd.points) == 0:
            # Fallback to raw points if outlier removal kills everything
            return pcd.get_axis_aligned_bounding_box()

        return clean_pcd.get_axis_aligned_bounding_box()

class PointCloudAggregator:
    """Handles the extraction, matching, and merging of partial point clouds."""

    def __init__(self, config):
        self.config = config
        self.instances: List[TrackedInstance] = []
        self.next_id = 0
        self.bounding_boxes_by_id = {}

    @staticmethod
    def resolve_instance(instance):
        while instance is not None and instance.merged_into is not None:
            instance = instance.merged_into
        return instance

    def ingest_frame(self,
                     points_frame: np.ndarray,
                     colors_frame: np.ndarray,
                     detections: List[Dict],
                     masks: List[np.ndarray]):
        """
        Extracts objects from a single frame and merges them into the scene.

        Args:
            points_frame: (H, W, 3) array of 3D coordinates.
            colors_frame: (H, W, 3) array of RGB colors.
            detections: List of detection dicts.
            masks: List of 2D binary masks matching detections.
        """
        pc_h, pc_w = points_frame.shape[:2]
        instance_refs = []

        for det, mask in zip(detections, masks):
            label_raw = det["label"]
            label_key = label_raw.lower().strip().replace(" ", "_").replace("/", "-")
            description = det.get("description", "")

            # 1. Resize mask to match point cloud resolution
            mask_pil = Image.fromarray(mask)
            mask_resized = mask_pil.resize((pc_w, pc_h), resample=Image.NEAREST)
            mask_bool = np.array(mask_resized).astype(bool)

            # 2. Extract partial cloud
            obj_points = points_frame[mask_bool]
            obj_colors = colors_frame[mask_bool]

            # 3. Filter NaNs and empty clouds
            valid_indices = ~np.isnan(obj_points).any(axis=1)
            obj_points = obj_points[valid_indices]
            obj_colors = obj_colors[valid_indices]

            if len(obj_points) < 50: # Ignore tiny noise clusters
                instance_refs.append(None)
                continue

            # 4. Try to match with existing instances (using OBB-based IoU)
            matched_instance, cross_category_merge = self._find_best_match(label_key, obj_points, description)

            if matched_instance:
                # Merge into existing instance
                # For cross-category merges, pass label/description to potentially update
                if cross_category_merge:
                    matched_instance.merge(obj_points, obj_colors, label_key, description)
                else:
                    matched_instance.merge(obj_points, obj_colors)
                instance_refs.append(matched_instance)
            else:
                # Create new instance
                new_inst = TrackedInstance(self.next_id, label_key, obj_points, obj_colors, description)
                self.instances.append(new_inst)
                self.next_id += 1
                instance_refs.append(new_inst)

        return instance_refs

    def consolidate_instances(self):
        """
        Recursively merges instances that have high IoU until no more merges are possible.
        For same category (determined by semantic similarity): uses max bidirectional IoU
        For cross category: uses standard IoU
        Uses the category and description from the instance with more points.
        Skips merging for floor/ground objects.
        """
        semantic_matcher = get_semantic_matcher(
            self.config.semantic_similarity_threshold, self.config.semantic_model
        )
        iteration = 0

        while True:
            iteration += 1
            merged_any = False

            # Find pairs of instances to merge
            i = 0
            while i < len(self.instances):
                j = i + 1
                while j < len(self.instances):
                    inst_a = self.instances[i]
                    inst_b = self.instances[j]

                    # Skip merging if either instance is floor/ground
                    if any(skip in inst_a.label.lower() for skip in self.config.skip_merge_labels) or \
                       any(skip in inst_b.label.lower() for skip in self.config.skip_merge_labels):
                        j += 1
                        continue

                    # Fast AABB overlap check - skip voxel IoU if bounding boxes don't overlap
                    if not self._aabb_overlap(inst_a.points, inst_b.points):
                        j += 1
                        continue

                    # Calculate voxel-based IoU (returns both standard and max bidirectional)
                    iou, max_bidir_iou = self._calculate_3d_iou_voxel(inst_a.points, inst_b.points)

                    # Use semantic matching to determine if labels refer to same category
                    same_category = semantic_matcher.are_same_category(inst_a.label, inst_b.label)
                    effective_iou = max_bidir_iou if same_category else iou

                    if effective_iou > self.config.cross_category_iou_threshold:
                        # Merge inst_b into inst_a (or vice versa based on point count)
                        if len(inst_a.points) >= len(inst_b.points):
                            # Keep inst_a's label/description, merge inst_b into inst_a
                            inst_a.points = np.vstack((inst_a.points, inst_b.points))
                            inst_a.colors = np.vstack((inst_a.colors, inst_b.colors))
                            inst_a.aabb = inst_a._compute_aabb(inst_a.points)
                            inst_b.merged_into = inst_a
                            print(f"    Consolidation iter {iteration}: Merged '{inst_b.label}_{inst_b.id}' into '{inst_a.label}_{inst_a.id}' (IoU={effective_iou:.3f}, same_cat={same_category})")
                        else:
                            # Keep inst_b's label/description, merge inst_a into inst_b
                            inst_b.points = np.vstack((inst_b.points, inst_a.points))
                            inst_b.colors = np.vstack((inst_b.colors, inst_a.colors))
                            inst_b.aabb = inst_b._compute_aabb(inst_b.points)
                            # Swap so we remove the correct one
                            self.instances[i] = inst_b
                            inst_a.merged_into = inst_b
                            print(f"    Consolidation iter {iteration}: Merged '{inst_a.label}_{inst_a.id}' into '{inst_b.label}_{inst_b.id}' (IoU={effective_iou:.3f}, same_cat={same_category})")

                        # Remove inst_b from list
                        self.instances.pop(j)
                        merged_any = True
                        # Don't increment j, check the new instance at position j
                    else:
                        j += 1
                i += 1

            if not merged_any:
                if iteration > 1:
                    print(f"    Consolidation complete after {iteration} iterations. {len(self.instances)} instances remain.")
                break

    def _find_best_match(self, label: str, new_points: np.ndarray, new_description: str = "") -> Tuple[Optional[TrackedInstance], bool]:
        """
        Finds the existing instance to merge with using voxel-based IoU.

        Strategy:
        1. First, look for semantically similar category instances with max bidirectional IoU above threshold
           (this helps merge small objects into large ones of same category)
           Uses semantic embeddings to determine if two labels refer to the same category
        2. If standard IoU > cross_category_iou_threshold with ANY instance, merge regardless of category
           (use category/description from instance with more points)
        Skips merging for floor/ground objects.

        Returns:
            (matched_instance, cross_category_merge):
            - matched_instance: The instance to merge with, or None
            - cross_category_merge: True if merging across different categories
        """
        semantic_matcher = get_semantic_matcher(
            self.config.semantic_similarity_threshold, self.config.semantic_model
        )

        # Skip merging if the new object is floor/ground
        if any(skip in label.lower() for skip in self.config.skip_merge_labels):
            return None, False

        best_same_category_iou = -1.0
        best_same_category_match = None

        best_cross_category_iou = -1.0
        best_cross_category_match = None

        for instance in self.instances:
            # Skip floor/ground instances
            if any(skip in instance.label.lower() for skip in self.config.skip_merge_labels):
                continue

            # Fast AABB overlap check - skip voxel IoU if bounding boxes don't overlap
            if not self._aabb_overlap(instance.points, new_points):
                continue

            # Calculate voxel-based IoU (returns both standard and max bidirectional)
            iou, max_bidir_iou = self._calculate_3d_iou_voxel(instance.points, new_points)

            # Use semantic matching to determine if labels refer to same category
            same_category = semantic_matcher.are_same_category(instance.label, label)

            if same_category:
                # Same category matching - use max bidirectional IoU
                # This allows small objects to merge into larger ones of same category
                if max_bidir_iou > best_same_category_iou:
                    best_same_category_iou = max_bidir_iou
                    best_same_category_match = instance
            else:
                # Cross-category matching - use standard IoU
                if iou > self.config.cross_category_iou_threshold and iou > best_cross_category_iou:
                    best_cross_category_iou = iou
                    best_cross_category_match = instance

        # Priority 1: Same category match above threshold
        if best_same_category_iou >= self.config.spatial_iou_threshold:
            return best_same_category_match, False

        # Priority 2: Cross-category match with high IoU
        if best_cross_category_match is not None:
            return best_cross_category_match, True

        return None, False

    def _aabb_overlap(self, points_a: np.ndarray, points_b: np.ndarray) -> bool:
        """
        Fast check if two point clouds' AABBs overlap.
        Used as a pre-filter before expensive voxel-based IoU calculation.
        """
        min_a = points_a.min(axis=0)
        max_a = points_a.max(axis=0)
        min_b = points_b.min(axis=0)
        max_b = points_b.max(axis=0)

        # Check if AABBs overlap in all 3 dimensions
        return np.all(min_a <= max_b) and np.all(max_a >= min_b)

    def _calculate_3d_iou_voxel(self, points_a: np.ndarray, points_b: np.ndarray) -> tuple:
        """
        Calculates Intersection over Union of two point clouds using spatial hashing
        with voxel size from config.

        Args:
            points_a: First point cloud (N, 3)
            points_b: Second point cloud (M, 3)

        Returns:
            tuple: (standard_iou, max_bidirectional_iou)
        """
        if len(points_a) == 0 or len(points_b) == 0:
            return 0.0, 0.0

        voxel_size = self.config.voxel_size

        # Convert points to voxel indices
        def points_to_voxels(points):
            voxel_indices = np.floor(points / voxel_size).astype(np.int64)
            # Convert to set of tuples for fast set operations
            return set(map(tuple, voxel_indices))

        voxels_a = points_to_voxels(points_a)
        voxels_b = points_to_voxels(points_b)

        # Calculate intersection and union
        intersection = voxels_a & voxels_b
        union = voxels_a | voxels_b

        n_intersection = len(intersection)
        n_union = len(union)
        n_a = len(voxels_a)
        n_b = len(voxels_b)

        if n_union == 0:
            return 0.0, 0.0

        # Standard IoU
        iou = n_intersection / n_union

        # Bidirectional IoU (useful for small objects inside large ones)
        iou_a = n_intersection / n_a if n_a > 0 else 0.0
        iou_b = n_intersection / n_b if n_b > 0 else 0.0
        max_bidirectional_iou = max(iou_a, iou_b)

        return iou, max_bidirectional_iou

    def _fit_plane_svd(self, points: np.ndarray) -> np.ndarray:
        """
        Fits a plane to points using SVD (Principal Component Analysis).
        Returns the normal vector of the plane.

        Args:
            points: (N, 3) array of 3D points

        Returns:
            normal: (3,) unit normal vector of the fitted plane
        """
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        # SVD to find principal components
        # The normal is the eigenvector with smallest eigenvalue
        _, _, Vt = np.linalg.svd(centered_points)

        # The last row of Vt is the normal (smallest singular value direction)
        normal = Vt[-1]

        return normal / np.linalg.norm(normal)

    def _compute_floor_rotation(self) -> Tuple[np.ndarray, bool]:
        """
        Finds the largest floor point cloud and computes a rotation matrix
        to make the floor parallel with the horizontal plane (XY plane, Z-up).

        Returns:
            rotation_matrix: 3x3 rotation matrix
            success: Whether a valid floor was found
        """
        # Find all floor instances
        floor_instances = [inst for inst in self.instances if 'floor' in inst.label.lower()]

        if not floor_instances:
            print("    No floor instances found, skipping floor alignment.")
            return np.eye(3), False

        # Find the largest floor by point count
        largest_floor = max(floor_instances, key=lambda x: len(x.points))
        print(f"    Using floor '{largest_floor.label}_{largest_floor.id}' ({len(largest_floor.points)} points) for alignment.")

        # Get floor points
        floor_points = np.asarray(largest_floor.points, dtype=np.float64)

        # Subsample if too many points
        max_points = 10000
        if len(floor_points) > max_points:
            indices = np.random.choice(len(floor_points), max_points, replace=False)
            floor_points = floor_points[indices]
            print(f"    Subsampled to {len(floor_points)} points for plane fitting.")

        # Check minimum points
        if len(floor_points) < 10:
            print(f"    Too few points ({len(floor_points)}) for plane fitting.")
            return np.eye(3), False

        print(f"    Fitting plane using SVD with {len(floor_points)} points...")

        # Fit plane using SVD (pure numpy, no Open3D)
        try:
            floor_normal = self._fit_plane_svd(floor_points)
        except Exception as e:
            print(f"    Plane fitting failed: {e}")
            return np.eye(3), False

        # Target: Z-axis pointing up (0, 0, 1) for XY plane alignment
        up_vector = np.array([0.0, 0.0, 1.0])

        # If normal points downward (negative Z), flip it
        if floor_normal[2] < 0:
            floor_normal = -floor_normal

        print(f"    Floor normal: [{floor_normal[0]:.3f}, {floor_normal[1]:.3f}, {floor_normal[2]:.3f}]")

        # Compute rotation from floor_normal to up_vector
        rotation_matrix = self._rotation_matrix_from_vectors(floor_normal, up_vector)

        # Calculate angle for logging
        angle_deg = np.degrees(np.arccos(np.clip(np.dot(floor_normal, up_vector), -1.0, 1.0)))
        print(f"    Rotation angle: {angle_deg:.2f} degrees")

        return rotation_matrix, True

    def _rotation_matrix_from_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Computes the rotation matrix that rotates vec1 to vec2.
        Uses Rodrigues' rotation formula.
        """
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)

        # Check if vectors are parallel
        cross = np.cross(a, b)
        dot = np.dot(a, b)

        if np.linalg.norm(cross) < 1e-8:
            # Vectors are parallel
            if dot > 0:
                return np.eye(3)  # Same direction
            else:
                # Opposite direction - rotate 180 degrees around any perpendicular axis
                # Find a perpendicular vector
                if abs(a[0]) < 0.9:
                    perp = np.cross(a, np.array([1, 0, 0]))
                else:
                    perp = np.cross(a, np.array([0, 1, 0]))
                perp = perp / np.linalg.norm(perp)
                # Rotation matrix for 180 degrees around perp
                return 2 * np.outer(perp, perp) - np.eye(3)

        # Rodrigues' formula
        v = cross
        s = np.linalg.norm(v)  # sin(angle)
        c = dot  # cos(angle)

        # Skew-symmetric cross-product matrix
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        return R

    def _keep_largest_component(self, points: np.ndarray, colors: np.ndarray,
                                  cell_size: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses spatial hashing to find connected components and keeps only the largest one.
        Points within the same or adjacent cells are considered connected.

        Args:
            points: (N, 3) array of 3D points
            colors: (N, 3) array of colors
            cell_size: Size of spatial hash cells (default 30cm)

        Returns:
            filtered_points: Points from the largest connected component
            filtered_colors: Corresponding colors
        """
        if len(points) < 10:
            return points, colors

        # Spatial hashing: assign each point to a cell
        cell_indices = np.floor(points / cell_size).astype(np.int32)

        # Build a mapping from cell to point indices
        from collections import defaultdict
        cell_to_points = defaultdict(list)
        for i, cell in enumerate(cell_indices):
            cell_key = tuple(cell)
            cell_to_points[cell_key].append(i)

        # Find connected components using BFS on cells
        # Two cells are connected if they are adjacent (26-connectivity in 3D)
        visited_cells = set()
        components = []  # List of (set of point indices)

        # Generate 26 neighbor offsets
        neighbor_offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx != 0 or dy != 0 or dz != 0:
                        neighbor_offsets.append((dx, dy, dz))

        for start_cell in cell_to_points.keys():
            if start_cell in visited_cells:
                continue

            # BFS from this cell
            component_points = set()
            queue = [start_cell]
            visited_cells.add(start_cell)

            while queue:
                current_cell = queue.pop(0)
                # Add all points in this cell to the component
                component_points.update(cell_to_points[current_cell])

                # Check all 26 neighbors
                for offset in neighbor_offsets:
                    neighbor = (current_cell[0] + offset[0],
                               current_cell[1] + offset[1],
                               current_cell[2] + offset[2])
                    if neighbor in cell_to_points and neighbor not in visited_cells:
                        visited_cells.add(neighbor)
                        queue.append(neighbor)

            components.append(component_points)

        # Find the largest component by volume (bounding box volume)
        if len(components) <= 1:
            return points, colors

        largest_component = None
        largest_volume = -1

        for comp in components:
            comp_indices = list(comp)
            comp_points = points[comp_indices]

            # Calculate bounding box volume
            min_coords = np.min(comp_points, axis=0)
            max_coords = np.max(comp_points, axis=0)
            volume = np.prod(max_coords - min_coords)

            if volume > largest_volume:
                largest_volume = volume
                largest_component = comp_indices

        # Return only the largest component
        return points[largest_component], colors[largest_component]

    def _compute_oriented_bounding_box(self, points: np.ndarray) -> Dict:
        """
        Computes the minimum area oriented bounding box that can rotate around Z axis.
        Uses convex hull and rotating calipers algorithm for optimal fit.

        Args:
            points: (N, 3) array of 3D points (Z-up coordinate system)

        Returns:
            Dictionary with center, size, and rotation (radians around Z axis)
        """
        if len(points) < 3:
            # Fallback to AABB
            min_pt = np.min(points, axis=0)
            max_pt = np.max(points, axis=0)
            return {
                'center': ((min_pt + max_pt) / 2).tolist(),
                'size': (max_pt - min_pt).tolist(),
                'rotation': 0.0
            }

        # Extract XY coordinates (Z is vertical)
        xy_points = points[:, :2]
        z_coords = points[:, 2]
        min_z, max_z = np.min(z_coords), np.max(z_coords)
        size_z = max_z - min_z

        # Compute 2D convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xy_points)
            hull_points = xy_points[hull.vertices]
        except Exception:
            # Fallback if convex hull fails (e.g., collinear points)
            hull_points = xy_points

        # Find minimum area bounding box using rotating calipers
        # Try angles from each edge of the convex hull
        n_hull = len(hull_points)

        best_area = float('inf')
        best_angle = 0.0
        best_size = None
        best_center_xy = None

        for i in range(n_hull):
            # Get edge vector
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % n_hull]
            edge = p2 - p1

            # Compute angle of this edge
            angle = np.arctan2(edge[1], edge[0])

            # Rotate all hull points by -angle to align edge with X axis
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)

            rotated = np.zeros_like(hull_points)
            rotated[:, 0] = hull_points[:, 0] * cos_a - hull_points[:, 1] * sin_a
            rotated[:, 1] = hull_points[:, 0] * sin_a + hull_points[:, 1] * cos_a

            # Find AABB of rotated points
            min_xy = np.min(rotated, axis=0)
            max_xy = np.max(rotated, axis=0)
            size_xy = max_xy - min_xy
            area = size_xy[0] * size_xy[1]

            if area < best_area:
                best_area = area
                best_angle = angle
                best_size = size_xy
                # Compute center in rotated space, then rotate back
                center_rotated = (min_xy + max_xy) / 2
                # Rotate center back to world space
                cos_a_inv = np.cos(angle)
                sin_a_inv = np.sin(angle)
                best_center_xy = np.array([
                    center_rotated[0] * cos_a_inv - center_rotated[1] * sin_a_inv,
                    center_rotated[0] * sin_a_inv + center_rotated[1] * cos_a_inv
                ])

        # Also try PCA angle as a candidate (it's often close to optimal)
        cx, cy = np.mean(xy_points, axis=0)
        dx = xy_points[:, 0] - cx
        dy = xy_points[:, 1] - cy
        cov_xx = np.sum(dx * dx)
        cov_xy = np.sum(dx * dy)
        cov_yy = np.sum(dy * dy)
        pca_angle = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)

        for angle in [pca_angle, pca_angle + np.pi/2]:
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)

            rotated = np.zeros_like(hull_points)
            rotated[:, 0] = hull_points[:, 0] * cos_a - hull_points[:, 1] * sin_a
            rotated[:, 1] = hull_points[:, 0] * sin_a + hull_points[:, 1] * cos_a

            min_xy = np.min(rotated, axis=0)
            max_xy = np.max(rotated, axis=0)
            size_xy = max_xy - min_xy
            area = size_xy[0] * size_xy[1]

            if area < best_area:
                best_area = area
                best_angle = angle
                best_size = size_xy
                center_rotated = (min_xy + max_xy) / 2
                cos_a_inv = np.cos(angle)
                sin_a_inv = np.sin(angle)
                best_center_xy = np.array([
                    center_rotated[0] * cos_a_inv - center_rotated[1] * sin_a_inv,
                    center_rotated[0] * sin_a_inv + center_rotated[1] * cos_a_inv
                ])

        # Normalize angle to [-pi/2, pi/2] range
        while best_angle > np.pi / 2:
            best_angle -= np.pi
        while best_angle < -np.pi / 2:
            best_angle += np.pi

        world_center_z = (min_z + max_z) / 2

        return {
            'center': [float(best_center_xy[0]), float(best_center_xy[1]), float(world_center_z)],
            'size': [float(best_size[0]), float(best_size[1]), float(size_z)],
            'rotation': float(best_angle)
        }

    def save_results(self, save_dir: str, scene_points: np.ndarray = None, scene_colors: np.ndarray = None) -> dict:
        """
        Saves all tracked instances to PLY files.
        Applies floor-based rotation to make the floor horizontal.
        Ensures objects are above the floor (right-side up) and shifts floor to z=0.

        Args:
            save_dir: Directory to save PLY files
            scene_points: (N, 3) array of scene points for stitched point cloud
            scene_colors: (N, 3) array of scene colors

        Returns:
            dict: Scene transformation parameters (rotation_matrix, flip_matrix, z_shift)
        """
        print(f"    Aggregator: Saving {len(self.instances)} unique object instances.")

        # Step 1: Compute floor rotation
        rotation_matrix, has_rotation = self._compute_floor_rotation()

        # Step 2: Apply rotation to all instances first to determine z_shift
        rotated_instances = []
        for inst in self.instances:
            points = inst.points.copy()
            if has_rotation:
                points = (rotation_matrix @ points.T).T
            rotated_instances.append((inst, points))

        # Step 3: Find floor z-level and check if upside down
        # Get floor instances after rotation
        floor_z_values = []
        non_floor_z_values = []
        for inst, points in rotated_instances:
            if 'floor' in inst.label.lower():
                floor_z_values.extend(points[:, 2].tolist())
            else:
                non_floor_z_values.extend(points[:, 2].tolist())

        # Determine if scene is upside down
        flip_z = False
        if floor_z_values and non_floor_z_values:
            floor_z_median = np.median(floor_z_values)
            non_floor_z_median = np.median(non_floor_z_values)

            # If non-floor objects are below floor on average, flip the scene
            if non_floor_z_median < floor_z_median:
                print(f"    Scene appears upside down (floor z={floor_z_median:.3f}, objects z={non_floor_z_median:.3f}). Flipping...")
                flip_z = True

        # Create flip matrix if needed (rotate 180 degrees around X axis)
        if flip_z:
            flip_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ], dtype=np.float64)
            # Apply flip to all rotated points
            for i, (inst, points) in enumerate(rotated_instances):
                rotated_instances[i] = (inst, (flip_matrix @ points.T).T)

        # Recalculate floor z after potential flip
        floor_z_min = float('inf')
        for inst, points in rotated_instances:
            if 'floor' in inst.label.lower():
                floor_z_min = min(floor_z_min, np.min(points[:, 2]))

        # If no floor found, use minimum z of all points
        if floor_z_min == float('inf'):
            for inst, points in rotated_instances:
                floor_z_min = min(floor_z_min, np.min(points[:, 2]))

        z_shift = -floor_z_min if floor_z_min != float('inf') else 0
        print(f"    Shifting z by {z_shift:.3f} to place floor at z=0")

        # Step 4: Save object instances with rotation and shift applied
        # Also compute and collect bounding boxes
        bounding_boxes = {}

        for inst, points in rotated_instances:
            filename = f"{inst.label}_{inst.id}.ply"
            filepath = os.path.join(save_dir, filename)

            # Apply z shift
            points[:, 2] += z_shift

            # Filter to keep only the largest connected component (30cm separation threshold)
            # Skip filtering for floor/wall/ceiling as they may legitimately be disconnected
            colors = inst.colors.copy()
            skip_labels = ['floor', 'wall', 'ceiling', 'room']
            should_filter = not any(skip in inst.label.lower() for skip in skip_labels)

            if should_filter and len(points) > 100:
                original_count = len(points)
                points, colors = self._keep_largest_component(points, colors, cell_size=0.05)
                if len(points) < original_count:
                    print(f"      Filtered {inst.label}_{inst.id}: {original_count} -> {len(points)} points (kept largest component)")

            # Compute oriented bounding box for this object
            obb = self._compute_oriented_bounding_box(points)
            object_key = f"{inst.label}_{inst.id}"
            bounding_boxes[object_key] = {
                'label': inst.label,
                'id': inst.id,
                'description': inst.description,
                'center': obb['center'],
                'size': obb['size'],
                'rotation': obb['rotation'],
                'num_points': len(points)
            }

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Normalize colors
            if colors.max() > 1.0:
                pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            else:
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # Downsample object point cloud
            if self.config.object_downsample_voxel_size > 0:
                original_count = len(pcd.points)
                pcd = pcd.voxel_down_sample(self.config.object_downsample_voxel_size)
                if len(pcd.points) < original_count:
                    print(f"      Downsampled {inst.label}_{inst.id}: {original_count} -> {len(pcd.points)} points")

            o3d.io.write_point_cloud(filepath, pcd)
            print(f"      Saved {filename} ({len(pcd.points)} points)")

        self.bounding_boxes_by_id = {
            value["id"]: value for value in bounding_boxes.values()
        }

        # Save bounding boxes to JSON
        bbox_filepath = os.path.join(save_dir, "bounding_boxes.json")
        with open(bbox_filepath, 'w') as f:
            json.dump(bounding_boxes, f, indent=2)
        print(f"    Saved bounding_boxes.json ({len(bounding_boxes)} objects)")

        # Step 5: Save stitched scene point cloud with rotation and shift applied
        if scene_points is not None and len(scene_points) > 0:
            print(f"    Saving stitched scene point cloud ({len(scene_points)} points)...")

            # Apply rotation
            if has_rotation:
                scene_points_rotated = (rotation_matrix @ scene_points.T).T
            else:
                scene_points_rotated = scene_points.copy()

            # Apply flip if needed
            if flip_z:
                scene_points_rotated = (flip_matrix @ scene_points_rotated.T).T

            # Apply z shift
            scene_points_rotated[:, 2] += z_shift

            # Downsample scene for memory efficiency
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(scene_points_rotated)

            if scene_colors is not None and len(scene_colors) == len(scene_points):
                if scene_colors.max() > 1.0:
                    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors / 255.0)
                else:
                    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)

            # Downsample to reduce file size and memory usage
            if self.config.scene_downsample_voxel_size > 0:
                scene_pcd_downsampled = scene_pcd.voxel_down_sample(self.config.scene_downsample_voxel_size)
                print(f"    Downsampled scene from {len(scene_pcd.points)} to {len(scene_pcd_downsampled.points)} points (voxel size: {self.config.scene_downsample_voxel_size}m)")
            else:
                scene_pcd_downsampled = scene_pcd

            scene_filepath = os.path.join(save_dir, "scene_stitched.ply")
            o3d.io.write_point_cloud(scene_filepath, scene_pcd_downsampled)
            print(f"      Saved scene_stitched.ply ({len(scene_pcd_downsampled.points)} points)")

        # Return transformation parameters for camera extrinsics
        return {
            'rotation_matrix': rotation_matrix.tolist() if has_rotation else np.eye(3).tolist(),
            'flip_z': flip_z,
            'flip_matrix': flip_matrix.tolist() if flip_z else np.eye(3).tolist(),
            'z_shift': float(z_shift),
            'has_rotation': has_rotation
        }
