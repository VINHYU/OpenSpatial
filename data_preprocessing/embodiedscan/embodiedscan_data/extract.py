import json
import logging
import os
import time
from multiprocessing import Pool
from typing import Optional

from tqdm import tqdm

from embodiedscan_data.datasets import get_dataset_config, ALL_DATASETS
from embodiedscan_data.datasets.base import DatasetConfig

logger = logging.getLogger(__name__)

# Global reference for worker processes
_worker_explorer = None
_worker_config = None
_worker_data_root = None


def _init_worker(config_name: str, data_root: str):
    """Initialize Explorer in each worker process."""
    global _worker_explorer, _worker_config, _worker_data_root
    from embodiedscan_data.explorer import ExtendedExplorer

    _worker_config = get_dataset_config(config_name)
    _worker_data_root = data_root
    explorer_kwargs = _worker_config.get_explorer_kwargs(data_root)
    _worker_explorer = ExtendedExplorer(**explorer_kwargs)


def _process_single(args):
    """Process a single (scene, camera) pair in a worker."""
    scene, camera = args
    try:
        save_path = _worker_config.get_save_path(_worker_data_root, scene)
        os.makedirs(save_path, exist_ok=True)
        project_root = os.path.dirname(os.path.abspath(_worker_data_root))
        info = _worker_explorer.get_info(
            scene, camera, render_box=True,
            save_path=save_path,
            root_path=project_root,
        )
        if info is None:
            return None

        # Inject pipeline fields
        info["dataset"] = _worker_config.name
        info["scene_id"] = _worker_config.get_scene_id(scene)
        info["depth_scale"] = _worker_config.depth_scale

        # Get intrinsic
        try:
            info["intrinsic"] = _worker_config.get_intrinsic(_worker_data_root, scene, camera)
        except Exception as e:
            logger.warning("Failed to get intrinsic for %s/%s: %s", scene, camera, e)
            return None

        # Get depth map (override if config provides it)
        depth_map = _worker_config.get_depth_map(_worker_data_root, scene, camera)
        if depth_map is not None:
            info["depth_map"] = depth_map

        # Post-process
        info = _worker_config.post_process(info, _worker_data_root, scene, camera)

        # Convert paths to relative
        for field in ("image", "depth_map", "intrinsic", "pose", "axis_align_matrix"):
            val = info.get(field)
            if val and isinstance(val, str) and os.path.isabs(val):
                info[field] = os.path.relpath(val, _worker_data_root)

        return info
    except Exception as e:
        logger.warning("Error processing %s/%s: %s", scene, camera, e)
        return None


def extract_dataset(
    dataset_name: str,
    data_root: str,
    output_dir: str,
    workers: int = 24,
    max_scenes: Optional[int] = None,
) -> str:
    """Extract per-image info for a dataset.

    Args:
        dataset_name: One of "scannet", "3rscan", "matterport3d", "arkitscenes"
        data_root: Root data directory
        output_dir: Output directory for JSONL
        workers: Number of parallel workers
        max_scenes: Limit number of scenes (for smoke testing)

    Returns:
        Path to output JSONL file
    """
    config = get_dataset_config(dataset_name)
    output_path = os.path.join(output_dir, f"{config.name}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # Resume: load existing IDs
    existing_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line.strip()).get("id"))
                except (json.JSONDecodeError, AttributeError):
                    pass
        logger.info("Resume: found %d existing records", len(existing_ids))

    # Collect tasks
    logger.info("Collecting scenes for %s...", dataset_name)
    scenes = config.list_scenes(data_root)
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    tasks = []
    for scene in scenes:
        if config.skip_scene(data_root, scene):
            continue
        cameras = config.list_cameras(data_root, scene)
        for camera in cameras:
            if config.skip_camera(data_root, scene, camera):
                continue
            tasks.append((scene, camera))

    logger.info("Found %d tasks across %d scenes", len(tasks), len(scenes))

    if not tasks:
        logger.warning("No tasks found for %s", dataset_name)
        return output_path

    # Process in parallel
    results = []
    failed = 0
    start = time.time()

    with Pool(processes=workers, initializer=_init_worker, initargs=(dataset_name, data_root)) as pool:
        pbar = tqdm(total=len(tasks), desc=f"Extracting {dataset_name}")
        try:
            for result in pool.imap_unordered(_process_single, tasks):
                if result is not None:
                    if result.get("id") not in existing_ids:
                        results.append(result)
                else:
                    failed += 1
                pbar.update(1)
        except KeyboardInterrupt:
            logger.info("Interrupted, saving partial results...")
            pool.terminate()
            pool.join()
        finally:
            pbar.close()

    elapsed = time.time() - start

    # Append results
    if results:
        with open(output_path, "a", encoding="utf-8") as f:
            for info in results:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  New results: {len(results)}")
    print(f"  Skipped (existing): {len(existing_ids)}")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")

    return output_path
