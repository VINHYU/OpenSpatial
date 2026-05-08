"""Extends upstream EmbodiedScanExplorer with `get_info` for per-frame
metadata extraction. Lives here so the data pipeline runs against vanilla
upstream `OpenRobotLab/EmbodiedScan` without requiring a private fork.
"""
import os

import cv2
import numpy as np

from embodiedscan.explorer import EmbodiedScanExplorer
from embodiedscan.visualization.utils import _9dof_to_box


class ExtendedExplorer(EmbodiedScanExplorer):
    """EmbodiedScanExplorer + per-frame `get_info` extraction."""

    def get_info(self, scene_name, camera_name, render_box=False,
                 save_path=None, root_path=None):
        """Extract per-frame info dict.

        Returns:
            dict or None: None if (scene, camera) not found.
        """
        dataset = scene_name.split('/')[0]
        select = None
        info = None
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                select = scene
        if select is None:
            return None
        for camera in select['images']:
            img_path = camera['img_path']
            img_path = os.path.join(self.data_root[dataset],
                                    img_path[img_path.find('/') + 1:])
            if dataset == 'scannet':
                cam_name = img_path.split('/')[-1][:-4]
            elif dataset == '3rscan':
                cam_name = img_path.split('/')[-1][:-10]
            elif dataset == 'matterport3d':
                cam_name = img_path.split('/')[-1][:-8] + img_path.split(
                    '/')[-1][-7:-4]
            else:
                cam_name = img_path.split('/')[-1][:-4]

            if cam_name == camera_name:
                info = dict()
                info['id'] = scene_name.replace('/', '__') + '__' + camera_name
                info['image'] = None
                info['pose'] = None
                info['intrinsic'] = None
                info['obj_tags'] = []
                info['depth_map'] = None
                info['bboxes_3d_world_coords'] = []

                info['image'] = os.path.join(root_path, img_path)
                if '3rscan' in scene_name:
                    depth_image_path = img_path.replace('.color.jpg', '.depth.pgm')
                    depth_map_path = img_path.replace('.color.jpg', '_depth_map.npy')
                    self._convert_pgm_to_npy(depth_image_path, depth_map_path)
                    depth_map_path = os.path.join(root_path, depth_map_path)
                elif 'matterport3d' in scene_name:
                    depth_map_path = None
                else:
                    depth_map_path = img_path.replace('.jpg', '.png')
                    depth_map_path = os.path.join(root_path, depth_map_path)
                info['depth_map'] = depth_map_path

                axis_align_matrix = select['axis_align_matrix']
                extrinsic = axis_align_matrix @ camera['cam2global']
                if 'cam2img' in camera:
                    intrinsic = camera['cam2img']
                else:
                    intrinsic = select['cam2img']
                if '3rscan' in scene_name:
                    extrinsic_path = img_path.replace('.color.jpg', '_pose.txt')
                    intrinsic_path = img_path.replace('.color.jpg', '_intrinsic.txt')
                    axis_align_matrix_path = img_path.replace('.color.jpg', '_axis_align_matrix.txt')
                    extrinsic_path = os.path.join(root_path, extrinsic_path)
                    intrinsic_path = os.path.join(root_path, intrinsic_path)
                    axis_align_matrix_path = os.path.join(root_path, axis_align_matrix_path)
                elif 'matterport3d' in scene_name:
                    extrinsic_path = img_path.replace('.jpg', '_pose.txt')
                    intrinsic_path = img_path.replace('.jpg', '_intrinsic_matrix.txt')
                    axis_align_matrix_path = img_path.replace('.jpg', '_axis_align_matrix.txt')
                    extrinsic_path = os.path.join(root_path, extrinsic_path)
                    intrinsic_path = os.path.join(root_path, intrinsic_path)
                    axis_align_matrix_path = os.path.join(root_path, axis_align_matrix_path)
                else:
                    extrinsic_path = os.path.join(save_path, camera_name + '_pose.txt')
                    intrinsic_path = os.path.join(save_path, camera_name + '_intrinsic.txt')
                    axis_align_matrix_path = os.path.join(save_path, camera_name + '_axis_align_matrix.txt')
                np.savetxt(extrinsic_path, extrinsic, fmt='%.9f')
                np.savetxt(intrinsic_path, intrinsic, fmt='%.9f')
                np.savetxt(axis_align_matrix_path, axis_align_matrix, fmt='%.9f')
                info['pose'] = extrinsic_path
                info['intrinsic'] = intrinsic_path
                info['axis_align_matrix'] = axis_align_matrix_path

                if render_box:
                    if self.verbose:
                        print('Rendering box')
                    try:
                        for i in camera['visible_instance_ids']:
                            instance = select['instances'][i]
                            box = _9dof_to_box(
                                instance['bbox_3d'],
                                self.classes[self.id_to_index[
                                    instance['bbox_label_3d']]],
                                self.color_selector,
                            )
                            label = self.classes[self.id_to_index[
                                instance['bbox_label_3d']]]
                            info['obj_tags'].append(label)
                            info['bboxes_3d_world_coords'].append(
                                instance['bbox_3d'])
                    except Exception as e:
                        print(f'Rendering box failed: {e}, skip this camera')
                        continue
                    if self.verbose:
                        print('Rendering complete')

        return info

    def _convert_pgm_to_npy(self, pgm_path, npy_path):
        """Decode a 16-bit PGM depth image into a .npy array (used by 3rscan)."""
        try:
            depth_image = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise IOError(f'failed to read {pgm_path}')
            np.save(npy_path, depth_image)
        except Exception as e:
            print(f'PGM->NPY conversion failed for {pgm_path}: {e}')
