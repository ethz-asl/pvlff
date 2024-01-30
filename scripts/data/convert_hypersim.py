"""
Converts hypersim scenes from https://github.com/apple/ml-hypersim
to the autolabel scene format.

usage:
    python scripts/data/convert_hypersim.py <hypersim folder> \
        --out <output-scene-directory> \
        --ori-semantic-labels <hypersim-semantic-labels-file> \
        --camera-parameter-file <csvfile-of-camera>
"""
import pandas as pd
import argparse
import cv2
import json
import math
import numpy as np
import open3d as o3d
import os
import glob
from natsort import os_sorted
import shutil
from tqdm import tqdm
import h5py


from autolabel.utils import Scene, transform_points


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ori-semantic-labels", type=str, required=True)
    parser.add_argument("--camera-parameter-file", type=str, required=True)
    return parser.parse_args()

def load_distance_meters_to_depth(hdf_file, width=1024, height=768, focal=886.81):
    with h5py.File(hdf_file, "r") as f:
        depth_meters = f["dataset"][:].astype(np.float32)
    
    npyImageplaneX = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([height, width, 1], focal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = depth_meters / np.linalg.norm(npyImageplane, 2, 2) * focal
    return npyDepth

def load_camera_poses(hdf_orientation, hdf_position, scale):
    
    with h5py.File(hdf_orientation, "r") as f:
        orientations = f["dataset"][:].astype(np.float32)

    with h5py.File(hdf_position, "r") as f:
        positions = f["dataset"][:].astype(np.float32)
    
    positions *= scale

    poses = []
    trans = np.eye(3)
    trans[1, 1] = -1
    trans[2, 2] = -1
    
    for orientation, position in zip(orientations, positions):
        T_WC = np.eye(4)
        T_WC[:3, :3] = orientation @ trans
        T_WC[:3, 3] = position
        poses.append(T_WC)
    return poses


class SceneConverter:

    def __init__(self, scene, out_scene, camera_settings, semantic_label_mapping):
        self.out_scene = out_scene
        self.in_scene = scene
        self._load_camera(camera_settings)
        self.semantic_label_mapping = pd.read_csv(semantic_label_mapping) # NYU 40 classes

        self._load_meta_data()

    def _load_camera(self, camera_settings):
        height = camera_settings['settings_output_img_height']
        width = camera_settings['settings_output_img_width']
        rate_unit_to_meter = camera_settings['settings_units_info_meters_scale']
        fov_x = camera_settings['settings_camera_fov']
        fx = width / 2.0 / math.tan(fov_x / 2)

        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        intrinsic = np.eye(3)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fx
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        self.camera = {
            'height': int(height), 'width': int(width),
            'rate_unit_to_meter': rate_unit_to_meter,
            'focal_length': fx,
            'intrinsic': intrinsic
        }
    
    def _load_meta_data(self):
        cam_list = pd.read_csv(os.path.join(self.in_scene, '_detail', 'metadata_cameras.csv'))
        cam_list = cam_list['camera_name'].values.tolist()

        self.meta_data = {
            'cam_list': cam_list
        }

    def _save_scene_metadata(self):
        metadata = { 'n_classes': int(self.semantic_label_mapping['id'].max()) }
        metadata_path = os.path.join(self.out_scene, 'metadata.json')
        with open(metadata_path, 'w') as f:
            f.write(json.dumps(metadata, indent=2))

    def _collect_paths(self, cam):
        rgb_path = os.path.join(self.in_scene, 'images', f'scene_{cam}_final_preview', 'frame.*.color.jpg')
        depth_path = os.path.join(self.in_scene, 'images', f'scene_{cam}_geometry_hdf5', 'frame.*.depth_meters.hdf5')
        semantic_path = os.path.join(self.in_scene, 'images', f'scene_{cam}_geometry_hdf5', 'frame.*.semantic.hdf5')
        instance_path = os.path.join(self.in_scene, 'images', f'scene_{cam}_geometry_hdf5', 'frame.*.semantic_instance.hdf5')
        
        rgb_frames = glob.glob(rgb_path)
        depth_frames = glob.glob(depth_path)
        semantic_frames = glob.glob(semantic_path)
        instance_frames = glob.glob(instance_path)
        
        rgb_frames = os_sorted(rgb_frames)
        depth_frames = os_sorted(depth_frames)
        semantic_frames = os_sorted(semantic_frames)
        instance_frames = os_sorted(instance_frames)

        poses_T_WC = load_camera_poses(
            hdf_orientation=os.path.join(self.in_scene, '_detail', f'{cam}', 'camera_keyframe_orientations.hdf5'),
            hdf_position=os.path.join(self.in_scene, '_detail', f'{cam}', 'camera_keyframe_positions.hdf5'),
            scale=self.camera['rate_unit_to_meter']
        )
        return rgb_frames, depth_frames, semantic_frames, instance_frames, poses_T_WC

    def _copy_frames_and_trajectory(self, cam, rgb_frames, depth_frames, semantic_frames, instance_frames, poses_T_WC):
        rgb_out = os.path.join(self.out_scene, 'rgb')
        depth_out = os.path.join(self.out_scene, 'depth')
        semantic_out = os.path.join(self.out_scene, 'gt_semantic')
        instance_out = os.path.join(self.out_scene, 'gt_instance')
        os.makedirs(rgb_out, exist_ok=True)
        os.makedirs(depth_out, exist_ok=True)
        os.makedirs(semantic_out, exist_ok=True)
        os.makedirs(instance_out, exist_ok=True)

        pose_dir = os.path.join(self.out_scene, 'pose')
        os.makedirs(pose_dir, exist_ok=True)

        for (rgb, depth, semantic, instance, pose_T_WC) in zip(tqdm(rgb_frames, desc=f"Copying {cam} frames"),
                                                        depth_frames, semantic_frames, instance_frames, poses_T_WC):
            rgb_out_path = os.path.join(rgb_out, f"{self.frame_index:06}.jpg")
            depth_out_path = os.path.join(depth_out, f"{self.frame_index:06}.png")
            semantic_out_path = os.path.join(semantic_out, f"{self.frame_index:06}.png")
            instance_out_path = os.path.join(instance_out, f"{self.frame_index:06}.png")
            
            shutil.copy(rgb, rgb_out_path)

            depth_img = load_distance_meters_to_depth(
                depth, self.camera['width'], self.camera['height'], self.camera['focal_length'])
            depth_img = (depth_img * 1000).astype(np.uint16)
            cv2.imwrite(depth_out_path, depth_img)

            with h5py.File(semantic, "r") as f:
                semantic_img = f["dataset"][:].astype(np.int16)
            semantic_img = (semantic_img + 1).astype(np.uint16)
            cv2.imwrite(semantic_out_path, semantic_img)
            
            with h5py.File(instance, "r") as f:
                instance_img = f["dataset"][:].astype(np.int16)
            instance_img = (instance_img + 1).astype(np.uint16)
            cv2.imwrite(instance_out_path, instance_img)

            pose_out_path = os.path.join(pose_dir, f"{self.frame_index:06}.txt")
            np.savetxt(pose_out_path, np.linalg.inv(pose_T_WC))
            
            self.frame_index += 1

    def _copy_intrinsics(self):
        np.savetxt(os.path.join(self.out_scene, 'intrinsics.txt'), self.camera['intrinsic'])

    def _compute_bounds(self):
        scene = Scene(self.out_scene)
        depth_frame = o3d.io.read_image(scene.depth_paths()[0])
        depth_size = np.asarray(depth_frame).shape[::-1]
        K = scene.camera.scale(depth_size).camera_matrix
        intrinsics = o3d.camera.PinholeCameraIntrinsic(int(depth_size[0]),
                                                       int(depth_size[1]),
                                                       K[0, 0], K[1, 1],
                                                       K[0, 2], K[1, 2])
        pc = o3d.geometry.PointCloud()

        poses = scene.poses#[::10]
        depths = scene.depth_paths()#[::10]
        for T_CW, depth in zip(poses, tqdm(depths, desc="Computing bounds")):
            T_WC = np.linalg.inv(T_CW)
            depth = o3d.io.read_image(depth)

            pc_C = o3d.geometry.PointCloud.create_from_depth_image(
                depth, depth_scale=1000.0, intrinsic=intrinsics)
            pc_C = np.asarray(pc_C.points)
            pc_W = transform_points(T_WC, pc_C)

            pc += o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pc_W)).uniform_down_sample(50)
        filtered, _ = pc.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        aabb = filtered.get_axis_aligned_bounding_box()
        with open(os.path.join(scene.path, 'bbox.txt'), 'wt') as f:
            min_str = " ".join([str(x) for x in aabb.get_min_bound()])
            max_str = " ".join([str(x) for x in aabb.get_max_bound()])
            f.write(f"{min_str} {max_str} 0.01")

    def run(self):
        self._save_scene_metadata()
        self._copy_intrinsics()

        self.frame_index = 0
        for cam in self.meta_data['cam_list']:
            rgb_frames, depth_frames, semantic_frames, instance_frames, poses_T_WC = self._collect_paths(cam)
            self._copy_frames_and_trajectory(cam, rgb_frames, depth_frames, semantic_frames, instance_frames, poses_T_WC)

        self._compute_bounds()


def create_labelmap(semantic_labels, out):
    semantic_labels = pd.read_csv(semantic_labels)
    ids = []
    prompts = []
    for semantic_id, semantic_name in zip(semantic_labels['semantic_id '], semantic_labels[' semantic_name  ']):
        ids.append(semantic_id + 1)
        prompts.append(semantic_name)
    data = pd.DataFrame({'id': ids, 'name': prompts})
    data.to_csv(out, index=False)
    return data

def main():
    flags = read_args()

    os.makedirs(flags.out, exist_ok=True)

    label_map = create_labelmap(
        flags.ori_semantic_labels,
        os.path.join(flags.out, 'label_map.csv')
    )

    all_camera_settings = pd.read_csv(flags.camera_parameter_file, 
                                      index_col="scene_name")

    scene_names = os.listdir(flags.dataset)

    for scene_name in scene_names:
        print(f"Converting scene [{scene_name}] ...")
        in_scene = os.path.join(flags.dataset, scene_name)
        out_scene = os.path.join(flags.out, scene_name)
        os.makedirs(out_scene, exist_ok=True)

        converter = SceneConverter(
            scene=in_scene, out_scene=out_scene, 
            camera_settings=all_camera_settings.loc[scene_name], 
            semantic_label_mapping=os.path.join(flags.out, 'label_map.csv'))
        converter.run()


if __name__ == "__main__":
    main()