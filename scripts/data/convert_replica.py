"""
Converts rendered replica scenes from https://github.com/Harry-Zhi/semantic_nerf
to the autolabel scene format.

usage:
    python scripts/data/convert_replica.py <replica folder> --out <output-scene-directory>
"""
import pandas
import argparse
import cv2
import json
import tempfile
import math
import numpy as np
import open3d as o3d
import os
import shutil
import subprocess
from tqdm import tqdm

from autolabel.utils import Scene, transform_points


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


class SceneConverter:

    def __init__(self, scene, out_scene, metadata):
        self.out_scene = out_scene
        self.in_scene = scene
        self.metadata = metadata
        self._collect_paths()

    def _collect_paths(self):
        rgb_path = os.path.join(self.in_scene, 'rgb')
        depth_path = os.path.join(self.in_scene, 'depth')
        semantic_path = os.path.join(self.in_scene, 'semantic_class')
        instance_path = os.path.join(self.in_scene, 'instance')
        rgb_frames = [f for f in os.listdir(rgb_path) if f[0] != '.']
        depth_frames = [f for f in os.listdir(depth_path) if f[0] != '.']
        semantic_frames = [
            f for f in os.listdir(semantic_path)
            if f[0] != '.' and 'semantic' in f
        ]
        instance_frames = [
            f for f in os.listdir(instance_path)
            if f[0] != '.' and 'semantic_instance' in f
        ]
        rgb_frames = sorted(rgb_frames,
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        depth_frames = sorted(depth_frames,
                              key=lambda x: int(x.split('_')[-1].split('.')[0]))
        semantic_frames = sorted(
            semantic_frames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        instance_frames = sorted(
            instance_frames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.rgb_frames = []
        self.depth_frames = []
        self.semantic_frames = []
        self.instance_frames = []
        for rgb, depth, semantic, instance in zip(rgb_frames, depth_frames,
                                        semantic_frames, instance_frames):
            self.rgb_frames.append(os.path.join(rgb_path, rgb))
            self.depth_frames.append(os.path.join(depth_path, depth))
            self.semantic_frames.append(os.path.join(semantic_path, semantic))
            self.instance_frames.append(os.path.join(instance_path, instance))

    def _copy_frames(self):
        self.rgb_out = os.path.join(self.out_scene, 'rgb')
        self.depth_out = os.path.join(self.out_scene, 'depth')
        self.semantic_out = os.path.join(self.out_scene, 'gt_semantic')
        self.instance_out = os.path.join(self.out_scene, 'gt_instance')
        os.makedirs(self.rgb_out, exist_ok=True)
        os.makedirs(self.depth_out, exist_ok=True)
        os.makedirs(self.semantic_out, exist_ok=True)
        os.makedirs(self.instance_out, exist_ok=True)

        semantic_frames = []
        for i, (rgb, depth, semantic, instance) in enumerate(
                zip(tqdm(self.rgb_frames, desc="Copying frames"),
                    self.depth_frames, self.semantic_frames, self.instance_frames)):
            rgb_out_path = os.path.join(self.rgb_out, f"{i:06}.png")
            depth_out_path = os.path.join(self.depth_out, f"{i:06}.png")
            semantic_out = os.path.join(self.semantic_out, f"{i:06}.png")
            instance_out_path = os.path.join(self.instance_out, f"{i:06}.png")
            shutil.copy(rgb, rgb_out_path)
            shutil.copy(depth, depth_out_path)
            shutil.copy(semantic, self.semantic_out)
            shutil.copy(instance, instance_out_path)

        metadata = { 'n_classes': int(self.metadata['id'].max() + 1) }
        metadata_path = os.path.join(self.out_scene, 'metadata.json')
        with open(metadata_path, 'w') as f:
            f.write(json.dumps(metadata, indent=2))

    def _copy_trajectory(self):
        pose_dir = os.path.join(self.out_scene, 'pose')
        os.makedirs(pose_dir, exist_ok=True)
        trajectory = np.loadtxt(os.path.join(self.in_scene, 'traj_w_c.txt'),
                                delimiter=' ').reshape(-1, 4, 4)
        for i, T_CW in enumerate(trajectory):
            pose_out = os.path.join(pose_dir, f"{i:06}.txt")
            np.savetxt(pose_out, np.linalg.inv(T_CW))

    def _copy_intrinsics(self):
        width = 640
        height = 480
        hfov = 90.0
        fx = width / 2.0 / math.tan(math.radians(hfov / 2.0))
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        camera_matrix = np.eye(3)
        camera_matrix[0, 0] = fx
        camera_matrix[1, 1] = fx
        camera_matrix[0, 2] = cx
        camera_matrix[1, 2] = cy
        np.savetxt(os.path.join(self.out_scene, 'intrinsics.txt'),
                   camera_matrix)

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

        poses = scene.poses[::10]
        depths = scene.depth_paths()[::10]
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
        self._copy_frames()
        self._copy_trajectory()
        self._copy_intrinsics()
        self._compute_bounds()

def create_labelmap(semantic_info_dir, out):
    metadata = os.path.join(semantic_info_dir, 'room_0', 'info_semantic.json')
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    ids = []
    prompts = []
    for class_info in metadata['classes']:
        ids.append(class_info['id'])
        prompts.append(class_info['name'])
    data = pandas.DataFrame({'id': ids, 'name': prompts})
    data.to_csv(out, index=False)
    return data


def main():
    flags = read_args()

    zip_files = [f for f in os.listdir(flags.dataset) if '.zip' in f]
    instance_zip = [f for f in zip_files if 'Instance' in f][0]

    tmpdir = tempfile.mkdtemp()
    try:
        success = subprocess.run(['unzip', os.path.join(flags.dataset, instance_zip), '-d', tmpdir])
        if success.returncode != 0:
            raise RuntimeError("Failed to extract instance segmentation")
        success = subprocess.run(['unzip', os.path.join(flags.dataset, 'semantic_info.zip'), '-d', tmpdir])
        if success.returncode != 0:
            raise RuntimeError("Failed to extract segmentation metadata")
        metadata = create_labelmap(os.path.join(tmpdir, 'semantic_info'), os.path.join(flags.out, 'label_map.csv'))

        for file in zip_files:
            if 'semantic_info' in file or 'Instance' in file or 'replica' in file:
                continue
            print("Extracting", file)
            scene_name = file.split('.')[0]
            tmp_scene_dir = os.path.join(tmpdir, scene_name)
            success = subprocess.run(['unzip', os.path.join(flags.dataset, file), '-d', tmpdir])
            if success.returncode != 0:
                raise RuntimeError("Failed to extract scene")
            out_scene = os.path.join(flags.out, scene_name)
            os.makedirs(out_scene, exist_ok=True)
            in_scene = os.path.join(tmp_scene_dir, 'Sequence_1')
            scene_instance_zip = os.path.join(flags.dataset, 'Replica_Instance_Segmentation', scene_name, 'Sequence_1', 'semantic_instance.zip')
            success = subprocess.run(['unzip', scene_instance_zip, '-d', tmp_scene_dir])
            if success.returncode != 0:
                raise RuntimeError("Failed to extract scene")
            success = subprocess.run(['mv', os.path.join(tmp_scene_dir, 'semantic_instance'), os.path.join(tmp_scene_dir, 'Sequence_1', 'instance')])
            if success.returncode != 0:
                raise RuntimeError("Failed to move instance folder")
            converter = SceneConverter(in_scene, out_scene, metadata)
            converter.run()
            shutil.rmtree(tmp_scene_dir)
    finally:
        shutil.rmtree(tmpdir)

    # Exporter(flags).run()

if __name__ == "__main__":
    main()
