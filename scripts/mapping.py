import argparse
import os
import shutil
import numpy as np
import pycolmap
import tempfile
import cv2
import open3d as o3d
from pathlib import Path
from autolabel.utils import Scene, transform_points, Camera
from autolabel.undistort import ImageUndistorter
from hloc import (extract_features, match_features, reconstruction,
                  pairs_from_exhaustive, pairs_from_retrieval)
from hloc.utils import viz_3d


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help="Scene to infer poses for.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--vis', action='store_true')
    return parser.parse_args()


class HLoc:

    def __init__(self, tmp_dir, scene, flags):
        self.flags = flags
        self.scene = scene
        self.scene_path = Path(self.scene.path)
        self.exhaustive = len((self.scene.raw_rgb_paths())) < 250

        self.tmp_dir = Path(tmp_dir)
        self.sfm_pairs = self.tmp_dir / 'sfm-pairs.txt'
        self.loc_pairs = self.tmp_dir / 'sfm-pairs-loc.txt'
        self.features = self.tmp_dir / 'features.h5'
        self.matches = self.tmp_dir / 'matches.h5'
        self.feature_conf = extract_features.confs['superpoint_aachen']
        self.retrieval_conf = extract_features.confs['netvlad']
        self.matcher_conf = match_features.confs['superglue']

    def _run_sfm(self):
        image_dir = Path(self.scene.path) / 'raw_rgb'
        image_list = []
        image_paths = self.scene.raw_rgb_paths()
        image_list_path = []
        indices = np.arange(len(image_paths))
        for index in indices:
            image_list.append(image_paths[index])
            image_list_path.append(
                str(Path(image_paths[index]).relative_to(image_dir)))
        if self.exhaustive:
            extract_features.main(self.feature_conf,
                                  image_dir,
                                  feature_path=self.features,
                                  image_list=image_list_path)
            pairs_from_exhaustive.main(self.sfm_pairs,
                                       image_list=image_list_path)
            match_features.main(self.matcher_conf,
                                self.sfm_pairs,
                                features=self.features,
                                matches=self.matches)
            model = reconstruction.main(
                self.tmp_dir,
                image_dir,
                self.sfm_pairs,
                self.features,
                self.matches,
                image_list=image_list_path,
                camera_mode=pycolmap.CameraMode.SINGLE,
                image_options={'camera_model': "OPENCV"},
                mapper_options={
                    'ba_refine_principal_point': True,
                    'ba_refine_extra_params': True,
                    'ba_refine_focal_length': True
                })
        else:
            retrieval_path = extract_features.main(self.retrieval_conf,
                                                   image_dir,
                                                   self.tmp_dir,
                                                   image_list=image_list_path)
            pairs_from_retrieval.main(retrieval_path,
                                      self.sfm_pairs,
                                      num_matched=50)
            feature_path = extract_features.main(self.feature_conf,
                                                 image_dir,
                                                 self.tmp_dir,
                                                 image_list=image_list_path)
            match_path = match_features.main(self.matcher_conf,
                                             self.sfm_pairs,
                                             self.feature_conf['output'],
                                             self.tmp_dir,
                                             matches=self.matches)
            model = reconstruction.main(
                self.tmp_dir,
                image_dir,
                self.sfm_pairs,
                feature_path,
                match_path,
                image_list=image_list_path,
                camera_mode=pycolmap.CameraMode.SINGLE,
                image_options={'camera_model': "OPENCV"},
                mapper_options={
                    'ba_refine_principal_point': True,
                    'ba_refine_extra_params': True,
                    'ba_refine_focal_length': True
                })

        if self.flags.vis:
            fig = viz_3d.init_figure()
            viz_3d.plot_reconstruction(fig,
                                       model,
                                       color='rgba(255,0,0,0.5)',
                                       name="mapping")
            fig.show()

        if self.flags.debug:
            # Save mapping metadata if running in debug mode.
            colmap_output_dir = os.path.join(self.scene.path, 'colmap_output')
            os.makedirs(colmap_output_dir, exist_ok=True)
            model.write_text(colmap_output_dir)

        # Save the intrinsics matrix and the distortion parameters.
        assert (len(model.cameras) == 1 and 1 in model.cameras)
        (focal_length_x, focal_length_y, c_x, c_y, k_1, k_2, p_1,
         p_2) = model.cameras[1].params
        self.colmap_K = np.eye(3)
        self.colmap_K[0, 0] = focal_length_x
        self.colmap_K[1, 1] = focal_length_y
        self.colmap_K[0, 2] = c_x
        self.colmap_K[1, 2] = c_y
        self.colmap_distortion_params = np.array([k_1, k_2, p_1, p_2])
        np.savetxt(fname=os.path.join(self.scene.path, 'intrinsics.txt'),
                   X=self.colmap_K)
        np.savetxt(fname=os.path.join(self.scene.path,
                                      'distortion_parameters.txt'),
                   X=self.colmap_distortion_params)

    def _undistort_images(self):
        print("Undistorting images according to the estimated intrinsics...")
        undistorted_image_folder = os.path.join(self.scene.path, "rgb")
        undistorted_depth_folder = os.path.join(self.scene.path, "depth")
        os.makedirs(undistorted_image_folder, exist_ok=True)
        os.makedirs(undistorted_depth_folder, exist_ok=True)

        color_undistorter = ImageUndistorter(K=self.colmap_K,
                                             D=self.colmap_distortion_params,
                                             H=self.scene.camera.size[1],
                                             W=self.scene.camera.size[0])

        depth_camera = Camera(self.colmap_K, self.scene.camera.size).scale(
            self.scene.depth_size())
        depth_undistorter = ImageUndistorter(K=depth_camera.camera_matrix,
                                             D=self.colmap_distortion_params,
                                             H=depth_camera.size[1],
                                             W=depth_camera.size[0])

        # Undistort all the images and save the undistorted versions.
        image_paths = self.scene.raw_rgb_paths()
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            undistorted_image = color_undistorter.undistort_image(image=image)
            cv2.imwrite(img=undistorted_image,
                        filename=os.path.join(undistorted_image_folder,
                                              os.path.basename(image_path)))

        depth_paths = self.scene.raw_depth_paths()
        for depth_path in depth_paths:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            undistorted_depth = depth_undistorter.undistort_image(image=depth)
            cv2.imwrite(img=undistorted_depth,
                        filename=os.path.join(undistorted_depth_folder,
                                              os.path.basename(depth_path)))

    def run(self):
        self._run_sfm()
        self._undistort_images()


class ScaleEstimation:
    min_depth = 0.05

    def __init__(self, scene, colmap_dir):
        self.scene = scene
        self.colmap_dir = colmap_dir
        self.reconstruction = pycolmap.Reconstruction(colmap_dir)
        self._read_trajectory()
        self._read_depth_maps()

    def _read_depth_maps(self):
        self.depth_maps = {}
        for path in self.scene.depth_paths():
            frame_name = os.path.basename(path).split('.')[0]
            self.depth_maps[frame_name] = cv2.imread(path, -1) / 1000.0
        depth_shape = next(iter(self.depth_maps.values())).shape
        depth_size = np.array([depth_shape[1], depth_shape[0]],
                              dtype=np.float64)
        self.depth_to_color_ratio = depth_size / np.array(
            self.scene.camera.size, dtype=np.float64)

    def _read_trajectory(self):
        poses = []
        for image in self.reconstruction.images.values():
            T_CW = np.eye(4)
            T_CW[:3, :3] = image.rotmat()
            T_CW[:3, 3] = image.tvec
            frame_name = image.name.split('.')[0]
            poses.append((frame_name, T_CW))
        self.poses = dict(poses)

    def _lookup_depth(self, frame, xy):
        xy_depth = np.floor(self.depth_to_color_ratio * xy).astype(int)
        return self.depth_maps[frame][xy_depth[1], xy_depth[0]]

    def _estimate_scale(self):
        images = self.reconstruction.images
        point_depths = []
        measured_depths = []
        for image in images.values():
            frame_name = image.name.split('.')[0]
            points = image.get_valid_points2D()
            points3D = self.reconstruction.points3D
            for point in points:
                depth_map_value = self._lookup_depth(frame_name, point.xy)

                if depth_map_value < self.min_depth:
                    continue

                T_CW = self.poses[frame_name]
                point3D = points3D[point.point3D_id]

                p_C = transform_points(T_CW, point3D.xyz)
                measured_depths.append(depth_map_value)
                point_depths.append(p_C[2])

        point_depths = np.stack(point_depths)
        measured_depths = np.stack(measured_depths)
        scales = measured_depths / point_depths
        return self._ransac(scales)

    def _ransac(self, scales):
        best_set = None
        best_inlier_count = 0
        indices = np.arange(0, scales.shape[0])
        inlier_threshold = np.median(scales) * 1e-2
        for i in range(10000):
            selected = np.random.choice(indices)
            estimate = scales[selected]
            inliers = np.abs(scales - estimate) < inlier_threshold
            inlier_count = inliers.sum()
            if inlier_count > best_inlier_count:
                best_set = scales[inliers]
                best_inlier_count = inlier_count
        print(
            f"Scale estimation inlier count: {best_inlier_count} / {scales.size}"
        )
        return np.mean(best_set)

    def _scale_poses(self, ratio):
        scaled_poses = {}
        for key, pose in self.poses.items():
            new_pose = pose.copy()
            new_pose[:3, 3] *= ratio
            scaled_poses[key] = new_pose
        return scaled_poses

    def run(self):
        scale_ratio = self._estimate_scale()
        return self._scale_poses(scale_ratio)


class PoseSaver:

    def __init__(self, scene, scaled_poses):
        self.scene = scene
        self.poses = scaled_poses

    def compute_bbox(self, poses):
        """
        poses: Metrically scaled transforms from camera to world frame.
        """
        # Compute axis-aligned bounding box of the depth values in world frame.
        # Then get the center.
        min_bounds = np.zeros(3)
        max_bounds = np.zeros(3)
        depth_frame = o3d.io.read_image(self.scene.depth_paths()[0])
        depth_size = np.asarray(depth_frame).shape[::-1]
        K = self.scene.camera.scale(depth_size).camera_matrix
        intrinsics = o3d.camera.PinholeCameraIntrinsic(int(depth_size[0]),
                                                       int(depth_size[1]),
                                                       K[0, 0], K[1, 1],
                                                       K[0, 2], K[1, 2])
        pc = o3d.geometry.PointCloud()
        depth_frames = dict([(os.path.basename(p).split('.')[0], p)
                             for p in self.scene.depth_paths()])
        items = [item for item in poses.items()]
        stride = max(len(self.scene.depth_paths()) // 100, 1)
        for key, T_WC in items[::stride]:
            if key not in depth_frames:
                print("WARNING: Can't find depth image {key}.png")
                continue
            depth = o3d.io.read_image(f"{depth_frames[key]}")

            pc_C = o3d.geometry.PointCloud.create_from_depth_image(
                depth, depth_scale=1000.0, intrinsic=intrinsics)
            pc_C = np.asarray(pc_C.points)
            pc_W = transform_points(T_WC, pc_C)

            min_bounds = np.minimum(min_bounds, pc_W.min(axis=0))
            max_bounds = np.maximum(max_bounds, pc_W.max(axis=0))
            pc += o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pc_W)).uniform_down_sample(50)

        filtered, _ = pc.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        bbox = filtered.get_oriented_bounding_box(robust=True)
        T = np.eye(4)
        T[:3, :3] = bbox.R.T
        o3d_aabb = o3d.geometry.PointCloud(filtered).transform(
            T).get_axis_aligned_bounding_box()
        center = o3d_aabb.get_center()
        T[:3, 3] = -center
        aabb = np.zeros((2, 3))
        aabb[0, :] = o3d_aabb.get_min_bound() - center
        aabb[1, :] = o3d_aabb.get_max_bound() - center
        return T, aabb, filtered

    def _write_poses(self, poses):
        pose_dir = os.path.join(self.scene.path, 'pose')
        os.makedirs(pose_dir, exist_ok=True)
        for key, T_CW in poses.items():
            pose_file = os.path.join(pose_dir, f'{key}.txt')
            np.savetxt(pose_file, T_CW)

    def _write_bounds(self, bounds):
        with open(os.path.join(self.scene.path, 'bbox.txt'), 'wt') as f:
            min_str = " ".join([str(x) for x in bounds[0]])
            max_str = " ".join([str(x) for x in bounds[1]])
            f.write(f"{min_str} {max_str} 0.01")

    def run(self):
        T_WCs = {}
        for key, T_CW in self.poses.items():
            T_WCs[key] = np.linalg.inv(T_CW)
        T, aabb, point_cloud = self.compute_bbox(T_WCs)

        T_CWs = {}
        for key, T_WC in T_WCs.items():
            T_CWs[key] = np.linalg.inv(T @ T_WC)
        self._write_poses(T_CWs)
        self._write_bounds(aabb)


class Pipeline:

    def __init__(self, flags):
        self.tmp_dir = tempfile.mkdtemp()
        self.flags = flags
        self.scene = Scene(flags.scene)

    def run(self):
        hloc = HLoc(self.tmp_dir, self.scene, self.flags)
        hloc.run()

        # Camera intrinsics might have changed so reload the scene.
        self.scene = Scene(self.scene.path)

        scale_estimation = ScaleEstimation(self.scene, self.tmp_dir)
        scaled_poses = scale_estimation.run()
        pose_saver = PoseSaver(self.scene, scaled_poses)
        pose_saver.run()

        if self.flags.debug:
            shutil.move(str(self.tmp_dir), "/tmp/sfm_debug")
        else:
            shutil.rmtree(self.tmp_dir)


if __name__ == "__main__":
    Pipeline(read_args()).run()
