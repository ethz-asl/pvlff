import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import torch
import pickle
import random
import h5py
from numba import njit
from PIL import Image
from autolabel.utils import Scene
from torch_ngp.nerf.provider import nerf_matrix_to_ngp

CV_TO_OPENGL = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

SAM_BIT_LEN = 32

def _load_sam_mask(file_path, image_size):
    sam_mask = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    sam_mask = cv2.resize(sam_mask, image_size, interpolation=cv2.INTER_NEAREST)
    sam_mask = sam_mask.view(np.uint32)
    return sam_mask

def _get_masks_from_sam_mask(sam_mask):
    ONE = 1
    masks = []
    areas = []
    for i in range(SAM_BIT_LEN):
        mask = np.bitwise_and(ONE << i, sam_mask).astype(bool)
        masks.append(mask)
        areas.append(mask.sum())
    return masks, areas

@njit
def _compute_direction(R_WC: np.ndarray, ray_indices: np.ndarray, w: int,
                       fx: float, fy: float, cx: float, cy: float,
                       randomize: bool) -> np.ndarray:
    directions = np.zeros((len(ray_indices), 3), dtype=np.float32)
    xs = (ray_indices % w).astype(np.float32)
    ys = ((ray_indices - xs) / w).astype(np.float32)
    if randomize:
        xs += np.random.random(ray_indices.size).astype(np.float32)
        ys += np.random.random(ray_indices.size).astype(np.float32)
    else:
        xs += 0.5
        ys += 0.5
    directions[:, 0] = (xs - cx) / fx
    directions[:, 1] = (ys - cy) / fy
    directions[:, 2] = 1.0
    norm = np.expand_dims(np.sqrt((directions * directions).sum(axis=1)), 1)
    directions /= norm
    for i in range(directions.shape[0]):
        directions[i] = R_WC @ directions[i]
    return directions, norm


class LenDataset(torch.utils.data.IterableDataset):

    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __iter__(self):
        iterator = iter(self.dataset)
        for _ in range(self.length):
            yield next(iterator)

    def __len__(self):
        return self.length


class LazyImageLoader:

    def __init__(self, images, size, interpolation=cv2.INTER_CUBIC):
        self.images = images
        self.size = size
        self.inter = interpolation
        self._cache = {}

    def __getitem__(self, i):
        image = self._cache.get(i, None)
        if image is None:
            image = self.images[i]
            image = np.array(Image.open(image), dtype=np.float32) / 255.
            image = cv2.resize(image, self.size, interpolation=self.inter)
            self._cache[i] = image
        return image

    def __len__(self):
        return len(self.images)

    @property
    def shape(self):
        return [len(self)]


class IndexSampler:

    def __init__(self):
        self.classes = np.array([])
        # Index is a dict[int, dict[int, array]]
        # where the first index is the class_id
        # and the second is the image index. The array
        # contains indices which are labeled with that class.
        self.index = {}
        # dict[int, array[float]]
        # where the first int is the class id, the array
        # contains probabilities to sample that class.
        self.image_weights = {}
        self.has_semantics = False
        self.image_range = np.array([])

    def update(self, semantic_maps):
        """
        Recomputes the index.
        0 is the null class, 1 is background and 2 onwards are actual classes.
        """
        assert len(semantic_maps.shape) == 2
        self.index = {}
        self.classes = np.unique(semantic_maps)
        self.classes = self.classes[self.classes != 0]  # remove null class
        class_counts = {}
        zeros = np.zeros(len(semantic_maps))
        for i, semantic in enumerate(semantic_maps):
            for class_id in self.classes:
                where_class = semantic == class_id
                if np.any(where_class):
                    self.has_semantics = True
                    image_indices = self.index.get(class_id, {})
                    image_indices[i] = np.argwhere(where_class.ravel()).ravel()
                    self.index[class_id] = image_indices

                    counts = class_counts.get(class_id, zeros).copy()
                    pixel_count = where_class.sum()
                    counts[i] += where_class.sum()
                    class_counts[class_id] = counts

        for class_id, counts in class_counts.items():
            total = counts.sum()
            assert total != 0
            class_counts[class_id] = counts / total
        self.image_weights = class_counts
        self.image_range = np.arange(len(semantic_maps), dtype=int)

    def sample_class(self):
        return np.random.choice(self.classes)

    def sample(self, class_id, count=1):
        """
        Samples an image and {count} pixel indices belonging to class_id in the sampled image.
        The images are sampled proportionally to how many class_id pixels exist in each image.
        returns: tuple(sampled image index, list(sampled pixel index))
        """
        images = self.index[class_id]
        probabilities = self.image_weights[class_id]
        image_index = np.random.choice(self.image_range, p=probabilities)
        pixel_indices = np.random.choice(images[image_index], count)
        return image_index, pixel_indices

    def semantic_indices(self):
        """
        Returns all image indices that have some semantic markings on them.
        """
        indices = set()
        for class_id, semantic_index in self.index.items():
            for index in semantic_index.keys():
                indices.add(index)
        return sorted(list(indices))

    def sample_sam(self, sam_mask, sam_center, num_pixel, method='proportional'):
        """
        Sample pixel index based on SAM masks.
        num_ray has to be the multiple of 16 (1 anchor ray, 1 positive ray, 14 negative rays).
        Returns:
            pixel_indices - (num_pixel, )
            anchor_indices - (num_pixel // 16, )
            positive_indices - (num_pixel // 16, )
            negative_indices - (num_pixel // 16, 14)
        """
        masks, areas = _get_masks_from_sam_mask(sam_mask)
        mask_weights = np.array(areas) / np.sum(areas)
        mask_choices = np.random.permutation(list(range(SAM_BIT_LEN)))
        num_anchor = num_pixel // 16

        pixel_indices = np.zeros((num_pixel, ), dtype=int)
        anchor_indices = np.arange(num_anchor) * 16
        positive_indices = anchor_indices + 1
        negative_indices = np.arange(num_pixel).reshape(num_anchor, 16)[:, 2:]

        if sam_center is not None:
            centers = np.zeros((num_anchor, sam_center.shape[-1]))
        else:
            centers = None

        a_ind, i = 0, 0
        while a_ind < num_anchor:
            if method == 'proportional':
                mask_choice = np.random.choice(np.arange(SAM_BIT_LEN), p=mask_weights)
            elif method == 'uniform':
                mask_choice = mask_choices[i % SAM_BIT_LEN]
            else:
                raise NotImplementedError("Other sampling methods are not supported.")
            i += 1
            mask = masks[mask_choice]
            if np.sum(mask) == 0:
                continue
            start = a_ind * 16
            pixel_indices[start: start+2] = np.random.choice(
                np.argwhere(mask.ravel()).ravel(), 2
            )
            pixel_indices[start+2: start+16] = np.random.choice(
                np.argwhere(np.logical_not(mask.ravel())).ravel(), 14
            )
            if sam_center is not None:
                centers[a_ind] = sam_center[mask_choice]
            a_ind += 1
        return pixel_indices, anchor_indices, positive_indices, negative_indices, centers



class BaseDataset(torch.utils.data.IterableDataset):
    semantic_image_sample_ratio = 0.5

    def __init__(self, batch_size, camera, sample_chunk_size=512):
        self.split = "train"
        self.camera = camera
        self.batch_size = batch_size
        self.pixel_indices = None
        self.index_sampler = None
        self.features = None
        self.resolution = int(self.camera.size[0] * self.camera.size[1])
        self.w = self.camera.size[0]
        self.h = self.camera.size[1]
        self.intrinsics = np.array([
            self.camera.camera_matrix[0, 0], self.camera.camera_matrix[1, 1],
            self.camera.camera_matrix[0, 2], self.camera.camera_matrix[1, 2]
        ])
        assert sample_chunk_size % 16 == 0
        self.sample_chunk_size = sample_chunk_size
        self.index_sampler = IndexSampler()

    def __iter__(self):
        if self.split == "train":
            while True:
                yield self._next_train()
        else:
            for i in range(self.rotations.shape[0]):
                yield self._get_test(i)

    def _next_train(self):
        chunks = self.batch_size // self.sample_chunk_size
        batch_size = chunks * self.sample_chunk_size
        num_anchors_per_chunk = self.sample_chunk_size // 16
        pixels = np.zeros((batch_size, 3), dtype=np.float32)
        depths = np.zeros(batch_size, dtype=np.float32)
        semantics = np.zeros(batch_size, dtype=int)
        sam_masks = np.zeros(batch_size, dtype=int)
        ray_o = np.zeros((batch_size, 3), dtype=np.float32)
        ray_d = np.zeros((batch_size, 3), dtype=np.float32)
        direction_norms = np.zeros((batch_size, 1), dtype=np.float32)
        anchor_indices = np.zeros((batch_size // 16, ), dtype=int)
        positive_indices = np.zeros((batch_size // 16, ), dtype=int)
        negative_indices = np.zeros((batch_size // 16, 14), dtype=int)
        if self.sam_centers is None:
            sam_centers = None
        else:
            sam_centers = np.zeros((batch_size // 16, self.sam_center_feature_dim))

        out = {
            'rays_o': ray_o,
            'rays_d': ray_d,
            'pixels': pixels,
            'direction_norms': direction_norms,
            'depth': depths,
            'semantic': semantics,
            'sam': sam_masks,
            'chunk_size': self.sample_chunk_size,
            'direction_norms': direction_norms,
            'anchor_indices': anchor_indices,
            'positive_indices': positive_indices,
            'negative_indices': negative_indices,
            'sam_centers': sam_centers
        }
        if self.features is not None:
            features = np.zeros((batch_size, self.feature_dim),
                                dtype=np.float32)
            out['features'] = features

        for chunk in range(chunks):
            start = chunk * self.sample_chunk_size
            end = (chunk + 1) * self.sample_chunk_size

            if self.sam_sampling is not None:
                image_index = np.random.randint(0, self.n_examples)
                if self.sam_centers is None:
                    sam_center = None
                else:
                    sam_center = self.sam_centers[image_index]
                ray_indices, a_inds, p_inds, n_inds, centers = self.index_sampler.sample_sam(
                    self.sam_masks[image_index], sam_center, self.sample_chunk_size, method=self.sam_sampling)
                anchor_start = chunk * num_anchors_per_chunk
                anchor_end = (chunk + 1) * num_anchors_per_chunk
                anchor_indices[anchor_start: anchor_end] = a_inds + start
                positive_indices[anchor_start: anchor_end] = p_inds + start
                negative_indices[anchor_start: anchor_end, :] = n_inds + start
                if sam_centers is not None:
                    sam_centers[anchor_start: anchor_end, :] = centers
                
            elif self.index_sampler.has_semantics and random.random(
            ) < self.semantic_image_sample_ratio:
                class_id = self.index_sampler.sample_class()
                image_index, ray_indices = self.index_sampler.sample(
                    class_id, self.sample_chunk_size)
            else:
                image_index = np.random.randint(0, self.n_examples)
                ray_indices = np.random.choice(self.pixel_indices,
                                               size=(self.sample_chunk_size,))

            pixels[start:end] = self.images[image_index][ray_indices]
            depths[start:end] = self.depths[image_index][ray_indices] / 1000.0
            semantics[start:end] = self.semantics[image_index][
                ray_indices].astype(int) - 1
            sam_masks[start:end] = self.sam_masks[image_index][ray_indices]
            ray_o[start:end] = np.broadcast_to(self.origins[image_index][None],
                                               (ray_indices.shape[0], 3))
            dirs, norm = self._compute_direction(image_index,
                                                 ray_indices,
                                                 randomize=True)
            ray_d[start:end] = dirs
            direction_norms[start:end] = norm

            if self.features is not None:
                width = int(self.w)
                x = ray_indices % width
                y = (ray_indices - x) / width
                xy = np.stack([x, y], axis=-1)
                xy_features = self._scale_to_feature_xy(xy)

                index = xy_features[:, 1] * self.feature_width + xy_features[:,
                                                                             0]
                features[start:end] = self.features[image_index][index, :]

        return out

    def _get_test(self, image_index):
        image = self.images[image_index].reshape(self.h, self.w, 3)
        ray_o = np.broadcast_to(self.origins[image_index],
                                (self.h, self.w, 3)).astype(np.float32)
        ray_d, norms = self._compute_direction(image_index,
                                               np.arange(self.resolution))
        ray_d = ray_d.reshape(self.h, self.w, 3).astype(np.float32)
        depth = (self.depths[image_index] / 1000.0).reshape(self.h, self.w)
        semantic = (self.semantics[image_index].astype(int) - 1).reshape(
            self.h, self.w)
        out = {
            'pixels': image,
            'rays_o': ray_o,
            'rays_d': ray_d,
            'depth': depth,
            'semantic': semantic,
            'H': self.h,
            'W': self.w,
            'direction_norms': norms,
        }
        if self.features is not None:
            out['features'] = self.features[image_index]
        return out

    def _convert_pose(self, T_CW, scene_center):
        """
        Returns the transformation that transforms from world to camera
        coordinates, with all the ngp hacks applied.
        """
        T_WC = np.linalg.inv(T_CW)
        T_WC[:3, 3] = T_WC[:3, 3] - scene_center
        T_WC = T_WC @ CV_TO_OPENGL
        return nerf_matrix_to_ngp(T_WC, scale=1.0)

    def _compute_rays(self):
        if self.split == "train":
            self.images = self.images.reshape(self.n_examples, self.resolution,
                                              3)
            self.depths = self.depths.reshape(self.n_examples, self.resolution)
            self.semantics = self.semantics.reshape(self.n_examples,
                                                    self.resolution)
            self.sam_masks = self.sam_masks.reshape(self.n_examples,
                                                    self.resolution)
            # self.sam_centers = self.sam_centers

    def _compute_direction(self, image_index, ray_indices, randomize=False):
        """
        image_index: int, index of image/pose in question
        ray_indices: np.array[int] N list of pixel indices for which to compute ray direction
        returns: np.array N x 3 floats
        """
        R_WC = self.rotations[image_index]
        return _compute_direction(R_WC, ray_indices, self.w, self.camera.fx,
                                  self.camera.fy, self.camera.cx,
                                  self.camera.cy, randomize)

    def _compute_image_mask(self, images):
        """
        From a few rgb images, determine which pixels should be sampled.
        If pixels are black in all frames, assume they are due to undistortion.
        """
        if isinstance(images, LazyImageLoader):
            indices = np.random.randint(0, len(images), size=5)
            images = np.stack([images[index] for index in indices])
        else:
            images = images[::10]
        # Remove any pixels that also very dark across all images.
        # There is likely something weird with the pipeline through which these came from.
        where_non_zero = np.any(images > (10. / 255.), axis=3)
        where_non_zero = np.any(where_non_zero.reshape(where_non_zero.shape[0],
                                                       -1),
                                axis=0)
        self.pixel_indices = np.argwhere(where_non_zero).ravel()


class SceneDataset(BaseDataset):

    def __init__(self,
                 split,
                 scene,
                 factor=4.0,
                 size=None,
                 batch_size=4096,
                 sample_chunk_size=512,
                 lazy=False,
                 features=None,
                 load_semantic=True,
                 load_sam_mask=True,
                 sam_sampling='proportional'):
        self.lazy = lazy
        self.scene = Scene(scene)
        self.image_names = self.scene.image_names()
        self.pixel_indices = None
        self.index_sampler = None
        self.features = None
        self.load_semantic = load_semantic
        self.load_sam_mask = load_sam_mask
        self.sam_sampling = sam_sampling
        camera = self.scene.camera
        if size is not None:
            small_size = size
        else:
            size = camera.size
            small_size = (int(size[0] / factor), int(size[1] / factor))
        image_count = min(
            [len(self.scene.rgb_paths()),
             len(self.scene.depth_paths())])
        self.indices = np.arange(0, image_count)
        camera = self.scene.camera.scale(small_size)
        super().__init__(batch_size, camera, sample_chunk_size)
        self.split = split
        self.sample_mask_size = 128
        self.sam_centers = None
        self._load_images()
        self._compute_rays()
        if features is not None:
            self._load_features(features)
        self.error_map = None
        self.n_classes = self.scene.n_classes

    def _load_images(self):
        images = []
        depths = []
        semantics = []
        sam_masks = []
        cameras = []

        color_images = self.scene.rgb_paths()
        depth_images = self.scene.depth_paths()

        aabb = self.scene.bbox()
        scene_center = (aabb[0] + aabb[1]) / 2
        aabb = aabb - scene_center

        poses = self.scene.poses

        for index in self.indices:
            frame = color_images[index]
            if self.lazy:
                images.append(frame)
            else:
                image = np.array(Image.open(frame), dtype=np.float32)[..., :3]
                image = cv2.resize(image,
                                   self.camera.size,
                                   interpolation=cv2.INTER_NEAREST)
                images.append(image / 255.)

            semantic_path = os.path.join(self.scene.path, 'semantic',
                                         os.path.basename(depth_images[index]))
            if self.load_semantic and os.path.exists(semantic_path):
                image = Image.open(semantic_path)
                image = image.resize(self.camera.size, Image.NEAREST)
                semantics.append(np.asarray(image))
            else:
                semantics.append(
                    np.zeros(self.camera.size[::-1], dtype=np.uint8))
                
            sam_mask_path = os.path.join(self.scene.path, 'sam_mask',
                                         os.path.splitext(os.path.basename(depth_images[index]))[0] + '.exr')
            if self.load_sam_mask and os.path.exists(sam_mask_path):
                image = _load_sam_mask(sam_mask_path, self.camera.size)
                sam_masks.append(image)
            else:
                sam_masks.append(
                    np.zeros(self.camera.size[::-1], dtype=np.uint32))

            T_CW = poses[index]
            T_WC = self._convert_pose(T_CW, scene_center)
            cameras.append(T_WC.astype(np.float32))

            if self.lazy:
                depths.append(depth_images[index])
            else:
                depth_image = cv2.imread(depth_images[index], -1)
                depth = cv2.resize(depth_image, self.camera.size,
                                   cv2.INTER_NEAREST)
                depths.append(depth)

        if self.lazy:
            self.images = LazyImageLoader(images,
                                          self.camera.size,
                                          interpolation=cv2.INTER_NEAREST)
            self.depths = LazyImageLoader(depths,
                                          self.camera.size,
                                          interpolation=cv2.INTER_NEAREST)
        else:
            self.images = np.stack(images, axis=0)
            self.depths = np.stack(depths, axis=0)

        self.semantics = np.stack(semantics)
        self.index_sampler.update(self.semantics.reshape(-1, self.resolution))
        self.sam_masks = np.stack(sam_masks)
        self._compute_image_mask(self.images)
        self.poses = np.stack(cameras, axis=0)
        self.rotations = np.ascontiguousarray(self.poses[:, :3, :3])
        self.origins = self.poses[:, :3, 3]
        self.n_examples = self.images.shape[0]

        self.min_bounds = aabb[0]
        self.max_bounds = aabb[1]

    def _next_update(self, image_index):
        sam_mask = self.sam_masks[image_index]
        masks, _ = _get_masks_from_sam_mask(sam_mask)
        ray_indices = np.zeros(SAM_BIT_LEN * self.sample_mask_size)

        for i in range(SAM_BIT_LEN):
            mask = masks[i]
            if np.sum(mask) == 0:
                ray_indices[i*self.sample_mask_size: (i+1)*self.sample_mask_size] = np.random.choice(
                    self.resolution, self.sample_mask_size
                )
            else:
                ray_indices[i*self.sample_mask_size: (i+1)*self.sample_mask_size] = np.random.choice(
                    np.argwhere(mask.ravel()).ravel(), self.sample_mask_size
                )

        ray_o = np.broadcast_to(self.origins[image_index][None],
                                (ray_indices.shape[0], 3))
        ray_d, norms = self._compute_direction(image_index,
                                               ray_indices,
                                               randomize=True)
        out = {
            'rays_o': ray_o,
            'rays_d': ray_d,
            'direction_norms': norms,
            'num_masks': SAM_BIT_LEN,
            'sample_mask_size': self.sample_mask_size
        }
        return out
    
    def update_sam_centers(self, image_index, sam_center):
        if self.sam_centers is None:
            self.sam_centers = {}
            self.sam_center_feature_dim = sam_center.shape[-1]
        self.sam_centers[image_index] = sam_center

    def semantic_map_updated(self, image_index):
        filename = f"{self.image_names[image_index]}.png"
        semantic_path = os.path.join(self.scene.path, 'semantic', f"{filename}")
        if os.path.exists(semantic_path):
            image = Image.open(semantic_path)
            image = np.asarray(image.resize(self.camera.size, Image.NEAREST))
            self.semantics[image_index, :] = image.reshape(self.resolution)
            self.index_sampler.update(self.semantics)
        else:
            print(f"Could not find image {semantic_path}")

    def update_sampler(self):
        """
        Called if the semantic annotations have been updated,
        e.g. during a user simulation.
        """
        self.index_sampler.update(self.semantics)

    def _load_features(self, features):
        with h5py.File(os.path.join(self.scene.path, 'features.hdf'),
                       'r') as hdf:
            features = hdf[f'features/{features}'][:]
            N, H, W, C = features.shape
            self.features = features.reshape(N, H * W, C)
            self.feature_width = W
            self.feature_height = H
            self.feature_dim = C
        scale_factor = np.array(
            [W / self.camera.size[0], H / self.camera.size[1]])
        self._scale_to_feature_xy = lambda xy: (xy * scale_factor).astype(int)


import threading
import time
from collections import deque


class DynamicDataset(BaseDataset):

    def __init__(self, batch_size, camera, capacity=None):
        super().__init__(batch_size, camera)
        self.capacity = capacity
        self.poses = []
        self.rotations = []
        self.origins = []
        self.images = []
        self.depths = []
        self.features = []
        self.semantics = []
        self.n_examples = 0
        self.prefetch_buffer = deque()
        self.prefetch_buffer_size = 25
        self.stopped = False
        self._prefetch_thread = threading.Thread(target=self._prefetch)
        self._prefetch_thread.start()

    def stop(self):
        self.stopped = True
        self._prefetch_thread.join()

    def _prefetch(self):
        while not self.stopped:
            if len(self.features) == 0 or len(
                    self.prefetch_buffer) >= self.prefetch_buffer_size:
                time.sleep(0.1)
                continue
            self.prefetch_buffer.append(self._next_train())

    def __iter__(self):
        while True:
            if len(self.prefetch_buffer) == 0:
                time.sleep(0.1)
            else:
                yield self.prefetch_buffer.popleft()

    def add_frame(self, T_CW, rgb, depth, features):
        if len(self.features) == 0:
            self._init_features(features)

        assert depth.dtype == np.uint16
        assert rgb.dtype == np.uint8
        assert len(features.shape) == 3
        assert features.shape[0] == self.feature_height

        if self.pixel_indices is None:
            self.resolution = rgb.shape[0] * rgb.shape[1]
            self.pixel_indices = np.arange(self.resolution)

        T_WC = self._convert_pose(T_CW)
        self.poses.append(T_WC)
        self.rotations.append(np.ascontiguousarray(T_WC[:3, :3]))
        self.origins.append(T_WC[:3, 3])
        self.images.append(rgb.reshape(-1, 3) / 255.)
        self.depths.append(depth.reshape(-1))
        self.features.append(
            features.reshape(self.feature_height * self.feature_width,
                             features.shape[2]))
        self.semantics.append(np.zeros(self.resolution, dtype=np.uint16))
        self.n_examples = len(self.images)

        if self.capacity is not None and len(self.poses) > self.capacity:
            remove_index = np.random.randint(0, len(self.poses))
            del self.poses[remove_index]
            del self.rotations[remove_index]
            del self.origins[remove_index]
            del self.images[remove_index]
            del self.depths[remove_index]
            del self.features[remove_index]
            del self.semantics[remove_index]
            self.n_examples = len(self.images)

    def __len__(self):
        return self.n_examples

    def _init_features(self, features):
        H, W, D = features.shape
        self.feature_height = H
        self.feature_width = W
        self.feature_dim = D
        scale_factor = np.array([
            self.feature_width / self.camera.size[0],
            self.feature_height / self.camera.size[1]
        ])
        self._scale_to_feature_xy = lambda xy: (xy * scale_factor).astype(int)
