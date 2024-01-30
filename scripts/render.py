import h5py
import numpy as np
import pandas
import cv2
import os
import pickle
from tqdm import tqdm
import torch

from autolabel.dataset import SceneDataset
from autolabel import model_utils
from autolabel import visualization
from autolabel.utils.feature_utils import get_feature_extractor
from pathlib import Path
from sklearn import decomposition
from sklearn.metrics.pairwise import cosine_similarity


def read_args():
    parser = model_utils.model_flag_parser()
    parser.add_argument('scene')
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument(
        '--max-depth',
        type=float,
        default=7.5,
        help="The maximum depth used in colormapping the depth frames.")
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help="Where to save the video.")
    parser.add_argument('--classes',
                        default=None,
                        type=str,
                        nargs='+',
                        help="Which classes to segment the scene into.")
    parser.add_argument('--label-map',
                        default=None,
                        type=str,
                        help="Path to list of labels.")
    return parser.parse_args()


class FeatureTransformer:

    def __init__(self, scene_path, feature_name, classes, checkpoint=None, without_features=False):
        if not without_features:
            with h5py.File(os.path.join(scene_path, 'features.hdf'), 'r') as f:
                features = f[f'features/{feature_name}']
                blob = features.attrs['pca'].tobytes()
                self.pca = pickle.loads(blob)
                self.feature_min = features.attrs['min']
                self.feature_range = features.attrs['range']
            self.first_fit = False
        else:
            self.pca = decomposition.PCA(n_components=3)
            self.feature_min = None
            self.feature_range = None
            self.first_fit = True


        if feature_name is not None:
            extractor = get_feature_extractor(feature_name, checkpoint)
            self.text_features = self._encode_text(extractor, classes)

    def _encode_text(self, extractor, text):
        return extractor.encode_text(text)

    def __call__(self, p_features):
        H, W, C = p_features.shape
        if self.first_fit:
            features = self.pca.fit_transform(p_features.reshape(H * W, C))
            self.first_fit = False
        else:
            features = self.pca.transform(p_features.reshape(H * W, C))

        if (self.feature_min is not None) and (self.feature_range is not None):
            features = np.clip((features - self.feature_min) / self.feature_range,
                            0., 1.)
        else:
            features = np.clip((features - np.min(features)) / (np.max(features) - np.min(features)),
                            0., 1.)
        return (features.reshape(H, W, 3) * 255.).astype(np.uint8)


def compute_semantics(outputs, classes, feature_transform):
    if classes is not None:
        features = outputs['semantic_features']
        features = (features / torch.norm(features, dim=-1, keepdim=True))
        text_features = feature_transform.text_features
        H, W, D = features.shape
        C = text_features.shape[0]
        similarities = torch.zeros((H, W, C), dtype=features.dtype)
        for i in range(H):
            similarities[i, :, :] = (features[i, :, None] *
                                     text_features).sum(dim=-1).cpu()
        return similarities.argmax(dim=-1)
    else:
        return outputs['semantic'].argmax(dim=-1).cpu().numpy()

def compute_instances(outputs, feature_centers):
    instance_feature = outputs['contrastive_features'].cpu().numpy()
    image_height, image_width, feature_dim = instance_feature.shape
    instance_feature = instance_feature.reshape(-1, feature_dim)
    sim_mat = cosine_similarity(instance_feature, feature_centers)
    pred_instance = np.argmax(sim_mat, axis=1)
    pred_instance = pred_instance.reshape(image_height, image_width)
    return pred_instance

def render(model,
           batch,
           feature_transform,
           semantic_color_map, 
           instance_color_map,
           size=(480, 360),
           maxdepth=10.0,
           classes=None,
           con_feature_transform=None):
    rays_o = torch.tensor(batch['rays_o']).cuda()
    rays_d = torch.tensor(batch['rays_d']).cuda()
    direction_norms = torch.tensor(batch['direction_norms']).cuda()
    outputs = model.render(rays_o,
                           rays_d,
                           direction_norms,
                           staged=True,
                           perturb=False,
                           num_steps=512,
                           upsample_steps=0)
    p_semantic = compute_semantics(outputs, classes, feature_transform)
    p_instance = compute_instances(outputs, model.instance_centers)
    frame = np.zeros((2 * size[1], 3 * size[0], 3), dtype=np.uint8)
    h_mid = size[1]
    w_ot, w_tt = size[0], size[0] * 2
    p_rgb = (outputs['image'].cpu().numpy() * 255.0).astype(np.uint8)
    p_depth = outputs['depth']
    frame[:h_mid, :w_ot, :] = p_rgb
    frame[h_mid:, :w_ot] = visualization.visualize_depth(
        p_depth.cpu().numpy(), maxdepth=maxdepth)[:, :, :3]
    frame[:h_mid, w_tt:] = semantic_color_map[p_semantic]
    frame[h_mid:, w_tt:] = instance_color_map[p_instance]
    
    if feature_transform is not None:
        p_features = feature_transform(
            outputs['semantic_features'].cpu().numpy())
        frame[:h_mid, w_ot:w_tt] = p_features

    if con_feature_transform is not None:
        p_con_features = con_feature_transform(
            outputs['contrastive_features'].cpu().numpy())
        frame[h_mid:, w_ot:w_tt] = p_con_features

    return frame


def main():
    flags = read_args()
    model_params = model_utils.read_params(flags.model_dir)

    view_size = (480, 360)
    dataset = SceneDataset('test',
                           flags.scene,
                           size=view_size,
                           batch_size=16384,
                           features=model_params.features,
                           load_semantic=False,
                           lazy=True)

    classes = flags.classes
    if flags.label_map is not None:
        label_map = pandas.read_csv(flags.label_map)
        classes = label_map['prompt'].values
    semantic_color_map = (np.random.rand(len(classes), 3) * 255).astype(np.uint8)

    feature_transform = None
    if model_params.features is not None:
        feature_transform = FeatureTransformer(flags.scene,
                                               model_params.features, classes,
                                               flags.checkpoint)

    con_feature_transform = FeatureTransformer(flags.scene,
                                               None, classes,
                                               without_features=True)

    n_classes = dataset.n_classes if dataset.n_classes is not None else 2
    model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                     n_classes, model_params).cuda()
    model = model.eval()
    model_utils.load_checkpoint(model,
                                os.path.join(flags.model_dir, 'checkpoints'))
    
    instance_color_map = (np.random.rand(model.instance_centers.shape[0], 3) * 255).astype(np.uint8)

    Path(flags.out).mkdir(exist_ok=True, parents=True)
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True):
            for frame_index in tqdm(dataset.indices[::flags.stride]):
                batch = dataset._get_test(frame_index)
                frame = render(model,
                               batch,
                               feature_transform,
                               semantic_color_map=semantic_color_map, 
                               instance_color_map=instance_color_map,
                               size=view_size,
                               maxdepth=flags.max_depth,
                               classes=classes,
                               con_feature_transform=con_feature_transform)
                cv2.imwrite(
                    os.path.join(flags.out, f"{frame_index}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )


if __name__ == "__main__":
    main()
