import numpy as np
import torch
import glob
import argparse
import pickle
import os
from autolabel.models import ALNetwork


def load_checkpoint(model, checkpoint_dir, device='cuda:0'):
    checkpoint_list = sorted(glob.glob(f'{checkpoint_dir}/*.pth'))
    best = [c for c in checkpoint_list if 'best.pth' in c]
    if len(best) != 0:
        checkpoint = best[0]
    else:
        checkpoint = checkpoint_list[-1]
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_dict['model'])

    instance_centers_list = sorted(glob.glob(f'{checkpoint_dir}/*.npy'))
    if len(instance_centers_list) > 0:
        model.set_instance_centers(
            np.load(instance_centers_list[-1])
        )
    instance_cluster_list = sorted(glob.glob(f'{checkpoint_dir}/*.pkl'))
    if len(instance_cluster_list) > 0:
        with open(instance_cluster_list[-1], 'rb') as inp:
            clust = pickle.load(inp)
        model.set_instance_clusterer(clust)
    return model


def model_flag_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--geometric-features', '-g', type=int, default=15)
    parser.add_argument('--encoding',
                        default='hg+freq',
                        choices=['freq', 'hg', 'hg+freq'],
                        type=str,
                        help="Network positional encoding to use.")
    parser.add_argument('--features',
                        type=str,
                        default=None,
                        choices=[None, 'fcn50', 'dino', 'lseg'],
                        help="Use semantic feature supervision.")
    parser.add_argument('--rgb-weight', default=1.0, type=float)
    parser.add_argument('--semantic-weight', default=1.0, type=float)
    parser.add_argument('--feature-weight', default=0.5, type=float)
    parser.add_argument('--depth-weight', default=0.1, type=float)
    parser.add_argument('--feature-dim', default=64, type=int)
    parser.add_argument('--contrastive-weight', default=0.1, type=float)
    parser.add_argument('--contrastive-feat-dim', default=8, type=int)
    parser.add_argument('--contrastive-temperature', default=0.1, type=float)
    return parser


def model_hash(flags):
    features = 'plain'
    if flags.features is not None:
        features = flags.features
    string = f"g{flags.geometric_features}_{flags.encoding}_{features}"
    string += f"_rgb{flags.rgb_weight}_d{flags.depth_weight}_s{flags.semantic_weight}"
    string += f"_f{flags.feature_weight}"
    string += f"_c{flags.contrastive_weight}"
    return string


def model_dir(scene_path, flags):
    mhash = model_hash(flags)
    if flags.workspace is None:
        return os.path.join(scene_path, 'nerf', mhash)
    scene_name = os.path.basename(os.path.normpath(flags.scene))
    return os.path.join(flags.workspace, scene_name, mhash)


def create_model(min_bounds, max_bounds, n_classes, flags, bound_scale=1.25):
    bound = np.max([np.abs(min_bounds), np.abs(max_bounds)], axis=0).max() * bound_scale
    return ALNetwork(num_layers=2,
                     num_layers_color=2,
                     hidden_dim_color=128,
                     hidden_dim=128,
                     geo_feat_dim=flags.geometric_features,
                     encoding=flags.encoding,
                     bound=float(bound),
                     hidden_dim_semantic=flags.feature_dim,
                     contrastive_feat_dim=flags.contrastive_feat_dim,
                     cuda_ray=False,
                     density_scale=1,
                     semantic_classes=n_classes)


def read_params(workspace):
    with open(os.path.join(workspace, 'params.pkl'), 'rb') as f:
        return pickle.load(f)


def write_params(workspace, flags):
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(workspace, 'params.pkl'), 'wb') as f:
        pickle.dump(flags, f)


def get_nerf_dir(scene, flags):
    scene_name = os.path.basename(os.path.normpath(scene))
    if flags.workspace is None:
        return os.path.join(scene, 'nerf')
    else:
        return os.path.join(flags.workspace, scene_name)
    