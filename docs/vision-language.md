# Prerequisites

## Installing LSeg

In addition to the regular installation instructions, you also need to install LSeg. This can be done by running the following commands with your python environment loaded.
```
git clone https://github.com/kekeblom/lang-seg
cd lang-seg
pip install -e .
```

## Data conversion

Follow the instructions in `docs/data.md` to convert the scenes from original datasets into the our format.

---
---

# [Neural Implicit Vision-language Feature Fields](https://arxiv.org/abs/2303.10962)

checkout to branch `lseg`

## Running ScanNet experiment

 Use the following commands to compute vision-language features, fit the scene representation and evaluate against the ground truth:
```
# Train . Has to be run separately for each scene.
python scripts/compute_feature_maps.py <dataset-dir>/<scene> --features lseg --checkpoint <lseg-weights>
python scripts/train.py --features lseg --feature-dim 512 --iters 25000 <dataset-dir>/<scene>

# Once trained on all scenes, evaluate.
# 3D queries evaluated against the 3D pointcloud
python scripts/language/evaluate.py --pc --label-map <label-map> --feature-checkpoint <lseg-weights> <dataset-dir>
# 2D queries against the ground truth semantic segmentation maps
python scripts/language/evaluate.py --label-map <label-map> --feature-checkpoint <lseg-weights> <dataset-dir>
```

`dataset-dir` is the path to the scannet converted scenes, `scene` is the name of the scene. `lseg-weights` is the path to the lseg checkpoint.

## Running the real-time ROS node

The `scripts/ros/` directory contains ROS nodes which can be used to integrate with a real-time SLAM system. These have been tested under ROS Noetic.

`scripts/ros/node.py` is the node which listens to keyframes and integrates the volumetric representation as they come in. It listens to the following topics:
- `/slam/rgb` image messages.
- `/slam/depth` depth frames encoded as uint16 values in millimeters.
- `/slam/keyframe` PoseStamped messages which correspond to camera poses for the rgb and depth messages.
- `/slam/camera_info` CameraInfo message containing the intrinsic parameters.
- `/slam/odometry` (optional) PoseStamped messages. Each time a message comes in, it renders an rgb frame and semantic segmentation map which is published at `/autolabel/image` and `/autolabel/features` respectively.
- `/autolabel/segmentation_classes` segmentation class prompts as a String message published by the `class_input.py` node.

It can be run with `python scripts/ros/node.py --checkpoint <lseg-weights> -b <bound>`. The bound parameter is optional and defaults to 2.5 meters. It defines the size of the volume, extending `bound` meters from `[-bound, -bound, -bound]` to `[bound, bound, bound]` in the x, y and z directions.

For an implementation of the SLAM node, you can use the ROS node from the [SpectacularAI SDK examples](https://github.com/SpectacularAI/sdk-examples/blob/main/python/oak/mapping_ros.py), in case you have an OAK-D stereo camera.

`scripts/ros/class_input.py` presents a graphical user interface which can be used to define the segmentation classes used by the ROS node. It published class at `/autolabel/segmentation_classes`.

---
---

# [Panoptic Vision-Language Feature Fields](https://arxiv.org/abs/2309.05448)

checkout to branch `panoptic`

## Training
To begin the training process, first run the precomputing steps:

```
# compute the vision-language features
python scripts/compute_feature_maps.py <dataset-dir>/<scene> \
    --features lseg \
    --checkpoint <lseg-weights> \
    --dim 512

# compute the instance masks using SAM
python scripts/compute_sam_mask.py <dataset-dir>/<scene> \
    --sam-vit-h-checkpoint <sam-weights>
```

`dataset-dir` is the path to the scannet converted scenes, `scene` is the name of the scene. `lseg-weights` is the path to the lseg checkpoint. `sam-weights` is the path to the SAM checkpoint (which can be downloaded [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)).

Then, fit the scene representation using the same training script with additional flags on:
```
python scripts/train.py <dataset-dir>/<scene> \
    --batch-size 2048 \
    --iters 20000 \
    --workspace <workspace> \
    --feature-dim 512 \
    --features lseg \
    --contrastive \
    --sam-sampling <sampling-method> \
    --slow-center \
    --cluster-instance-features
```


`workspace` is the folder where the model is saved. `--contrastive` is the option to train instance feature field using contrastive learning. `--sam-sampling` denotes the strategy to sample the SAM masks for training. The strategies include `proportional`, `uniform` and `None`, where `proportional` means sampling the masks according to their areas, `uniform` means sampling these masks uniformly, and `None` means not using sampling strategy and training with multiple positive pairs. `--slow-center` denotes whether to use "slow center strategy". `--cluster-instance-features` denotes to run the clustering after the training and save the cluster centers together with the clusterer itself.

## Evaluation

Scene-level Panoptic Quality and 2D Semantic Segmentation
```
python scripts/language/evaluate.py <dataset-dir> \
    --vis <evaluation-folder/vis> \ # the folder to save the visualization results.
    --workspace <workspace> \
    --out <evaluation-folder> \ # the folder to save the evaluation results.
    --label-map <label-map> \
    --feature-checkpoint <lseg-weights> \
    --panoptic # the flag to evaluate scene-level PQ and 2D semantic segmentation.
#    --debug # whether to save the visualization images.    
```

3D Semantic Segmentation (only for ScanNet)
```
python scripts/language/evaluate.py <dataset-dir> \
    --vis <evaluation-folder/vis> \ # the folder to save the visualization results.
    --workspace <workspace> \
    --out <evaluation-folder> \ # the folder to save the evaluation results.
    --label-map <label-map> \
    --feature-checkpoint <lseg-weights> \
    --pc # the flag to 3D semantic segmentation.
```

