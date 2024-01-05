<h1 align="center">Panoptic Vision-Language Feature Fields</h1>

<p align="center">
<strong><a href="https://haoranchen1104.github.io/">Haoran Chen</a></strong>,
<strong><a href="https://keke.dev/">Kenneth Blomqvist</a></strong>,
<strong><a href="https://scholar.google.com/citations?user=qwSANZoAAAAJ&hl=en&oi=ao">Francesco Milano</a></strong>, <strong><a href="https://asl.ethz.ch/">Roland Siegwart</a></strong>
</p>

<h2 align="center">IEEE RA-L 2024</h2>
<h3 align="center"><a href="https://arxiv.org/abs/2309.05448">Paper</a> | <a href="TODO">Video</a> | <a href="https://ethz-asl.github.io/pvlff/">Project Page</a></h3>

<p align="center">
  <a href="">
    <img src="./assets/teaser.png" alt="Panoptic Vision-Language Feature Fields" width="90%">
  </a>
</p>

Recently, methods have been proposed for 3D _open-vocabulary_ semantic segmentation. Such methods are able to segment scenes into arbitrary classes based on text descriptions provided during runtime. In this paper, we propose to the best of our knowledge the first algorithm for _open-vocabulary panoptic_ segmentation in 3D scenes. Our algorithm, Panoptic VisionLanguage Feature Fields (PVLFF), learns a semantic feature field of the scene by distilling vision-language features from a pretrained 2D model, and jointly fits an instance feature field through contrastive learning using 2D instance segments on input frames. Despite not being trained on the target classes, our method achieves panoptic segmentation performance similar to the state-of-the-art _closed-set_ 3D systems on the HyperSim, ScanNet and Replica dataset and additionally outperforms current 3D open-vocabulary systems in terms of semantic segmentation. We ablate the components of our method to demonstrate the effectiveness of our model architecture. 

## Table of Contents

1. [Installation](#installation)
2. [Running experiments](#running-experiments)
3. [Citation](#citation)
4. [Acknowledgements](#acknowledgements)

## Installation

TODO


## Running experiments

TODO

## Citation

If you find our code or paper useful, please cite:

```bibtex
@journal{Chen2024PVLFF,
  author    = {Chen, Haoran and Blomqvist, Kenneth and Milano, Francesco and Siegwart, Roland},
  title     = {Panoptic Vision-Language Feature Fields},
  journal   = {IEEE Robotics and Automation Letters (RA-L)},
  year      = {2024}
}
```

## Acknowledgements

TODO