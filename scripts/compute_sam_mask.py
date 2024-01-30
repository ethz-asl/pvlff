import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--sam-vit-h-checkpoint', type=str, required=True)
    parser.add_argument('--prefer-union-mask', action='store_true')
    return parser.parse_args()

def _iou(image_vector1, image_vector2):
    intersection = np.logical_and(image_vector1, image_vector2).sum()
    union = np.logical_or(image_vector1, image_vector2).sum()
    iou = intersection / union
    return iou, intersection, union

def generate_float32_mask(masks, prefer_union_mask=True):
    indices = []
    for i in np.random.permutation(list(range(len(masks)))):
        if len(indices) >= 32:
            break

        overlapped = False
        for j, ind in enumerate(indices):
            iou, intersection, union = _iou(masks[i]['segmentation'].reshape(-1), masks[ind]['segmentation'].reshape(-1))
            if iou > 0.8:
                overlapped = True
                break
            if prefer_union_mask:
                if intersection / masks[ind]['area'] > 0.8:
                    indices[j] = i
                    overlapped = True
                    break
                elif intersection / masks[i]['area'] > 0.8:
                    overlapped = True
                    break
        if not overlapped:
            indices.append(i)
    
    one = 1
    uint32_mask = np.zeros_like(masks[0]['segmentation'], dtype=np.uint32)
    for i, ind in enumerate(indices):
        mask = masks[ind]
        number = (one << i)
        uint32_mask += (number * mask['segmentation']).astype(np.uint32)
    return uint32_mask.view(np.float32)

def main():
    flags = read_args()
    sam_checkpoint = flags.sam_vit_h_checkpoint
    model_type = "vit_h"
    device = "cuda"

    scene_folder = Path(flags.scene)

    image_folder = scene_folder / "rgb"
    output_folder = scene_folder / "sam_mask"
    output_folder.mkdir(parents=True, exist_ok=True)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_files = os.listdir(image_folder)
    image_files.sort()

    for image_file in tqdm(image_files):
        image = cv2.imread(
            os.path.join(image_folder, image_file)
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        masks.sort(key=lambda x: x['area'], reverse=True)
        masks = [mask for mask in masks if mask['area'] > 100]
        
        sam_mask = generate_float32_mask(masks, prefer_union_mask=flags.prefer_union_mask)
        cv2.imwrite(
            os.path.join(output_folder, os.path.splitext(image_file)[0] + '.exr'),
            sam_mask
        )

if __name__ == "__main__":
    main()
