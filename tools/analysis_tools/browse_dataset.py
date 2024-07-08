# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.models.utils import mask2ndarray
from mmdet.registry import DATASETS, VISUALIZERS
from mmdet.structures.bbox import BaseBoxes

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


import cv2  # Add this import
import numpy as np

def denormalize(img, mean, std):
    """Denormalize image."""
    img = img * std + mean
    return img

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    # Get normalization parameters from the config
    img_norm_cfg = cfg.train_pipeline[-2]  # Assuming Normalize is the second last step
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)

    progress_bar = ProgressBar(len(dataset))
    for item in dataset:
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_samples'].numpy()
        gt_instances = data_sample.gt_instances
        img_path = osp.basename(item['data_samples'].img_path)

        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None

        # Denormalize the image for visualization
        img_vis = denormalize(img, mean, std)

        # Check if the image is grayscale
        if img_vis.shape[-1] == 1:
            img_vis = img_vis.squeeze(-1)  # Remove the last dimension if it's 1 (grayscale)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for consistent visualization
        else:
            img_vis = img_vis[..., [2, 1, 0]]  # bgr to rgb for color images

        # Clip the image values to [0, 255] and convert to uint8
        img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)

        # Debug print to check image values
        print(f"Image min: {np.min(img_vis)}, max: {np.max(img_vis)}, mean: {np.mean(img_vis)}")

        gt_bboxes = gt_instances.get('bboxes', None)
        if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor
        gt_masks = gt_instances.get('masks', None)
        if gt_masks is not None:
            masks = mask2ndarray(gt_masks)
            gt_instances.masks = masks.astype(bool)
        data_sample.gt_instances = gt_instances

        visualizer.add_datasample(
            osp.basename(img_path),
            img_vis,
            data_sample,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)

        progress_bar.update()


if __name__ == '__main__':
    main()