# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def get_img_paths(folder):
    """Get image paths from the input folder."""
    img_paths = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def main():
    parser = ArgumentParser()
    parser.add_argument('path', help='Image Path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='hubmap',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # get image paths
    img_paths = get_img_paths(args.path)

    # show the results
    for i in range(len(img_paths)):
        result = inference_segmentor(model, img_paths[i])
        show_result_pyplot(
            model,
            img_paths[i],
            result,
            get_palette(args.palette),
            opacity=args.opacity,
            out_file=args.out_file)


if __name__ == '__main__':
    main()
