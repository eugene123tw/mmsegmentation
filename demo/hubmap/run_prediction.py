# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os
import typing as t
import zlib
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
import pandas as pd
from mmcv.runner import load_checkpoint
from pycocotools import _mask as coco_mask

from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor


def get_img_paths(folder):
    """Get image paths from the input folder."""
    img_paths = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != bool:
        raise ValueError(
            'encode_binary_mask expects a binary mask, received dtype == %s' %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            'encode_binary_mask expects a 2d mask, received shape == %s' %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]['counts']

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def _fill_poly(canvas, contour, full_mask):
    if full_mask is not None:
        cv2.fillPoly(full_mask, pts=[contour], color=1)
    cv2.fillPoly(canvas, pts=[contour], color=1)


def _copy_paste(canvas, score_map, x1, y1, x2, y2):
    canvas[y1:y2, x1:x2] = score_map > 0.0
    return encode_binary_mask(canvas.astype(bool))


def mask_to_polygons(prob_mask,
                     threshold=0.5,
                     area_threshold=100,
                     img_h=512,
                     img_w=512,
                     debug=False,
                     padding=1):

    encoded_strings = []
    scores = []

    bitmask = (prob_mask >= threshold).astype(np.uint8)
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    bitmask = cv2.dilate(bitmask, kernel, 3)
    bitmask = np.pad(bitmask, (padding, padding), mode='constant')

    contours, hierarchies = cv2.findContours(
        bitmask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchies is None:
        return [], []

    if debug:
        full_mask = np.zeros((img_h + 2 * padding, img_w + 2 * padding),
                             dtype=np.uint8)
    else:
        full_mask = None

    for contour, hierarchy in zip(contours, hierarchies[0]):
        # skip inner contours
        if hierarchy[3] != -1 or len(contour) <= 2 or cv2.contourArea(
                contour) < area_threshold:
            continue
        canvas = np.zeros((img_h + 2 * padding, img_w + 2 * padding),
                          dtype=np.uint8)

        x1, x2 = min(contour[:, 0, 0]), max(contour[:, 0, 0])
        y1, y2 = min(contour[:, 0, 1]), max(contour[:, 0, 1])
        prob_crop = prob_mask[y1:y2, x1:x2]
        bit_crop = bitmask[y1:y2, x1:x2]
        score_map = prob_crop * bit_crop
        score = np.mean(score_map[bit_crop == 1])

        # Method 1: fill contour
        _fill_poly(canvas, contour, full_mask)
        canvas = canvas[padding:-padding, padding:-padding]
        encoded_string = encode_binary_mask(canvas.astype(bool))

        # Method 2: crop prob map and paste it on canvas
        # encoded_string = _copy_paste(canvas, score_map, x1, y1, x2, y2)

        encoded_strings.append(encoded_string)
        scores.append(score)
    if debug:
        full_mask = full_mask[padding:-padding, padding:-padding]
    return encoded_strings, scores, full_mask


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.backbone.pretrained = None
    config.model.train_cfg = None
    config.model.test_cfg.get_prob = True

    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def hubmap_single_seg_model(image_root, config, ckpt):

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, ckpt)

    # get image paths
    img_paths = get_img_paths(image_root)

    ids = []
    heights = []
    widths = []
    prediction_strings = []

    for img_path in img_paths:
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        results = inference_segmentor(model, img_path)
        result = results[0]
        # index 0 belongs to the background class
        mask = result[2]
        pred_string = ''
        encoded_strings, scores, _ = mask_to_polygons(mask)
        scores = np.array(scores)
        indices = np.argsort(scores)[::-1]

        scores = scores[indices]
        encoded_strings = [encoded_strings[i] for i in indices]

        n = 0
        for encoded_string, score in zip(encoded_strings, scores):
            if n == 0:
                pred_string += f"0 {score} {encoded_string.decode('utf-8')}"
            else:
                pred_string += f" 0 {score} {encoded_string.decode('utf-8')}"
                n += 1
        height, width = cv2.imread(str(img_path)).shape[:2]
        ids.append(image_id)
        prediction_strings.append(pred_string)
        heights.append(height)
        widths.append(width)
    return ids, prediction_strings, heights, widths


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', help='Image Path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    args = parser.parse_args()
    ids, prediction_strings, heights, widths = hubmap_single_seg_model(
        args.path, args.config, args.checkpoint)
    submission = pd.DataFrame()
    submission['id'] = ids
    submission['height'] = heights
    submission['width'] = widths
    submission['prediction_string'] = prediction_strings
    submission = submission.set_index('id')
    submission.to_csv('submission.csv')
    print(submission)
