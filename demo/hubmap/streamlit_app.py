# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

import cv2
import mmcv
import numpy as np
import streamlit as st
from mmcv.runner import load_checkpoint

from demo.hubmap.run_prediction import mask_to_polygons
from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor

IMAGE_PATH = '/home/yuchunli/_DATASET/HuBMAP-vasculature-custom-s5/images/val'
ANNOTATION_PATH = '/home/yuchunli/_DATASET/HuBMAP-vasculature-custom-s5/annotations/val'

CONFIG = '/home/yuchunli/git/mmsegmentation/work_dirs/smp_timm-resnext101_32x8d/smp_timm-resnext101_32x8d.py'
CKPT = '/home/yuchunli/git/mmsegmentation/work_dirs/smp_timm-resnext101_32x8d/best_mFscore_iter_1000.pth'


@st.cache_data
def get_images_list(path_to_folder: str) -> list:
    """Return the list of images from folder
    Args:
        path_to_folder (str): absolute or relative path to the folder with images
    """
    image_names_list = [
        x for x in os.listdir(path_to_folder)
        if x[-3:] in ['jpg', 'peg', 'png', 'tif']
    ]
    return image_names_list


@st.cache_data
def load_image(image_name: str, path_to_folder: str, bgr2rgb: bool = True):
    """Load the image
    Args:
        image_name (str): name of the image
        path_to_folder (str): path to the folder with image
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    path_to_image = os.path.join(path_to_folder, image_name)
    image = cv2.imread(path_to_image)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@st.cache_data
def load_annotation(image_name: str, path_to_folder: str):
    anno_path = Path(path_to_folder)
    anno_name = Path(image_name).stem + '.png'

    file_client = mmcv.FileClient()
    img_bytes = file_client.get(anno_path / anno_name)
    gt_semantic_seg = mmcv.imfrombytes(
        img_bytes, flag='unchanged',
        backend='pillow').squeeze().astype(np.uint8)
    return gt_semantic_seg


def blend(image, mask):
    """Blend image with mask
    Args:
        image (np.ndarray): image
        mask (np.ndarray): mask
    Returns:
        blended (np.ndarray): blended image
    """
    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    mask = np.repeat((mask > 0)[:, :, np.newaxis], 3, axis=2)
    mask = mask * color_mask

    blended = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
    return blended


def select_image(path_to_images: str, path_to_annotations: str, thres=0.3):
    """ Show interface to choose the image, and load it
    Args:
        path_to_images (dict): path ot folder with images
        interface_type (dict): mode of the interface used
    Returns:
        (status, image)
        status (int):
            0 - if everything is ok
            1 - if there is error during loading of image file
            2 - if user hasn't uploaded photo yet
    """
    image_names_list = get_images_list(path_to_images)
    if len(image_names_list) < 1:
        return 1, 0
    else:
        image_name = st.sidebar.selectbox('Select an image:', image_names_list)
        try:
            image = load_image(image_name, path_to_images)
            # load annotation
            mask = load_annotation(image_name, path_to_annotations)
            gt_blended = blend(image, mask)

            pred_mask = hubmap_single_seg_model(
                image, CONFIG, CKPT, thres=thres)
            pred_blended = blend(image, pred_mask)

            return 0, gt_blended, pred_blended
        except cv2.error:
            return 1, 0, 0


@st.cache_data
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
    config.model.pretrained = None
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


def hubmap_single_seg_model(img, config, ckpt, thres=0.3):

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, ckpt)

    results = inference_segmentor(model, img)
    result = results[0]
    # index 0 belongs to the background class
    mask = result[0]
    encoded_strings, scores, full_mask = mask_to_polygons(
        mask, threshold=thres, debug=True)
    return full_mask


def main():
    if not os.path.isdir(IMAGE_PATH):
        st.title('There is no directory: ' + IMAGE_PATH)

    thres = st.sidebar.slider('Threshold', 0.0, 1.0, 0.3, 0.01)

    # select image
    status, image, pred_blended = select_image(IMAGE_PATH, ANNOTATION_PATH,
                                               thres)
    if status == 1:
        st.title("Can't load image")
    if status == 2:
        st.title('Please, upload the image')
    else:
        st.image([image, pred_blended],
                 caption=['Original image', 'Predicted mask'])

        # comment about refreshing
        st.write("*Press 'R' to refresh*")


if __name__ == '__main__':
    main()
