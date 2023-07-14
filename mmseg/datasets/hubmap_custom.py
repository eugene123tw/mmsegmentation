# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch

from mmseg.datasets import CustomDataset
from . import DATASETS


def custom_intersect_and_union(pred_label, label):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
         torch.Tensor: The union of prediction and ground truth
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    area_intersect = (pred_label & label).sum().unsqueeze(0)
    area_union = (pred_label | label).sum().unsqueeze(0)
    area_pred_label = pred_label.sum().unsqueeze(0)
    area_label = label.sum().unsqueeze(0)
    return area_intersect, area_union, area_pred_label, area_label


@DATASETS.register_module()
class HubMAPCustomDataset(CustomDataset):
    CLASSES = (
        # 'background',
        'blood_vessel', )
    PALETTE = [
        # [0, 0, 0],
        [0, 255, 0]
    ]

    def __init__(self, **kwargs):
        super(HubMAPCustomDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(custom_intersect_and_union(pred, seg_map))

        return pre_eval_results
