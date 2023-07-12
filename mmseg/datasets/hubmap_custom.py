# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets import CustomDataset
from . import DATASETS


@DATASETS.register_module()
class HubMAPCustomDataset(CustomDataset):
    CLASSES = ('background', 'blood_vessel', )
    PALETTE = [[0, 0, 0], [0, 255, 0]]

    def __init__(self, **kwargs):
        super(HubMAPCustomDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
