# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

import cv2
import mmcv
import numpy as np
import pandas as pd
from datumaro import DatasetItem, Mask, Polygon
from datumaro.components.project import Dataset
from sklearn.model_selection import KFold

# Tiles from Dataset 1 have annotations that have been expert reviewed.
# Tiles from Dataset 2 contains sparse annotations that have NOT been expert reviewed.

# All of the test set tiles are from Dataset 1 (reviewed by experts).
# The training annotations contains Dataset 2 tiles from the public test WSI, but not from the private test WSI.
# Two of the WSIs make up the training set, two WSIs make up the public test set, and one WSI makes up the private test set.

BBOX_INDEX = 0
MASK_INDEX = 1
LABLE_INDEX = 1  # ONLY PICK BLOOD VESSEL


def polygon_to_bitmask(coordinates, width, height):
    # Create a blank image with the given width and height
    coordinates = np.array(coordinates).astype(np.int32)
    bitmask = np.zeros(shape=(height, width))
    cv2.fillPoly(bitmask, coordinates, 1)
    bitmask = bitmask.astype(bool)
    return bitmask


class HuBMAPVasculatureDataset:

    def __init__(self, data_root) -> None:
        self.data_root = data_root
        # self.labels = ['glomerulus', 'blood_vessel', 'unsure']
        # self.labels = ['blood_vessel']
        self.labels = ['blood_vessel']
        self.df = pd.read_csv(os.path.join(self.data_root, 'tile_meta.csv'))
        np.random.seed(42)
        self.dsitem_dict = self._make_dsitems()

    def _make_dsitems(self):
        train_root = os.path.join(self.data_root, 'train')
        with open(os.path.join(self.data_root, 'polygons.jsonl'),
                  'r') as json_file:
            results = list(json_file)

        dsitem_dict = {}
        for index, result in enumerate(results):
            result = json.loads(result)
            image_id = result['id']
            img = cv2.imread(os.path.join(train_root, f'{image_id}.tif'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            attributes = {'filename': f'{self.data_root}/train/{image_id}.tif'}
            datumaro_masks = []

            for anno in result['annotations']:
                if anno['type'] not in self.labels:
                    continue

                label_idx = self.labels.index(anno['type'])
                bitmask = polygon_to_bitmask(anno['coordinates'], img.shape[1],
                                             img.shape[0])
                mask = Mask(image=bitmask, label=label_idx)
                datumaro_masks.append(mask)

            if len(datumaro_masks):
                dsitem_dict[image_id] = DatasetItem(
                    id=image_id,
                    annotations=datumaro_masks,
                    image=img,
                    attributes=attributes)
        return dsitem_dict

    def analyse_dataset(self):
        with open(os.path.join(self.data_root, 'polygons.jsonl'),
                  'r') as json_file:
            results = list(json_file)

        labels = ['glomerulus', 'blood_vessel', 'unsure']

        polygon_areas = {label: [] for label in labels}
        for index, result in enumerate(results):
            result = json.loads(result)
            for anno in result['annotations']:
                polygon = Polygon(
                    points=np.array(anno['coordinates']).flatten())
                polygon_areas[anno['type']].append(polygon.get_area())

        for label in labels:
            print(f'Max Area of {label}: {np.max(polygon_areas[label])}')
            print(f'Min Area of {label}: {np.min(polygon_areas[label])}')
            print(f'Medium Area of {label}: {np.median(polygon_areas[label])}')
            print(f'Std Area of {label}: {np.std(polygon_areas[label])}')
            avg_area = np.mean(polygon_areas[label])
            low_3std = avg_area - 2 * np.std(polygon_areas[label])
            high_3std = avg_area + 2 * np.std(polygon_areas[label])

            print(f'Area Percentage of {label}: {avg_area / (512**2) * 100}%')
            print(
                f'2 std Area of {label}: {low_3std} < {avg_area} < {high_3std}'
            )
            print('\n')

    def strategy_1(self):
        """Train on Dataset 1, test on Dataset 2."""
        dsitems = []
        for index, row in self.df.iterrows():
            if self.dsitem_dict.get(row['id']) is not None:
                dsitem = self.dsitem_dict[row['id']]
                if row['dataset'] == 1:
                    dsitem.subset = 'train'
                elif row['dataset'] == 2:
                    dsitem.subset = 'val'
                dsitems.append(dsitem)
        return dsitems

    def strategy_2(self):
        """Train on Dataset 2, test on Dataset 1."""
        dsitems = []
        for index, row in self.df.iterrows():
            if self.dsitem_dict.get(row['id']) is not None:
                dsitem = self.dsitem_dict[row['id']]
                if row['dataset'] == 2:
                    dsitem.subset = 'train'
                elif row['dataset'] == 1:
                    dsitem.subset = 'val'
                dsitems.append(dsitem)
        return dsitems

    def strategy_3(self):
        """Train on Dataset 1, test on Dataset 1."""
        pass

    def strategy_4(self):
        """Train on WSI_1 (Dataset 1 + Dataset 2) , test on WSI_2 (Dataset
        1)"""
        dsitems = []
        for index, row in self.df.iterrows():
            if self.dsitem_dict.get(row['id']) is not None:
                dsitem = self.dsitem_dict[row['id']]
                if row['source_wsi'] == 1:
                    dsitem.subset = 'train'
                    dsitems.append(dsitem)
                elif row['source_wsi'] == 2 and row['dataset'] == 1:
                    dsitem.subset = 'val'
                    dsitems.append(dsitem)
        return dsitems

    def strategy_5(self):
        """Train on wsi 3, 4 + dataset 2, test on wsi 1,2 + dataset 1."""
        dsitems = []
        for index, row in self.df.iterrows():
            if self.dsitem_dict.get(row['id']) is not None:
                dsitem = self.dsitem_dict[row['id']]
                if row['dataset'] == 2:
                    if row['source_wsi'] == 3 or row['source_wsi'] == 4:
                        dsitem.subset = 'train'
                        dsitems.append(dsitem)

                if row['dataset'] == 1:
                    if row['source_wsi'] == 1 or row['source_wsi'] == 2:
                        dsitem.subset = 'val'
                        dsitems.append(dsitem)
        return dsitems

    def export(self, dsitems, export_path):
        mmcv.mkdir_or_exist(os.path.join(export_path, 'images'))
        mmcv.mkdir_or_exist(os.path.join(export_path, 'images', 'train'))
        mmcv.mkdir_or_exist(os.path.join(export_path, 'images', 'val'))
        mmcv.mkdir_or_exist(os.path.join(export_path, 'annotations'))
        mmcv.mkdir_or_exist(os.path.join(export_path, 'annotations', 'train'))
        mmcv.mkdir_or_exist(os.path.join(export_path, 'annotations', 'val'))

        for dsitem in dsitems:
            img_folder = os.path.join(export_path, 'images', dsitem.subset)
            anno_folder = os.path.join(export_path, 'annotations',
                                       dsitem.subset)
            image_id = dsitem.id
            anno = np.zeros(
                (dsitem.media.data.shape[0], dsitem.media.data.shape[1]))
            for mask in dsitem.annotations:
                anno += mask.image
            anno = np.clip(anno, 0, 1)

            mmcv.imwrite(
                cv2.cvtColor(dsitem.media.data, cv2.COLOR_RGB2BGR),
                os.path.join(img_folder, f'{image_id}.png'))

            mmcv.imwrite(anno, os.path.join(anno_folder, f'{image_id}.png'))

    def _get_unannotated_images(self):
        image_list = []
        for index, row in self.df.iterrows():
            if row['dataset'] == 3:
                image_list.append(row)
        return image_list

    def class_balance_weights(self, dsitems):
        y = [0, 0]
        for dsitem in dsitems:
            anno = np.zeros(
                (dsitem.media.data.shape[0], dsitem.media.data.shape[1]))
            for mask in dsitem.annotations:
                anno += mask.image
            anno = np.clip(anno, 0, 1)
            y[0] += np.sum(anno == 0)
            y[1] += np.sum(anno == 1)
        print(y[0] / y[1])


if __name__ == '__main__':
    dataset = HuBMAPVasculatureDataset(
        data_root='/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature'
    )
    dsitems = dataset.strategy_5()
    dataset.class_balance_weights(dsitems)
    # dataset.export(
    #     dsitems,
    #     export_path='/home/yuchunli/_DATASET/HuBMAP-vasculature-custom-s5')

    dataset.analyse_dataset()
