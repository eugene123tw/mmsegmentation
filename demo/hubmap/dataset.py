import os
import json
import numpy as np
import cv2
from datumaro import DatasetItem, Mask
from datumaro.components.project import Dataset
from sklearn.model_selection import KFold


class HuBMAPVasculatureDataset:
    """Ship Detection from HuggingFace Dataset."""

    def __init__(self, data_root) -> None:
        self.data_root = data_root
        self.labels = ['glomerulus', 'blood_vessel', 'unsure']
        # self.labels = ['blood_vessel']

    def polygon_to_bitmask(self, coordinates, width, height):
        # Create a blank image with the given width and height
        coordinates = np.array(coordinates).astype(np.int32)
        bitmask = np.zeros(shape=(height, width))
        cv2.fillPoly(bitmask, coordinates, 1)
        bitmask = bitmask.astype(bool)
        return bitmask

    def make_coco_train(self, export_path, fold=5):
        train_root = os.path.join(self.data_root, 'train')
        with open(os.path.join(self.data_root, 'polygons.jsonl'), 'r') as json_file:
            results = list(json_file)

        dsitems = []
        for index, result in enumerate(results):
            result = json.loads(result)
            image_id = result['id']
            img = cv2.imread(os.path.join(train_root, f'{image_id}.tif'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            attributes = {'filename': f'{self.data_root}/train/{image_id}.tif'}
            datumaro_masks = []
            width, height = img.shape[1], img.shape[0]
            for anno in result['annotations']:
                bitmask = self.polygon_to_bitmask(anno['coordinates'], width, height)
                if anno['type'] not in self.labels:
                    continue
                label_idx = self.labels.index(anno['type'])
                mask = Mask(image=bitmask, label=label_idx)
                datumaro_masks.append(mask)

            dsitems.append(
                DatasetItem(
                    id=index,
                    annotations=datumaro_masks,
                    image=img,
                    attributes=attributes))

        kf = KFold(n_splits=fold, shuffle=True, random_state=42)
        for fold, (train_indices, val_indices) in enumerate(kf.split(dsitems)):
            for index in train_indices:
                dsitems[index].subset = f'train'
            for index in val_indices:
                dsitems[index].subset = f'val'

            dataset = Dataset.from_iterable(dsitems, categories=self.labels)
            dataset.export(f"{export_path}-fold-{fold}", 'cityscapes', default_image_ext='.tif', save_media=True)


if __name__ == '__main__':
    dataset = HuBMAPVasculatureDataset(
        data_root='/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature')

    dataset.make_coco_train(export_path="/home/yuchunli/_DATASET/HuBMAP-vasculature-cityscapes")
