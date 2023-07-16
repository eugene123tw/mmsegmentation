# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmseg import digit_version
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import build_dp, get_device, setup_multi_processes
from mmseg.models import build_segmentor

from collections import OrderedDict


import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import eval_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg search mask threshold')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    args = parser.parse_args()
    return args


def patch_evaluate(self, results, metric='mIoU', logger=None, gt_seg_maps=None, **kwargs):
    if isinstance(metric, str):
        metric = [metric]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metric).issubset(set(allowed_metrics)):
        raise KeyError('metric {} is not supported'.format(metric))

    threshold = kwargs.get('threshold', 0.5)

    eval_results = {}
    # test a list of files

    # only get blood vessel mask
    results = [result[2] > threshold for result in results]

    gt_seg_maps = np.array(list(self.get_gt_seg_maps()))
    gt_seg_maps[gt_seg_maps == 1] = 0  # replace glomerulus with background
    gt_seg_maps[gt_seg_maps == 2] = 1  # replace blood vessel with foreground

    num_classes = 2  # Background, Blood Vessel
    ret_metrics = eval_metrics(
        results,
        gt_seg_maps,
        num_classes,
        self.ignore_index,
        metric,
        label_map=dict(),
        reduce_zero_label=False)

    # Because dataset.CLASSES is required for per-eval.
    class_names = ('background', 'blood_vessel')

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print_log('per class results:', logger)
    print_log('\n' + class_table_data.get_string(), logger=logger)
    print_log('Summary:', logger)
    print_log('\n' + summary_table_data.get_string(), logger=logger)

    # each metric dict
    for key, value in ret_metrics_summary.items():
        if key == 'aAcc':
            eval_results[key] = value / 100.0
        else:
            eval_results['m' + key] = value / 100.0

    ret_metrics_class.pop('Class', None)
    for key, value in ret_metrics_class.items():
        eval_results.update({
            key + '.' + str(name): value[idx] / 100.0
            for idx, name in enumerate(class_names)
        })

    return eval_results


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.model.test_cfg.get_prob = True

    # monkey patch evaluate in HubMAPCustomDataset
    from mmseg.datasets.hubmap_custom import HubMAPCustomDataset
    HubMAPCustomDataset.evaluate = patch_evaluate

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(num_gpus=1, shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    cfg.device = get_device()

    warnings.warn(
        'SyncBN is only supported with DDP. To be compatible with DP, '
        'we convert SyncBN to BN. Please use dist_train.sh which can '
        'avoid this error.')
    if not torch.cuda.is_available():
        assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
            'Please use MMCV >= 1.4.4 for CPU training!'
    model = revert_sync_batchnorm(model)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    results = single_gpu_test(model, data_loader)

    best_threshold = 0.0
    best_score = 0.0
    for threshold in np.arange(0.0, 1.0, 0.01):
        metric = dataset.evaluate(results, 'mFscore', threshold=threshold)
        score = metric['Fscore.blood_vessel']
        if score > best_score:
            best_score = score
            best_threshold = threshold
    print(f'blood vessel: best threshold: {best_threshold}, best score: {best_score}')


if __name__ == '__main__':
    main()
