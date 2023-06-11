_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py',
    '../_base_/datasets/hubmap_cityscapes_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3),
    test_cfg=dict(mode='whole')
)
evaluation = dict(metric='mDice')
data = dict(samples_per_gpu=4, workers_per_gpu=4)
runner = dict(type='IterBasedRunner', max_iters=1000)