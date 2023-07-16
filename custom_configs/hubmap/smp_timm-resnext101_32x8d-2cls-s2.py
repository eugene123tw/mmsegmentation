_base_ = [
    './_base_/datasets/hubmap_custom_512x512_s2_aug_2cls.py',
]

num_classes = 3

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

# [0.52584678, 10.17238453]

loss = [
    dict(type='CrossEntropyLoss', loss_weight=1.0),
    dict(type='SMPDiceLoss', mode='multiclass', loss_weight=3.0),
]

model = dict(
    type='SMPUnetPlusPlus',
    backbone=dict(
        type='resnext101_32x8d',
        pretrained='imagenet',
    ),
    decode_head=dict(
        num_classes=num_classes, align_corners=False, loss_decode=loss),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole', multi_class=True))

log_config = dict(interval=50, hooks=[
    dict(type='TextLoggerHook', by_epoch=False),
    dict(type='TensorboardLoggerHook', by_epoch=False)
])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

total_iters = 30

# optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=int(total_iters * 500))
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(
    by_epoch=False,
    interval=min(500, int(total_iters * 500)),
    metric=['mIoU', 'mDice'],
    pre_eval=True,
    save_best='IoU.blood_vessel')
