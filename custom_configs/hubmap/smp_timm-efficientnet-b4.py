_base_ = [
    './_base_/datasets/hubmap_custom_512x512_aug.py',
]

num_classes = 1

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

loss = [
    dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    dict(
        type='SMPDiceLoss',
        mode='multilabel',
        alpha=0.1,
        beta=0.9,
        loss_weight=3.0),
    # dict(type='RecallCrossEntropy')
]

model = dict(
    type='SMPUnet',
    backbone=dict(
        type='timm-efficientnet-b4',
        pretrained='noisy-student',
    ),
    decode_head=dict(
        num_classes=num_classes, align_corners=False, loss_decode=loss),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole', multi_class=False))

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

total_iters = 30

optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=int(total_iters * 500))
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(
    by_epoch=False,
    interval=min(500, int(total_iters * 500)),
    metric=['mIoU', 'mFscore'],
    pre_eval=True,
    beta=2,
    save_best='mFscore')
