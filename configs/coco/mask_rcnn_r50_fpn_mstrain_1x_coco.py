_base_ = 'mask_rcnn_r50_fpn_1x_coco.py'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(frozen_stages=-1, norm_cfg=norm_cfg, norm_eval=False),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(train=dict(pipeline=train_pipeline))

custom_imports = None