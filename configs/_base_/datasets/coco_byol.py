# dataset settings
train_dataset_type = 'MultiViewCocoDataset'
test_dataset_type = 'CocoDataset'
data_root = 'data/coco/'

classes = ['selective_search']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

load_pipeline = [
    dict(type='LoadImageFromFile', channel_order='rgb'),  # default order is BGR
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pad', size_divisor=32)
]

# following BYOL
train_pipeline1 = [
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal']),
    dict(type='ImgToTensor', keys=['img']),  # (H, W, C) -> (C, H, W)
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0),
    dict(type='Solarization', p=0.),
    dict(type='TensorNormalize', **img_norm_cfg),
    dict(type='SelfSupFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
train_pipeline2 = [
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal']),
    dict(type='ImgToTensor', keys=['img']),  # (H, W, C) -> (C, H, W)
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0),
    dict(type='Solarization', p=0.2),
    dict(type='TensorNormalize', **img_norm_cfg),
    dict(type='SelfSupFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MultiViewCocoDataset',
        dataset=dict(
            type='CocoDataset',
            classes=classes,
            ann_file=data_root + 'filtered_proposals/train2017_ratio3size0008@0.5.json',
            img_prefix=data_root + 'train2017/',
            pipeline=load_pipeline),
        num_views=2,
        pipelines=[train_pipeline1, train_pipeline2]),
    val=dict(
        type=test_dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=test_dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'], save_best='auto', gpu_collect=True)
