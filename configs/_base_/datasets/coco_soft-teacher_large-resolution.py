# dataset settings
train_dataset_type = 'MultiViewCocoDataset'
test_dataset_type = 'CocoDataset'
data_root = 'data/coco/'

classes = ['selective_search']


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False)
]

train_pipeline1 = [
    dict(
        type='Resize',
        img_scale=[(1600, 400), (1600, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='Pad', size_divisor=32),
    dict(type='RandFlip', flip_ratio=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Identity'),
            dict(type='AutoContrast'),
            dict(type='RandEqualize'),
            dict(type='RandSolarize'),
            dict(type='RandColor'),
            dict(type='RandContrast'),
            dict(type='RandBrightness'),
            dict(type='RandSharpness'),
            dict(type='RandPosterize')
        ]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_pipeline2 = [
    dict(
        type='Resize',
        img_scale=[(1600, 400), (1600, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='Pad', size_divisor=32),
    dict(type='RandFlip', flip_ratio=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Identity'),
            dict(type='AutoContrast'),
            dict(type='RandEqualize'),
            dict(type='RandSolarize'),
            dict(type='RandColor'),
            dict(type='RandContrast'),
            dict(type='RandBrightness'),
            dict(type='RandSharpness'),
            dict(type='RandPosterize')
        ]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
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

# do not evaluate during pre-training
evaluation = dict(interval=65535, gpu_collect=True)