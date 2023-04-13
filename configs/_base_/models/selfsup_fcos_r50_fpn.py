# model settings
model = dict(
    type='SelfSupDetector',
    backbone=dict(
        type='SelfSupFCOS',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=4,  # froze all stages
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5,
            relu_before_extra_convs=True),
        bbox_head=dict(
            type='SelfSupFCOSHead',
            in_channels=256,
            num_classes=256,
            stacked_convs=4,
            feat_channels=256,
            init_cfg=dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=None),
            loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_cls=dict(type='ContrastiveLoss', loss_weight=1.0, temperature=0.5),
            loss_bbox=dict(type='IoULoss', loss_weight=1.0)),
        train_cfg=dict(
            # assigner is not used in FCOS, only defined for the use of sampler
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                gpu_assign_thr=-1),  # make IoU computation on CPU to avoid GPU out of memory
            sampler=dict(
                type='RandomSampler',
                num=2048, # total number of bboxes for each sample
                pos_fraction=1.0,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=1,
            debug=False)))
