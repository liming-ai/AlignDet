_base_ = [
    '../_base_/datasets/coco_soft-teacher_large-resolution.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='SelfSupDetector',
    backbone = dict(
        type='SelfSupCBv2',
        backbone=dict(
            type='CBSwinTransformer',
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
            use_checkpoint=False),
        neck=dict(
            type='CBFPN',
            in_channels=[192, 384, 768, 1536],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
        roi_head=dict(
            type='SelfSupCBv2Head',
            interleaved=True,
            mask_info_flow=True,
            num_stages=3,
            stage_loss_weights=[1, 0.5, 0.25],
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=[
                dict(
                    type='SelfSupShared4Conv1FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=256,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=True,
                    loss_cls=dict(type='ContrastiveLoss', loss_weight=1.0, temperature=0.5),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                loss_weight=1.0)),
                dict(
                    type='SelfSupShared4Conv1FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=256,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(type='ContrastiveLoss', loss_weight=1.0, temperature=0.5),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                loss_weight=1.0)),
                dict(
                    type='SelfSupShared4Conv1FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=256,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(type='ContrastiveLoss', loss_weight=1.0, temperature=0.5),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
            ],
            mask_roi_extractor=None,
            mask_head=None),
        # model training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=[
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False)
            ]),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5))))


custom_hooks = [
    dict(type='MomentumUpdateHook'),
    dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'I2B',
                'group': 'pretrain'},
             interval=50,
             num_eval_images=0,
             log_checkpoint=False)]

find_unused_parameters = True

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# FP16
fp16 = dict(loss_scale='dynamic')

# data = dict(samples_per_gpu=8)