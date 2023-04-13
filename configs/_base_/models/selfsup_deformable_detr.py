model = dict(
    type='SelfSupDetector',
    backbone=dict(
        type='SelfSupDeformableDETR',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=4,  # Frozen all stages
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            # init_cfg=dict(type='Pretrained', checkpoint='https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar')
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='ChannelMapper',
            in_channels=[512, 1024, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        bbox_head=dict(
            type='SelfSupDeformableDETRHead',
            num_query=300,
            num_classes=256,  # We replace the num_classes here to feature dim
            in_channels=2048,
            sync_cls_avg_factor=True,
            as_two_stage=False,
            transformer=dict(
                type='DeformableDetrTransformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256)
                        ],
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                        'ffn', 'norm')))),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True,
                offset=-0.5),
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32, 64]),
            loss_cls=dict(type='ContrastiveLoss', loss_weight=2.0, temperature=0.5),  # Contrastive loss
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type='SelfSupHungarianAssigner',
                # These costs are not actually used
                cls_cost=dict(type='ZeroCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        test_cfg=dict(max_per_img=100)
    )
)