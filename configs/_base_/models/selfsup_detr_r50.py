# model settings
model = dict(
    type='SelfSupDetector',
    backbone=dict(
        type='SelfSupDETR',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            frozen_stages=4,  # Frozen all stages
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            # init_cfg=dict(type='Pretrained', checkpoint='https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar')),
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        bbox_head=dict(
            type='SelfSupDETRHead',
            num_classes=256,  # We replace the num_classes here to feature dim
            in_channels=2048,
            transformer=dict(
                type='Transformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DetrTransformerDecoder',
                    return_intermediate=True,
                    num_layers=6,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm')),
                )),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),  # Contrastive loss
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            loss_contrastive=dict(type='ContrastiveLoss', loss_weight=1.0, temperature=0.5),
            query_init_path=None,
            obj_assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='ZeroCost', weight=0),  # only use coordinate for instance-level assignment
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))
        ),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                # These costs are not actually used
                cls_cost=dict(type='ClassificationCost', weight=1.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        test_cfg=dict(max_per_img=100))
)
