_base_ = [
    '../../_base_/models/selfsup_mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_soft-teacher.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]


# MomentumUpdateHook is defined in mmselfsup
custom_hooks = [
    dict(type='MomentumUpdateHook'),
    dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'I2B',
                'group': 'pretrain'},
             interval=50,
             num_eval_images=0,
             log_checkpoint=False)]

evaluation = dict(interval=65535, metric='bbox')


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        backbone=dict(
            _delete_=True,
            type='SwinTransformer',
            embed_dims=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            frozen_stages=4,  # frozen all stages
            convert_weights=True,
            init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
        neck=dict(in_channels=[96, 192, 384, 768])))


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)

# There are bugs in mmdet, frozen_stages does not work for swin backbones.
find_unused_parameters = True