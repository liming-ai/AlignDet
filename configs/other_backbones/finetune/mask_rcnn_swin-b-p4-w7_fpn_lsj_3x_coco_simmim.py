_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_ins-lsj.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
pretrained = 'pretrain/simmim/simmim_800e_official.pth'  # noqa
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
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
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[128, 256, 512, 1024],
        norm_cfg=norm_cfg))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,  # Same with original implementation in SimMIM paper.
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[24, 33])
runner = dict(max_epochs=36)

custom_imports = None

# Learning rate and batch size follows the original implementation.
# Please refer to the `Appendix C.1.` in SimMIM paper.
data = dict(samples_per_gpu=4)
auto_scale_lr = dict(enable=True, base_batch_size=32)