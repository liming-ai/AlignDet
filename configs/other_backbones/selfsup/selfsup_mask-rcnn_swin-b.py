_base_ = ['selfsup_mask-rcnn_swin-s.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'  # noqa
model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=128,
            num_heads=[4, 8, 16, 32],
            frozen_stages=4,  # frozen all stages
            init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
        neck=dict(in_channels=[128, 256, 512, 1024])))

# There are bugs in mmdet, frozen_stages does not work for swin backbones.
find_unused_parameters = True