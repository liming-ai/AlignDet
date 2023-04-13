_base_ = ['mask_rcnn_swin-s-p4-w7_fpn_ms-crop_1x_coco.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=128,
        num_heads=[4, 8, 16, 32],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512, 1024]))