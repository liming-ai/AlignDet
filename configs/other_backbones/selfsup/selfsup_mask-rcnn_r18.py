_base_ = ['../../selfsup/mask_rcnn.py']


model = dict(
    backbone=dict(
        backbone=dict(
            depth=18,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
        neck=dict(in_channels=[64, 128, 256, 512])))