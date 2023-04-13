_base_ = ['../../coco/mask_rcnn_r50_fpn_1x_coco.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(in_channels=[24, 32, 96, 1280]))