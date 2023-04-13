_base_ = ['../../selfsup/mask_rcnn.py']


model = dict(
    backbone=dict(
        backbone=dict(
            _delete_=True,
            type='MobileNetV2',
            frozen_stages=7,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
        neck=dict(in_channels=[24, 32, 96, 1280])))

find_unused_parameters = True