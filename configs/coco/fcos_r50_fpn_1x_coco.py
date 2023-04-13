# please refer to this config:
# https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py

_base_ = [
    '../_base_/models/fcos_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# SyncBN is used by default
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg))

# optimizer
optimizer = dict(lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

custom_imports = None