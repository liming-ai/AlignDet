_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
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
optimizer = dict(lr=0.01)

custom_imports = None