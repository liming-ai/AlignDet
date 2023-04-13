_base_ = [
    '../_base_/models/fcos_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_12k.py',
    '../_base_/default_runtime.py'
]

# SyncBN is used by default
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    bbox_head=dict(num_classes=20))

# optimizer following coco
optimizer = dict(lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=12000)
evaluation = dict(interval=12000)
custom_imports = None