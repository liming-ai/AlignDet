_base_ = [
    '../_base_/models/detr_r50.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]

# model
# SyncBN is used by default
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    bbox_head=dict(num_classes=20))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=None)
# learning policy following DETReg and UP-DETR
lr_config = dict(policy='step', step=[70])
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='mAP', save_best='auto')

custom_imports = None