_base_ = [
    '../_base_/models/fcos_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_90k.py',
    '../_base_/default_runtime.py'
]

# SyncBN is used by default
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg))

# optimizer
optimizer = dict(lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'I2B',
                'group': 'semi-coco'},
             interval=50,
             num_eval_images=0,
             log_checkpoint=False)]

# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3)

data_root = 'data/coco/'
data = dict(train=dict(ann_file=data_root + 'semi_supervised_annotations/instances_train2017.1@1.json'))
evaluation = dict(interval=90000)
checkpoint_config = dict(interval=90000)
custom_imports = None