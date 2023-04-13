_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_12k.py',
    '../_base_/default_runtime.py'
]

# SyncBN is used by default
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        norm_cfg=norm_cfg),
    neck=dict(
        norm_cfg=norm_cfg),
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'I2B',
                'group': 'semi-coco'},
             interval=50,
             num_eval_images=0,
             log_checkpoint=False)]

data_root = 'data/coco/'
data = dict(train=dict(ann_file=data_root + 'semi_supervised_annotations/instances_train2017.1@1.json'))
evaluation = dict(interval=12000)
checkpoint_config = dict(interval=12000)
custom_imports = None