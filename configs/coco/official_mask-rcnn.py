_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# norm_cfg = dict(type='SyncBN', requires_grad=True)
# model = dict(
#     backbone=dict(norm_cfg=norm_cfg),
#     neck=dict(norm_cfg=norm_cfg))

custom_imports = None