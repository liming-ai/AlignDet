_base_ = [
    '../_base_/models/selfsup_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_soft-teacher.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]


custom_hooks = [
    dict(type='MomentumUpdateHook'),
    dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'AlignDet',
                'group': 'pretrain'},
             interval=50,
             num_eval_images=0,
             log_checkpoint=False)]