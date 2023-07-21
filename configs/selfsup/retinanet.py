_base_ = [
    '../_base_/models/selfsup_retinanet_r50_fpn.py',
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


load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='SelectTopKProposals', topk=50)  # avoid GPU out of memory, -1 means load all proposals.
]

data = dict(train=dict(dataset=dict(pipeline=load_pipeline)))
optimizer = dict(lr=0.01)