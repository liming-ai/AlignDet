_base_ = [
    '../_base_/models/selfsup_deformable_detr.py',
    '../_base_/datasets/coco_soft-teacher.py',
    '../_base_/default_runtime.py'
]


custom_hooks = [
    dict(type='MomentumUpdateHook'),
    dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'I2B',
                'group': 'pretrain' },
             interval=50,
             num_eval_images=0,
             log_checkpoint=False)]

load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='SelectTopKProposals', topk=150)  # same with DETR, this value should be smaller than the number of queries
]
data = dict(train=dict(dataset=dict(pipeline=load_pipeline)))

# optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
auto_scale_lr = dict(enable=True, base_batch_size=32)