_base_ = [
    '../_base_/models/selfsup_detr_r50.py',
    '../_base_/datasets/coco_soft-teacher.py',
    '../_base_/default_runtime.py'
]

classes = [f'cluster_{i+1}' for i in range(256)]

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
    dict(type='SelectTopKProposals', topk=50)  # This value should be smaller than the number of queries
]

data_root = 'data/coco/'
data = dict(
    train=dict(
        dataset=dict(
            classes=classes,
            ann_file=data_root + 'filtered_proposals/train2017_ratio3size0008@0.5_cluster-id-as-class.json',
            pipeline=load_pipeline)))


# optimizer, sqrt lr scale is used for pretrain
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
auto_scale_lr = dict(enable=True, base_batch_size=64)
