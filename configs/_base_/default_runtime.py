checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'I2B',
                'group': 'finetune'},
             interval=50,
             num_eval_images=0,
             log_checkpoint=False)]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

# import custom files
custom_imports = dict(
    imports=[
        # loading mmselfsup directly
        'mmselfsup.datasets.pipelines',
        # momentum update hook
        'selfsup.core.hook.momentum_update_hook',
        # custom pipelines
        'selfsup.datasets.pipelines.selfsup_pipelines',
        'selfsup.datasets.pipelines.rand_aug',
        'selfsup.datasets.single_view_coco',
        'selfsup.datasets.multi_view_coco',
        # loss
        'selfsup.models.losses.contrastive_loss',
        # custom heads
        'selfsup.models.dense_heads.fcos_head',
        'selfsup.models.dense_heads.retina_head',
        'selfsup.models.dense_heads.detr_head',
        'selfsup.models.dense_heads.deformable_detr_head',
        'selfsup.models.roi_heads.bbox_heads.convfc_bbox_head',
        'selfsup.models.roi_heads.standard_roi_head',
        'selfsup.models.roi_heads.htc_roi_head',
        'selfsup.models.roi_heads.cbv2_roi_head',
        # necks
        'selfsup.models.necks.cb_fpn',
        # backbones for cbv2
        'selfsup.models.backbones.cbv2',
        'selfsup.models.backbones.swinv1',
        # custom detectors
        'selfsup.models.detectors.selfsup_detector',
        'selfsup.models.detectors.selfsup_fcos',
        'selfsup.models.detectors.selfsup_detr',
        'selfsup.models.detectors.selfsup_deformable_detr',
        'selfsup.models.detectors.selfsup_retinanet',
        'selfsup.models.detectors.selfsup_mask_rcnn',
        'selfsup.models.detectors.selfsup_htc',
        'selfsup.models.detectors.selfsup_cbv2',
        'selfsup.models.detectors.cbv2',
        # DETR pretraining
        'selfsup.core.bbox.assigners.hungarian_assigner',
        'selfsup.core.bbox.assigners.pseudo_hungarian_assigner',
        'selfsup.core.bbox.match_costs.match_cost'], allow_failed_imports=False)
