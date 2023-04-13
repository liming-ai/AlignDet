_base_ = ['mask_rcnn_swin-s-p4-w7_fpn_ms-crop_1x_coco.py']

model = dict(
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None))

evaluation = dict(interval=1, metric='bbox', save_best='auto', gpu_collect=True)

# Here the first lr decay is at 27th epoch, following mmdet and official implementation.
lr_config = dict(step=[27, 33])
runner = dict(max_epochs=36)