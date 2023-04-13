_base_ = 'mask_rcnn_r18_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None))

evaluation = dict(interval=1, metric='bbox', save_best='auto', gpu_collect=True)