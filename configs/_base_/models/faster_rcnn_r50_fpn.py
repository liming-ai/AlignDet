_base_ = ['mask_rcnn_r50_fpn.py']

model = dict(
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None))