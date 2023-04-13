_base_ = ['detr_r50_8x2_150e_coco.py']

# learning policy
lr_config = dict(step=[80])
runner = dict(max_epochs=100)