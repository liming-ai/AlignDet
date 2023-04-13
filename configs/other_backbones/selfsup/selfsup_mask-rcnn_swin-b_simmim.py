_base_ = ['selfsup_mask-rcnn_swin-b.py']

# Download the pretrained model from https://github.com/microsoft/SimMIM
pretrained = 'pretrain/simmim/simmim_800e_official.pth'  # noqa
model = dict(
    backbone=dict(
        backbone=dict(
            init_cfg=dict(type='Pretrained', checkpoint=pretrained))))

# Learning rate and batch size follows the original implementation.
# Please refer to the `Appendix C.1.` in SimMIM paper.
optimizer = dict(lr=6e-5)
data = dict(samples_per_gpu=4)
auto_scale_lr = dict(enable=True, base_batch_size=32)