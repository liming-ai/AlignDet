# Data
## Data Download
Please download the [COCO 2017 dataset](https://cocodataset.org/), and the folder structure is:
```
data
├── coco
│   ├── annotations
│   ├── filtered_proposals
│   ├── semi_supervised_annotations
│   ├── test2017
│   ├── train2017
│   └── val2017
```

## Environments
```bash
pip3 install openmim seaborn
mim install mmselfsup mmdet
```

# Pre-training and Fine-tuning
## Pre-training Example
```bash
bash tools/dist_train.sh configs/selfsup/mask_rcnn.py 8 --work-dir work_dirs/selfsup_mask-rcnn
```

## Fine-tuning Example
1. Using `tools/model_converters/extract_detector_weights.py` to extract the weights.
```bash
python3 tools/model_converters/extract_detector_weights.py \
work_dirs/selfsup_mask-rcnn/epoch_12.pth  \ # pretrain weights
work_dirs/selfsup_mask-rcnn/final_model.pth  # finetune weights
```

2. Fine-tuning models like normal mmdet training process, usually the learning rate is increased by 1.5 times, and the weight decay is reduced to half of the original setting.
```bash
bash tools/dist_train.sh configs/coco/mask_rcnn_r50_fpn_1x_coco.py 8 \
--cfg-options load_from=work_dirs/selfsup_mask-rcnn/final_model.pth \ # load weights
optimizer.lr=3e-2 optimizer.weight_decay=5e-5  \ # adjust lr and wd
--work-dir work_dirs/finetune_mask-rcnn_1x_coco_lr3e-2_wd5e-5
```

