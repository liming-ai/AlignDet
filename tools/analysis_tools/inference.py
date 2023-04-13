from mmdet.apis import init_detector, inference_detector

# Specify the path to model config and checkpoint file
random_config = 'configs/vis/random.py'
pretrain_config = 'configs/vis/pretrain.py'
finetune_config = 'configs/vis/finetune.py'

sup_pretrain = 'pretrain/finetune_mask-rcnn_1x_coco_lr3e-2_wd5e-5/epoch_12.pth'
our_pretrain = 'pretrain/selfsup_mask-rcnn_mstrain-soft-teacher_sampler-4096_temp0.5/final_model.pth'

# build the model from a config file and a checkpoint file
random_model = init_detector(pretrain_config, None,         device='cuda:0')
sup_model    = init_detector(finetune_config, sup_pretrain, device='cuda:0')
our_model    = init_detector(pretrain_config, our_pretrain, device='cuda:0')

# test a single image and show the results
img = '000000339129.jpg'  # or img = mmcv.imread(img), which will only load it once

result = inference_detector(random_model, img)
random_model.CLASSES = ['object']
random_model.show_result(img, result, out_file='random.jpg')

result = inference_detector(sup_model, img)
sup_model.show_result(img, result, out_file='sup.jpg')

result = inference_detector(our_model, img)
our_model.CLASSES = ['object']
our_model.show_result(img, result, out_file='result.jpg')