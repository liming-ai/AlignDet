import torch
import random
import torchvision
import argparse
import cv2 as cv
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from torchvision import transforms
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature maps')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('image', help='image path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--threshold', default=0.1, type=float, help='visualization threshold')
    parser.add_argument('--seed', default=0, type=int, help='Random Seed')

    args = parser.parse_args()
    return args

def img2tensor(img):
    img = cv.imread(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((2688, 1600)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = transform(np.array(img))

    return img.unsqueeze(0)


def set_random_seed(seed=1, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_random_seed(args.seed)
    cfg = Config.fromfile(args.config)
    sup_cfg = Config.fromfile('configs/coco/official_mask-rcnn.py')

    writer = SummaryWriter(log_dir='featmaps', flush_secs=60)

    cmap = 'coolwarm'
    fontsize = 20

    # build the model
    cfg.model.train_cfg = None
    pretrained_model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    random_init_model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    sup_model = build_detector(sup_cfg.model, test_cfg=cfg.get('test_cfg'))

    sup_model.init_weights()
    pretrained_model.init_weights()
    random_init_model.init_weights()
    load_checkpoint(sup_model, 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth', map_location='cpu')

    pretrained_model.eval()
    random_init_model.eval()
    sup_model.eval()

    # load checkpoint
    if args.checkpoint is not None:
        load_checkpoint(pretrained_model, args.checkpoint, map_location='cpu')

    with torch.no_grad():
        img = img2tensor(args.image)
        feats = random_init_model.online_backbone.extract_feat(img)

    # backbone_feat = pretrained_model.backbone.backbone(img2tensor(args.image))[-1]
    # backbone_feat = F.interpolate(backbone_feat, (4096, 4096))
    # backbone_feat = backbone_feat[0].mean(1).detach().numpy()
    # backbone_feat = (backbone_feat - backbone_feat.min()) / (backbone_feat.max() - backbone_feat.min())
    # plt.imshow(backbone_feat, cmap=cmap)
    # plt.xticks([])
    # plt.yticks([])
    # plt.tight_layout()

    # plt.figure(figsize=(15, 15))
    # plt.title('Heatmaps of FPN Features')
    for i, feat in enumerate(feats):
        if i == 3:
            break

        plt.subplot(3, 3, i+1)
        feat = F.interpolate(feat, (4096, 4096))
        feat = torchvision.transforms.functional.resize(feat, (224, 224))
        feat = feat.transpose(0, 1)
        grid = torchvision.utils.make_grid(feat, normalize=True, scale_each=True, nrow=16, pad_value=255)
        writer.add_images(f'Random_Layer{i+1}', grid.unsqueeze(1), global_step=0, dataformats='NCHW')

        # feat = feat[0][0].detach().numpy()
        # feat = (feat - feat.min()) / (feat.max() - feat.min())
        # feat[feat < args.threshold] = 0

        # plt.imshow(grid, cmap=cmap)
        # if i == 1:
        #     plt.xlabel('(a) Features of the Randomly Initialized FPN', fontsize=fontsize)
        # plt.xticks([])
        # plt.yticks([])

    with torch.no_grad():
        feats = sup_model.extract_feat(img)

    for i, feat in enumerate(feats):
        if i == 3:
            break

        # plt.subplot(3, 3, i+1+3)
        feat = F.interpolate(feat, (4096, 4096))
        feat = torchvision.transforms.functional.resize(feat, (224, 224))
        feat = feat.transpose(0, 1)
        grid = torchvision.utils.make_grid(feat, normalize=True, scale_each=True, nrow=16, pad_value=255)
        writer.add_images(f'Sup_Layer{i+1}', grid.unsqueeze(1), global_step=0, dataformats='NCHW')
        # feat = feat[0][2].detach().numpy()
        # feat = (feat - feat.min()) / (feat.max() - feat.min())
        # feat[feat < args.threshold] = 0

        # plt.imshow(feat, cmap=cmap)
        # if i == 1:
        #     plt.xlabel('(b) Features of Supervised Pre-trained FPN with Ground Truth Annotations', fontsize=fontsize)
        # plt.xticks([])
        # plt.yticks([])

    with torch.no_grad():
        feats = pretrained_model.online_backbone.extract_feat(img)

    for i, feat in enumerate(feats):
        if i == 3:
            break

        if i == 2:
            import pdb; pdb.set_trace()

        # plt.subplot(3, 3, i+1+3+3)
        feat = F.interpolate(feat, (4096, 4096))  # (1, num_channels, H, W)
        feat = torchvision.transforms.functional.resize(feat, (224, 224))
        feat = feat.transpose(0, 1)
        grid = torchvision.utils.make_grid(feat, normalize=True, scale_each=True, nrow=16, pad_value=255)
        writer.add_images(f'Our_Layer{i+1}', grid.unsqueeze(1), global_step=0, dataformats='NCHW')
        # grid = torchvision.utils.make_grid(feat)
        # feat = feat[0][2].detach().numpy()
        # feat = (feat - feat.min()) / (feat.max() - feat.min())
        # feat[feat < args.threshold] = 0

        # plt.imshow(feat, cmap=cmap)
        # if i == 1:
        #     plt.xlabel('(c) Features of Our Unsupervised Pre-trained FPN', fontsize=fontsize)
        # plt.xticks([])
        # plt.yticks([])

    # plt.tight_layout()
    # plt.savefig(f"pretrained_fpn.png")
    # plt.savefig(f"pretrained_fpn.pdf")

if __name__ == '__main__':
    main()
