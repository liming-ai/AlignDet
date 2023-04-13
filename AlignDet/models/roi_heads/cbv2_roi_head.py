import numpy as np
import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from mmdet.models.builder import HEADS
from mmdet.models.utils.brick_wrappers import adaptive_avg_pool2d
from .htc_roi_head import SelfSupHybridTaskCascadeRoIHead


def upd_loss(losses, idx, weight):
        new_losses = dict()
        for k,v in losses.items():
            new_k = '{}{}'.format(k,idx)
            if weight != 1 and 'loss' in k:
                new_k = '{}_w{}'.format(new_k, weight)
            if isinstance(v,list) or isinstance(v,tuple):
                new_losses[new_k] = [i*weight for i in v]
            else:new_losses[new_k] = v*weight
        return new_losses

@HEADS.register_module()
class SelfSupCBv2Head(SelfSupHybridTaskCascadeRoIHead):

    @force_fp32(apply_to=('online_output'))
    def loss(self, online_output, target_output, online_info, target_info):
        losses = dict()
        num_backbones = len(online_output)

        assert num_backbones > 1
        loss_weights = [0.5] + [1] * (num_backbones - 1)

        for i in range(num_backbones):
            loss = self.loss_single(online_output[i], target_output[i], online_info, target_info)
            loss = upd_loss(loss, i, loss_weights[i])
            losses.update(loss)

        return losses


    @force_fp32(apply_to=('online_output'))
    def loss_single(self, online_output, target_output, online_info, target_info):
        losses = dict()

        for stage in range(len(online_output)):
            lw = self.stage_loss_weights[stage]
            if self.with_bbox:
                loss_bbox = self.bbox_head[stage].loss(
                    online_output[stage],
                    target_output[stage],
                    online_info,
                    target_info)

                for name, value in loss_bbox.items():
                    if 'rpn' not in name:
                        losses[f's{stage}.{name}'] = (
                            value * lw if 'loss' in name else value)

        for name, value in loss_bbox.items():
            if 'rpn' in name:
                losses[name] = value

        return losses