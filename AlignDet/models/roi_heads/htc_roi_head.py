# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from mmdet.models.builder import HEADS
from mmdet.models.utils.brick_wrappers import adaptive_avg_pool2d
from mmdet.models.roi_heads import HybridTaskCascadeRoIHead


@HEADS.register_module()
class SelfSupHybridTaskCascadeRoIHead(HybridTaskCascadeRoIHead):
    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage, x, rois, semantic_feat=semantic_feat)

        bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes,
                                             gt_labels, rcnn_train_cfg)

        return bbox_results, bbox_targets, rois


    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):

        return_data = []

        # No gt_mask in the pre-training
        semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results, bbox_targets, rois = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)

            roi_labels = bbox_targets[0]

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

            return_data.append([bbox_results, bbox_targets, rois])

        return return_data


    @force_fp32(apply_to=('online_output'))
    def loss(self, online_output, target_output, online_info, target_info):
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