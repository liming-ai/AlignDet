# Copyright 2023 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from re import M
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmdet.models import HEADS, DETRHead, build_loss, AnchorFreeHead
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

from mmdet.core import (bbox_cxcywh_to_xyxy, build_assigner, build_sampler, multi_apply, reduce_mean, bbox_xyxy_to_cxcywh)
from mmdet.models.utils import build_transformer


@HEADS.register_module()
class SelfSupDETRHead(DETRHead):
    def __init__(self,
                 *args,
                 query_init_path=None,
                 obj_assigner=None,
                 loss_contrastive=dict(
                    type='ContrastiveLoss',
                    loss_weight=1.0,
                    temperature=0.5),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.query_init_path = query_init_path
        self.loss_contrastive = build_loss(loss_contrastive)

        if obj_assigner is not None:
            self.obj_assigner = build_assigner(obj_assigner)
        else:
            self.obj_assigner = None

        obj_sampler_cfg = dict(type='PseudoSampler')
        self.obj_sampler = build_sampler(obj_sampler_cfg, context=self)

    def _init_layers(self):
        super()._init_layers()
        self.projector = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        super().init_weights()

        if self.query_init_path is not None:
            device = self.query_embedding.weight.device
            centers = np.load(self.query_init_path)
            centers = torch.from_numpy(centers).to(device)
            self.query_embedding.weight.data = centers

    def forward_single(self, x, img_metas):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight, pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(self.reg_ffn(outs_dec))).sigmoid()
        all_proj_feats = self.projector(outs_dec)
        return all_cls_scores, all_bbox_preds, all_proj_feats

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    ins_labels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (obj_labels_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list, ins_labels_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (obj_labels_list, labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           ins_labels,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        # category-level assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # object-level assigner and sampler
        obj_assign_result = self.obj_assigner.assign(
            bbox_pred, cls_score, gt_bboxes, ins_labels, img_meta, gt_bboxes_ignore)
        obj_sampling_result = self.obj_sampler.sample(
            obj_assign_result, bbox_pred, gt_bboxes)
        obj_pos_inds = obj_sampling_result.pos_inds

        # object label targets
        obj_labels = gt_bboxes.new_full((num_bboxes, ),
                                        self.num_classes,
                                        dtype=torch.long)
        obj_labels[obj_pos_inds] = ins_labels[obj_sampling_result.pos_assigned_gt_inds]

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (obj_labels, labels, label_weights,
                bbox_targets, bbox_weights, pos_inds, neg_inds)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss_with_assigned_labels(
        self,
        all_cls_scores_list,
        all_bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore=None
    ):
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, labels_lists = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict, labels_lists

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        # each gt_bbox is a single label
        device = cls_scores[0].device
        ins_labels_list = [torch.arange(len(x)).to(device) for x in gt_labels_list]
        for i in range(1, len(gt_labels_list)):
            ins_labels_list[i] += ins_labels_list[i-1].max() + 1

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, ins_labels_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (obj_labels_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, torch.cat(obj_labels_list)


    def obj_loss(self,
                 online_feats,
                 target_feats,
                 online_ins_labels,
                 target_ins_labels):

        online_feats = online_feats[0].reshape(-1, self.embed_dims)
        target_feats = target_feats[0].reshape(-1, self.embed_dims)
        online_ins_labels = torch.cat(online_ins_labels)
        target_ins_labels = torch.cat(target_ins_labels)

        loss = 0.
        num_valid_labels = 0

        for label in torch.unique(online_ins_labels):
            # ignore the background class
            if label == self.num_classes:
                continue

            query_inds = (online_ins_labels == label)
            key_inds   = (target_ins_labels == label)
            online_neg_inds = (online_ins_labels != label)
            target_neg_inds = (target_ins_labels != label)

            num_valid_labels += 1

            query = online_feats[query_inds]
            key = target_feats[key_inds] if key_inds.sum() > 0 else query
            neg = torch.cat([online_feats[online_neg_inds], target_feats[target_neg_inds]])

            loss = loss + self.loss_contrastive(query, key, neg)

        return loss / num_valid_labels


    @force_fp32(apply_to=('online_cls_scores_list', 'online_bbox_preds_list'))
    def loss(self, online_output, target_output, online_info, target_info):
        online_cls_scores_list, online_bbox_preds_list = online_output[:2]

        loss_dict, online_obj_labels_lists = self.loss_with_assigned_labels(
            online_cls_scores_list,
            online_bbox_preds_list,
            online_info['gt_bboxes'],
            online_info['gt_labels'],
            online_info['img_metas'],
            online_info['gt_bboxes_ignore'] if 'gt_bboxes_ignore' in online_info.keys() else None
        )

        _, target_obj_labels_lists = self.loss_with_assigned_labels(
            target_output[0],
            target_output[1],
            target_info['gt_bboxes'],
            target_info['gt_labels'],
            target_info['img_metas'],
            target_info['gt_bboxes_ignore'] if 'gt_bboxes_ignore' in target_info.keys() else None
        )

        online_proj_feats = online_output[-1]
        target_proj_feats = target_output[-1]

        loss_contrastive = self.obj_loss(
            online_proj_feats, target_proj_feats,
            online_obj_labels_lists, target_obj_labels_lists
        )

        loss_dict['loss_contrastive'] = loss_contrastive

        return loss_dict