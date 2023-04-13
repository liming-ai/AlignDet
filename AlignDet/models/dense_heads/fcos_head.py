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

import random
import torch
import torch.nn as nn

from mmdet.core import reduce_mean, build_assigner, build_sampler
from mmdet.models import HEADS, FCOSHead
from mmcv.runner import force_fp32


INF = 1e8



@HEADS.register_module()
class SelfSupFCOSHead(FCOSHead):
    def _init_predictor(self):
        ''' Original implementation
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.num_classes, 3, padding=1)
        '''
        # conv_cls is defined differ from original implementation
        # Here we follow MoCov2, using the MLP consist of [fc-relu-fc]
        self.conv_cls = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.num_base_priors * self.num_classes, 1, padding=0)
        )
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg,
                       'sampler') and self.train_cfg.sampler.type.split(
                           '.')[-1] != 'PseudoSampler':
                self.sampling = True
                sampler_cfg = self.train_cfg.sampler
            else:
                self.sampling = False
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)


    def get_cls_reg_targets(self,
                            cls_score,
                            info):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_score]

        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_score[0].dtype,
            device=cls_score[0].device)

        # each gt_bbox is a single label
        device = cls_score[0].device
        instance_gt_labels = [torch.arange(len(x)).to(device) for x in info['gt_labels']]
        for i in range(1, len(instance_gt_labels)):
            instance_gt_labels[i] += instance_gt_labels[i-1].max() + 1

        labels, bbox_targets = self.get_targets(
            all_level_points, info['gt_bboxes'], instance_gt_labels)

        return labels, bbox_targets, all_level_points


    def loss_contrastive(self,
                         cls_online,
                         cls_target,
                         online_labels,
                         target_labels,
                         instance_labels,
                         num_total_samples):

        loss = 0.
        num_valid_labels = 0

        for label in torch.unique(online_labels):
            # ignore the background class
            if label == self.num_classes:
                continue

            query_inds = (online_labels == label)
            key_inds   = (target_labels == label)
            online_neg_inds = (online_labels != label)
            target_neg_inds = (target_labels != label)

            query = cls_online[query_inds]
            key = cls_target[key_inds] if key_inds.sum() > 0 else query
            neg = torch.cat([cls_online[online_neg_inds], cls_target[target_neg_inds]])

            num_valid_labels += 1

            loss = loss + self.loss_cls(query, key, neg, avg_factor=num_total_samples)

        return loss / max(num_valid_labels, 1)


    def _sample(self, data, num):
        inds = random.sample(range(len(data)), num)
        return data[inds]


    @force_fp32(apply_to=('online_output'))
    def loss(self, online_output, target_output, online_info, target_info):
        cls_online, bbox_preds, centernesses = online_output
        cls_target = target_output[0]

        device = bbox_preds[0].device
        instance_labels = torch.arange(sum([x.size(0) for x in online_info['gt_bboxes']]))

        # generate labels for each point
        online_labels, bbox_targets, all_level_points = self.get_cls_reg_targets(cls_online, online_info)
        target_labels, _, _= self.get_cls_reg_targets(cls_target, target_info)

        num_imgs = cls_online[0].size(0)

        # flatten cls_scores, bbox_preds and centerness
        online_flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_online
        ]
        target_flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_target
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        online_flatten_cls_scores = torch.cat(online_flatten_cls_scores)
        target_flatten_cls_scores = torch.cat(target_flatten_cls_scores)
        online_flatten_labels = torch.cat(online_labels)
        target_flatten_labels = torch.cat(target_labels)

        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_bbox_targets = torch.cat(bbox_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        online_pos_inds = ((online_flatten_labels >= 0) & (online_flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        target_pos_inds = ((target_flatten_labels >= 0) & (target_flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        online_neg_inds = (online_flatten_labels == bg_class_ind).nonzero().reshape(-1)
        target_neg_inds = (target_flatten_labels == bg_class_ind).nonzero().reshape(-1)

        # compute max number of samples of pos and neg for each view
        online_num_pos = torch.tensor(len(online_pos_inds), dtype=torch.float, device=device)
        target_num_pos = torch.tensor(len(target_pos_inds), dtype=torch.float, device=device)
        online_num_pos = max(reduce_mean(online_num_pos), 1.0)
        target_num_pos = max(reduce_mean(target_num_pos), 1.0)

        # sampling
        expected_num_pos = int(self.sampler.num * self.sampler.pos_fraction)
        online_num_pos = min(online_pos_inds.numel(), expected_num_pos)
        target_num_pos = min(target_pos_inds.numel(), expected_num_pos)
        online_num_neg = self.sampler.num - online_num_pos
        target_num_neg = self.sampler.num - target_num_pos

        online_pos_inds = self._sample(online_pos_inds, online_num_pos)
        target_pos_inds = self._sample(target_pos_inds, target_num_pos)
        online_neg_inds = self._sample(online_neg_inds, online_num_neg)
        target_neg_inds = self._sample(target_neg_inds, target_num_neg)

        # query
        online_inds = torch.cat([online_pos_inds, online_neg_inds])
        target_inds = torch.cat([target_pos_inds, target_neg_inds])

        # contrastive loss
        loss_cls = self.loss_contrastive(
            online_flatten_cls_scores[online_inds],
            target_flatten_cls_scores[target_inds],
            online_flatten_labels[online_inds],
            target_flatten_labels[target_inds],
            instance_labels=instance_labels,
            num_total_samples=online_num_pos
        )

        # regression loss
        pos_bbox_preds = flatten_bbox_preds[online_pos_inds]
        pos_centerness = flatten_centerness[online_pos_inds]
        pos_bbox_targets = flatten_bbox_targets[online_pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(online_pos_inds) > 0:
            pos_points = flatten_points[online_pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=online_num_pos)
        else:
            loss_cls = online_flatten_cls_scores[online_pos_inds].sum()
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)