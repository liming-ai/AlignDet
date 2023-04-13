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

import torch
import torch.nn as nn

from mmdet.models.utils import build_linear_layer
from mmcv.runner import force_fp32
from mmdet.models import HEADS, ConvFCBBoxHead


@HEADS.register_module()
class SelfSupConvFCBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        # NOTE: We replace the original linear layer with mocov2 neck [fc-relu-fc]
        if self.with_cls:
            self.fc_cls = nn.Sequential(
                nn.Linear(self.cls_last_dim, self.cls_last_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cls_last_dim, self.num_classes)
            )
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),      # remove the cls_fcs
                        dict(name='reg_fcs')
                    ])
            ]

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, view1_output, view2_output, view1, view2):

        online_bbox_results, online_bbox_targets, online_rois, losses = view1_output
        target_bbox_results, target_bbox_targets, _, _ = view2_output

        if 'reduction_override' in view1.keys():
            reduction_override = view1['reduction_override']
        else:
            reduction_override = None

        cls_online, bbox_pred = online_bbox_results['cls_score'], online_bbox_results['bbox_pred']
        cls_target = target_bbox_results['cls_score']

        online_labels, online_label_weights, online_bbox_targets, online_bbox_weights = online_bbox_targets
        target_labels, target_label_weights, _, _ = target_bbox_targets

        avg_factor = max(torch.sum(online_label_weights > 0).float().item(), 1.)

        # loss_cls
        loss_cls_ = 0.
        num_valid_labels = 0

        for label in torch.unique(online_labels):
            # ignore the background class
            if label == self.num_classes:
                continue

            #                    label                      sample           #
            query_inds = (online_labels == label) * (online_label_weights > 0)
            key_inds   = (target_labels == label) * (target_label_weights > 0)
            online_neg_inds = (online_labels != label) * (online_label_weights > 0)
            target_neg_inds = (target_labels != label) * (target_label_weights > 0)

            num_valid_labels += 1

            query = cls_online[query_inds]
            key = cls_target[key_inds] if key_inds.sum() > 0 else query
            neg = torch.cat([cls_online[online_neg_inds], cls_target[target_neg_inds]])

            loss_cls_ = loss_cls_ + self.loss_cls(query, key, neg, avg_factor=avg_factor)

        losses['loss_cls'] = loss_cls_ / num_valid_labels

        # loss_bbox
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (online_labels >= 0) & (online_labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(online_rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           online_labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    online_bbox_targets[pos_inds.type(torch.bool)],
                    online_bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=online_bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


@HEADS.register_module()
class SelfSupShared2FCBBoxHead(SelfSupConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(SelfSupShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class SelfSupShared4Conv1FCBBoxHead(SelfSupConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(SelfSupShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
