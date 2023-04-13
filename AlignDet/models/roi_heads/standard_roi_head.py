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

from mmcv.runner import force_fp32
from mmdet.core import bbox2roi
from mmdet.models import HEADS, StandardRoIHead


@HEADS.register_module()
class SelfSupStandardRoIHead(StandardRoIHead):
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        return bbox_results, bbox_targets, rois

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward
        # bbox_results: cls_score, bbox_pred, bbox_feats
        bbox_results, bbox_targets, rois = self._bbox_forward_train(
            x, sampling_results, gt_bboxes, gt_labels, img_metas)

        return bbox_results, bbox_targets, rois

    @force_fp32(apply_to=('online_output'))
    def loss(self, online_output, target_output, online_info, target_info):
        online_bbox_results, online_bbox_targets, online_rois = online_output
        target_bbox_results, target_bbox_targets, _ = target_output

        # bbox_results: dict
        # dict_keys(['cls_score', 'bbox_pred', 'bbox_feats'])

        # bbox_targets: tuple
        # (labels, label_weights, bbox_targets, bbox_weights)

        losses = dict()
        if self.with_bbox:
            loss_bbox = self.bbox_head.loss(
                online_bbox_results,
                target_bbox_results,
                online_bbox_targets,
                target_bbox_targets,
                online_rois)
            losses.update(loss_bbox)

        return losses