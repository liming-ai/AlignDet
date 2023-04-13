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

from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import HungarianAssigner


@BBOX_ASSIGNERS.register_module()
class PseudoHungarianAssigner(HungarianAssigner):
    def assign(self,
               bbox_pred,
               cls_pred,
               roi_feat,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):

        device = cls_pred.device
        num_gts = gt_bboxes.shape[0]
        num_queries = cls_pred.shape[0]

        assigned_gt_inds = torch.zeros(num_queries)
        assigned_gt_inds[:num_gts] = torch.arange(num_gts) + 1
        assigned_labels = assigned_gt_inds - 1

        return AssignResult(
            num_gts, assigned_gt_inds.long().to(device), None, labels=assigned_labels.to(device))