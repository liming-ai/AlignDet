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

from mmdet.models.builder import DETECTORS
from mmdet.models import HybridTaskCascade


@DETECTORS.register_module()
class SelfSupHTC(HybridTaskCascade):
    def extract_feat(self, img):
        with torch.no_grad():
            x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        device = x[0].device
        instance_gt_labels = [torch.arange(len(x)).to(device) for x in gt_labels]
        for i in range(1, len(instance_gt_labels)):
            instance_gt_labels[i] += instance_gt_labels[i-1].max() + 1

        return_data = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, instance_gt_labels, gt_bboxes_ignore, gt_masks,**kwargs)

        for i in range(len(return_data)):
            return_data[i].append(losses)

        return return_data