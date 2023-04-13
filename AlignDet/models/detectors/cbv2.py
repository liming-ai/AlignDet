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

from mmdet.models import DETECTORS, HybridTaskCascade


@DETECTORS.register_module()
class CBNetHybridTaskCascade(HybridTaskCascade):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      loss_weights=None,
                      **kwargs):
        xs = self.extract_feat(img)

        if not isinstance(xs[0], (list, tuple)):
            xs = [xs]
            loss_weights = None
        elif loss_weights is None:
            loss_weights = [0.5] + [1]*(len(xs)-1)  # Reference CBNet paper


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

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            for i,x in enumerate(xs):
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                if len(xs) > 1:
                    rpn_losses = upd_loss(rpn_losses, idx=i, weight=loss_weights[i])
                losses.update(rpn_losses)
        else:
            proposal_list = proposals

        for i,x in enumerate(xs):
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            if len(xs) > 1:
                roi_losses = upd_loss(roi_losses, idx=i, weight=loss_weights[i])
            losses.update(roi_losses)

        return losses
