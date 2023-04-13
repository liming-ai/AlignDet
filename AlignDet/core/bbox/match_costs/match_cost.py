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
import torch.nn.functional as F

from mmdet.core.bbox.match_costs import ClassificationCost
from mmdet.core.bbox.match_costs.builder import MATCH_COST


@MATCH_COST.register_module()
class ZeroCost(ClassificationCost):
    def __call__(self, cls_pred, gt_labels):
        return 0.


@MATCH_COST.register_module()
class DissimilarCost(ClassificationCost):
    def __call__(self, cls_feat, roi_feat, temperature=0.5):
        similarity = F.normalize(cls_feat) @ F.normalize(roi_feat).T
        similarity = (similarity / temperature).softmax(-1)
        dissimilar_cost = 1 - similarity
        return dissimilar_cost

@MATCH_COST.register_module()
class NegativeCosineSimilarityCost(ClassificationCost):
    def __call__(self, cls_feat, roi_feat):
        similarity = F.normalize(cls_feat) @ F.normalize(roi_feat).T
        return 1 - similarity