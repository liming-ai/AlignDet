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
import torch.nn.functional as F

from mmdet.models import LOSSES, CrossEntropyLoss

@LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.criterion = CrossEntropyLoss()

    def forward(self, query, key, negatives, reduction_override=None, avg_factor=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        num_query = query.size(0)
        num_key = key.size(0)

        # query: (m, feat_dim)
        # key  : (n, feat_dim)
        # neg  : (k, feat_dim)
        query = F.normalize(query)
        key = F.normalize(key)
        neg = F.normalize(negatives.detach())

        key = key.unsqueeze_(1)                              # (n, feat_dim) => (n, 1, feat_dim)
        neg = neg.unsqueeze_(0).expand(num_key, -1, -1)      # (k, feat_dim) => (1, k, feat_dim) => (n, k, feat_dim)
        feats = torch.cat([key, neg], dim=1)                 # (n, 1, feat_dim) + (n, k, feat_dim)  => (n, 1+k, feat_dim)

        query = query.unsqueeze(0).expand(num_key, -1, -1)   # (m, feat_dim) => (n, m, feat_dim)
        logits = torch.bmm(query, feats.permute(0, 2, 1))    # (n, m, feat_dim) @ (n, feat_dim, 1+k) => (n, m, 1+k)
        logits = logits.reshape(num_query*num_key, -1)       # (n, m, 1+k) => (n*m, 1+k)
        logits = logits / self.temperature

        labels = torch.zeros((num_query*num_key, ), dtype=torch.long).to(query.device)

        return self.criterion(logits, labels) * self.loss_weight

@LOSSES.register_module()
class NegativeCosineSimilarityLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, neg):
        target = target.detach()

        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = -(pred_norm * target_norm).sum(dim=1).mean()
        return loss * self.loss_weight