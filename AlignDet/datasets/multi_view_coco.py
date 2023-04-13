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

from copy import deepcopy
import inspect

from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS, PIPELINES, build_dataset
from mmdet.datasets.pipelines import Compose
from torchvision import transforms as _transforms


# register all existing transforms in torchvision
_INCLUDED_TRANSFORMS = ['ColorJitter']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] in _INCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@DATASETS.register_module()
class MultiViewCocoDataset:
    def __init__(self, dataset, num_views, pipelines):
        assert num_views == len(pipelines)

        self.dataset = build_dataset(dataset)

        self.CLASSES = self.dataset.CLASSES
        self.PALETTE = getattr(self.dataset, 'PALETTE', None)
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag

        # processing multi_views pipeline
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        results = self.dataset[idx]
        return list(map(lambda pipeline: pipeline(deepcopy(results)), self.pipelines))