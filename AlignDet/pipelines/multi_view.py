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

from mmdet.datasets.pipelines import Compose
from mmdet.datasets import PIPELINES
from mmcv.utils import build_from_cfg

class MultiViewAug:
    def __init__(self, num_views, pipelines, keys):
        assert len(num_views) == len(pipelines) == len(keys)

        self.keys = keys
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)

        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.trans = trans

    def __call__(self, results):
        for trans, key in zip(self.trans, self.keys):
            results[key] = trans(results['img'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'