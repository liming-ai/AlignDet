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

from torch import nn
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class ResetNormsHook(Hook):
    def before_run(self, runner):
        for m in runner.model.modules():
            if isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
                print(f'Reset running_stats and parameters for {m._get_name()}')
                m.reset_running_stats()
                m.reset_parameters()

            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                print(f'Reset parameters for {m._get_name()}')
                m.reset_parameters()