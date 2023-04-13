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