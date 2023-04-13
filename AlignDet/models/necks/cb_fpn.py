# CBNetV2: A Composite Backbone Network Architecture for Object Detection
# https://arxiv.org/abs/2107.00420
# https://github.com/VDIGPKU/CBNetV2
from mmdet.models import NECKS
from mmdet.models import FPN


@NECKS.register_module()
class CBFPN(FPN):
    '''
    FPN with weight sharing
    which support mutliple outputs from cbnet
    '''
    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]

        if self.training:
            outs = []
            for x in inputs:
                out = super().forward(x)
                outs.append(out)
            return outs
        else:
            out = super().forward(inputs[-1])
            return out
