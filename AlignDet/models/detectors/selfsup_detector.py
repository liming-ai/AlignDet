import torch

from mmdet.models import DETECTORS, BaseDetector, build_detector


@DETECTORS.register_module()
class SelfSupDetector(BaseDetector):
    def __init__(self,
                 backbone,
                 loss_cls_weight=1.0,
                 base_momentum=0.996,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SelfSupDetector, self).__init__(init_cfg)
        assert train_cfg is None and test_cfg is None

        self.online_backbone = build_detector(backbone)
        self.target_backbone = build_detector(backbone)

        self.backbone = self.online_backbone

        for param_ol, param_tgt in zip(self.online_backbone.parameters(),
                                       self.target_backbone.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

        self.base_momentum = base_momentum
        self.momentum = base_momentum

        self.loss_cls_weight = loss_cls_weight

    @torch.no_grad()
    def momentum_update(self):
        def _ema_eman(online_module, target_module):
            online_dict = online_module.state_dict().items()
            target_dict = target_module.state_dict().items()

            for (k_o, v_o), (k_t, v_t) in zip(online_dict, target_dict):
                assert k_o == k_t, "state_dict names are different!"
                assert v_o.shape == v_t.shape, "state_dict shapes are different!"

                if 'num_batches_tracked' in k_t:
                    v_t.copy_(v_o)
                else:
                    v_t.copy_(v_t * self.momentum + (1. - self.momentum) * v_o)

        def _ema_normal(online_module, target_module):
            for (name_ol, param_ol), (_, param_tgt) in zip(online_module.named_parameters(),
                                                           target_module.named_parameters()):

                # We should keep the query embedding values of the two models the same in DETR.
                if 'query_embedding' in name_ol:
                    param_tgt.data = param_ol.data
                else:
                    param_tgt.data = param_tgt.data * self.momentum + \
                                     param_ol.data * (1. - self.momentum)

        # EMA operation
        _ema_normal(self.online_backbone, self.target_backbone)

    def forward_train(self, view1, view2):
        view1_output = self.online_backbone(**view1)
        with torch.no_grad():
            view2_output = self.target_backbone(**view2)

        if hasattr(self.backbone, 'roi_head'):
            # Two-stage
            if hasattr(self.online_backbone.roi_head, 'loss'):
                # Cascade heads
                loss = self.online_backbone.roi_head.loss(view1_output, view2_output, view1, view2)
            else:
                # Single head
                loss = self.online_backbone.roi_head.bbox_head.loss(view1_output, view2_output, view1, view2)
        else:
            # One-stage
            loss = self.online_backbone.bbox_head.loss(view1_output, view2_output, view1, view2)

        return loss

    def forward(self, data, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*data, **kwargs)
        else:
            return self.forward_test(data, **kwargs)

    def train_step(self, data, optimizer):
        # `data` here is a list instead of a dict in original mmdet
        losses = self(data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data[0]['img_metas']))

        return outputs

    def extract_feat(self, imgs):
        pass

    def simple_test(self, img, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        pass