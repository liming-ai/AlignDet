import torch
from mmdet.models import DETECTORS, SingleStageDetector


@DETECTORS.register_module()
class SelfSupFCOS(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SelfSupFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                          test_cfg, pretrained, init_cfg)

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
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        cls_score, bbox_pred, centerness = self.bbox_head.forward(x)
        return cls_score, bbox_pred, centerness