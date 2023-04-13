import torch
from mmdet.models import DETECTORS, DeformableDETR, SingleStageDetector

@DETECTORS.register_module()
class SelfSupDeformableDETR(DeformableDETR):
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
        return self.bbox_head.forward(x, img_metas, gt_bboxes)