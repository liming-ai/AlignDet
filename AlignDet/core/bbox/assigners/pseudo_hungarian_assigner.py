import torch

from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import HungarianAssigner


@BBOX_ASSIGNERS.register_module()
class PseudoHungarianAssigner(HungarianAssigner):
    def assign(self,
               bbox_pred,
               cls_pred,
               roi_feat,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):

        device = cls_pred.device
        num_gts = gt_bboxes.shape[0]
        num_queries = cls_pred.shape[0]

        assigned_gt_inds = torch.zeros(num_queries)
        assigned_gt_inds[:num_gts] = torch.arange(num_gts) + 1
        assigned_labels = assigned_gt_inds - 1

        return AssignResult(
            num_gts, assigned_gt_inds.long().to(device), None, labels=assigned_labels.to(device))