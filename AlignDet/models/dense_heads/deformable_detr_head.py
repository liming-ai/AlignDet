import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmcv.cnn import Linear, constant_init

from mmdet.core import multi_apply, bbox2roi
from mmdet.models import HEADS, build_roi_extractor
from mmdet.models.utils.transformer import inverse_sigmoid

from .detr_head import SelfSupDETRHead


@HEADS.register_module()
class SelfSupDeformableDETRHead(SelfSupDETRHead):
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[8, 16, 32, 64]),
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        super(SelfSupDeformableDETRHead, self).__init__(
            *args, transformer=transformer, **kwargs)

        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        # NOTE: Here we follow MoCov2, using the MLP consist of [fc-relu-fc]
        fc_cls = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.num_classes),
        )

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)


    def init_weights(self):
        self.transformer.init_weights()
        """We use contrastive loss during training"""
        # if self.loss_cls.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.cls_branches:
        #         nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas, gt_bboxes):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        # NOTE: extract roi_align features to calculate unsupervised cls_cost
        with torch.no_grad():
            rois = bbox2roi([box for box in gt_bboxes])
            roi_feats = self.bbox_roi_extractor(mlvl_feats, rois)
            roi_feats = F.adaptive_avg_pool2d(roi_feats, (1, 1)).squeeze(-1).squeeze(-1)

            roi_feats_list = [roi_feats.split([x.shape[0] for x in gt_bboxes], dim=0)][0]
            roi_feats_list = [x for x in roi_feats_list]

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        # Using roi_align feats and coordinates to initialize the query
        # This help to alignment the feature spaces of query and feats
        # Please refer to https://arxiv.org/abs/2203.06883 for more details
        num_gts = rois.shape[0]
        roi_query = self.roi_feat_to_embed(roi_feats)

        selfsup_query_embeds = torch.cat([
            roi_query, query_embeds[num_gts:]
        ], dim=0)

        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    selfsup_query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                enc_outputs_class, \
                enc_outputs_coord.sigmoid(), roi_feats_list
        else:
            return outputs_classes, outputs_coords, \
                None, None, roi_feats_list

    @force_fp32(apply_to=('bbox_preds'))
    def loss(self, online_output, target_output, online_info, target_info):
        cls_online, bbox_preds, enc_cls_scores, enc_bbox_preds, online_roi_feats_list = online_output
        cls_target, _, _, _, target_roi_feats_list = target_output

        gt_bboxes_list, gt_labels_list = online_info['gt_bboxes'], online_info['gt_labels']
        gt_bboxes_ignore = online_info['gt_bboxes_ignore'] if 'gt_bboxes_ignore' in online_info.keys() else None

        # each gt_bbox is a single label
        device = cls_online[0].device
        gt_labels_list = [torch.arange(len(x)).to(device) for x in gt_labels_list]
        for i in range(1, len(gt_labels_list)):
            gt_labels_list[i] += gt_labels_list[i-1].max() + 1

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(cls_online)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [online_info['img_metas'] for _ in range(num_dec_layers)]

        # RoIAlign features
        all_roi_feats_list = [online_roi_feats_list for _ in range(num_dec_layers)]

        online_labels, online_label_weights = multi_apply(
            self.get_cls_targets,
            cls_online,
            bbox_preds,
            all_roi_feats_list,
            all_gt_bboxes_list,
            all_gt_labels_list,
            img_metas_list,
            all_gt_bboxes_ignore_list
        )

        target_labels, target_label_weights = multi_apply(
            self.get_cls_targets,
            target_output[0],
            target_output[1],
            [target_roi_feats_list for _ in range(num_dec_layers)],
            all_gt_bboxes_list,
            all_gt_labels_list,
            [target_info['img_metas'] for _ in range(num_dec_layers)],
            all_gt_bboxes_ignore_list
        )

        loss_contrastive = self.loss_contrastive(
            cls_online, online_labels, online_label_weights,
            cls_target, target_labels, target_label_weights,
            gt_labels_list
        )

        losses_bbox, losses_iou = multi_apply(
            self.loss_single, cls_online, bbox_preds, all_roi_feats_list,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(online_info['img_metas']))
            ]
            enc_loss_contrastive = self.loss_contrastive(
                online_output[-2], online_labels, online_label_weights,
                target_output[-2], target_labels, target_label_weights)
            enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 online_info['img_metas'], gt_bboxes_ignore)

            loss_dict['enc_loss_contrastive'] = enc_loss_contrastive
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        # loss from the last decoder layer
        loss_dict['loss_contrastive'] = loss_contrastive
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_bbox_i, loss_iou_i in zip(losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict


    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   img_metas,
                   rescale=False):
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)
        return result_list