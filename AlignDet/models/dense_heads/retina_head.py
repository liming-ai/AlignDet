import torch
import torch.nn as nn

from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, RetinaHead
from mmdet.core import images_to_levels, multi_apply


@HEADS.register_module()
class SelfSupRetinaHead(RetinaHead):
    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        # self.retina_cls = nn.Conv2d(
        #     self.feat_channels,
        #     self.num_base_priors * self.num_classes,  # modify here
        #     1,
        #     padding=0)

        # following mocov2, fc-relu-fc
        self.retina_cls = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.num_base_priors * self.num_classes, 1, padding=0)
        )
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

    def get_cls_reg_targets(self, cls_scores, info, gt_bboxes_ignore=None):
        gt_bboxes, gt_labels, img_metas = info['gt_bboxes'], info['gt_labels'], info['img_metas']

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        # each gt_bbox is a single label
        device = cls_scores[0].device
        instance_gt_labels = [torch.arange(len(x)).to(device) for x in gt_labels]
        for i in range(1, len(instance_gt_labels)):
            instance_gt_labels[i] += instance_gt_labels[i-1].max() + 1

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=instance_gt_labels,
            label_channels=label_channels)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        return cls_reg_targets, all_anchor_list

    def loss_single_cls(self,
                        cls_online,
                        cls_target,
                        online_labels,
                        target_labels,
                        online_label_weights,
                        target_label_weights,
                        instance_labels,
                        num_total_samples):

        loss = 0.
        num_valid_labels = 0

        for label in torch.unique(online_labels):
            # ignore the background class
            if label == self.num_classes:
                continue

            #                     label                      sample          #
            query_inds = (online_labels == label) * (online_label_weights > 0)
            key_inds   = (target_labels == label) * (target_label_weights > 0)
            online_neg_inds = (online_labels != label) * (online_label_weights > 0)
            target_neg_inds = (target_labels != label) * (target_label_weights > 0)

            num_valid_labels += 1

            query = cls_online[query_inds]
            key = cls_target[key_inds] if key_inds.sum() > 0 else query
            neg = torch.cat([cls_online[online_neg_inds], cls_target[target_neg_inds]])

            loss = loss + self.loss_cls(query, key, neg, avg_factor=num_total_samples)

        return loss / num_valid_labels


    def loss_single_bbox(self, bbox_pred, anchors, bbox_targets, bbox_weights, num_total_samples):
        # regression loss for a single level
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        return loss_bbox, None


    @force_fp32(apply_to=('bbox_preds'))
    def loss(self, online_output, target_output, online_info, target_info):
        cls_online, bbox_preds = online_output
        cls_target, _ = target_output

        # extract cls and reg targets for each feature level
        online_targets, all_anchor_list = self.get_cls_reg_targets(cls_online, online_info)
        target_targets, _ = self.get_cls_reg_targets(cls_target, target_info)
        if online_targets is None:
            return None

        # generate online labels
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = online_targets
        target_labels_list, target_label_weights_list = target_targets[:2]

        # each instance bbox is considered as a class
        instance_labels = torch.arange(sum([x.size(0) for x in online_info['gt_bboxes']]))

        all_online_labels = torch.cat(labels_list, dim=1).reshape(-1)
        all_target_labels = torch.cat(target_labels_list, dim=1).reshape(-1)
        all_online_label_weights = torch.cat(label_weights_list, dim=1).reshape(-1)
        all_target_label_weights = torch.cat(target_label_weights_list, dim=1).reshape(-1)

        # TODO: GPU OOM
        cls_online = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in cls_online])
        cls_target = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in cls_target])

        loss_cls = self.loss_single_cls(
            cls_online,
            cls_target,
            all_online_labels,
            all_target_labels,
            all_online_label_weights,
            all_target_label_weights,
            instance_labels=instance_labels,
            num_total_samples=num_total_pos
        )

        losses_bboxes, _ = multi_apply(
            self.loss_single_bbox,
            bbox_preds,
            all_anchor_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_pos
        )

        return dict(
            loss_cls=loss_cls,
            losses_bboxes=losses_bboxes)