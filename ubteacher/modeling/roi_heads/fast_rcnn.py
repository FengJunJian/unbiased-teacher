# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    #FastRCNNOutputs,
)

# from torchvision.ops import sigmoid_focal_loss
from detectron2.layers import  cat, cross_entropy
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats

# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.FC_loss = FocalLoss(
            gamma=1.5,
            num_classes=self.num_classes,
        )
        #self.FC_loss=sigmoid_focal_loss

    # def losses1(self, predictions, proposals):
    #     """
    #     Args:
    #         predictions: return values of :meth:`forward()`.
    #         proposals (list[Instances]): proposals that match the features
    #             that were used to compute predictions.
    #     """
    #     scores, proposal_deltas = predictions
    #     losses={
    #         "loss_cls": self.comput_focal_loss(),
    #         "loss_box_reg": self.box_reg_loss(),
    #     }
    #
    #     return losses
    def losses(self,predictions,proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        # if self.use_sigmoid_ce:
        #     loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        # else:
        loss_cls=self.comput_focal_loss(scores, gt_classes)

        # loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def comput_focal_loss(self,scores,gt_classes):
        # if self._no_instances:
        #     return 0.0 * self.pred_class_logits.sum()
        # else:
        # if gt_classes.numel() == 0 and reduction == "mean":
        #     return input.sum() * 0.0  # connect the gradient
        total_loss = self.FC_loss(input=scores, target=gt_classes)
        total_loss = total_loss / self.gt_classes.shape[0]

        return total_loss


# class FastRCNNFocalLoss(FastRCNNOutputs):
#     """
#     A class that stores information about outputs of a Fast R-CNN head.
#     It provides methods that are used to decode the outputs of a Fast R-CNN head.
#     """
#
#     def __init__(
#         self,
#         box2box_transform,
#         pred_class_logits,
#         pred_proposal_deltas,
#         proposals,
#         smooth_l1_beta=0.0,
#         box_reg_loss_type="smooth_l1",
#         num_classes=80,
#     ):
#         super(FastRCNNFocalLoss, self).__init__(
#             box2box_transform,
#             pred_class_logits,
#             pred_proposal_deltas,
#             proposals,
#             smooth_l1_beta,
#             box_reg_loss_type,
#         )
#         self.num_classes = num_classes
#         self.FC_loss = FocalLoss(
#             gamma=1.5,
#             num_classes=self.num_classes,
#         )
#
#     def losses(self):
#         return {
#             "loss_cls": self.comput_focal_loss(),
#             "loss_box_reg": self.box_reg_loss(),
#         }
#
#     def comput_focal_loss(self):
#         if self._no_instances:
#             return 0.0 * self.pred_class_logits.sum()
#         else:
#
#             total_loss = self.FC_loss(input=self.pred_class_logits, target=self.gt_classes)
#             total_loss = total_loss / self.gt_classes.shape[0]
#
#             return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()
