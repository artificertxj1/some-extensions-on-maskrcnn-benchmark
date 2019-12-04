# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from ModelCore.layers import smooth_l1_loss, balanced_l1_loss
from ModelCore.modeling.box_coder import BoxCoder
from ModelCore.modeling.matcher import Matcher
from ModelCore.structures.boxlist_ops import boxlist_iou
from ModelCore.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler, IOUBalancedPositiveNegativeSampler
)
from ModelCore.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False,
        reg_loss='smooth_l1_loss',
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.reg_loss = reg_loss


    def match_targets_to_proposals(self, proposal, target):


        if len(target) > 0:
            match_quality_matrix = boxlist_iou(target, proposal)
            matched_max_ious, _ = match_quality_matrix.max(dim=0)
            #print(match_quality_matrix.shape, matched_max_ious.shape, len(proposal))
            matched_idxs = self.proposal_matcher(match_quality_matrix)

            #print(matched_max_ious.shape, matched_idxs.shape)
            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields("labels")
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            labels = target.get_field("labels")
            #print(labels.type())
            matched_targets = target[matched_idxs.clamp(min=0)]
            matched_targets.add_field("matched_idxs", matched_idxs)
            matched_targets.add_field("matched_max_ious", matched_max_ious)
        else:
            device = proposal.bbox.device
            matched_targets = proposal.copy_with_fields([])
            matched_idxs = torch.full((len(proposal), ),
                                      self.proposal_matcher.BELOW_LOW_THRESHOLD,
                                      device=device,
                                      dtype=torch.int64)
            matched_max_ious = torch.full((len(proposal), ),
                                          0.0,
                                          device=device,
                                          dtype=torch.float32)
            matched_labels = torch.full((len(proposal), ),
                                        0.,
                                        device=device,
                                        dtype=torch.float32)
            matched_targets.add_field("matched_idxs", matched_idxs)
            matched_targets.add_field("matched_max_ious", matched_max_ious)
            matched_targets.add_field("labels", matched_labels)
        return matched_targets


    def prepare_targets(self, proposals, targets):
        labels = []
        max_ious = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):

            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            #matched_targets = self.proposal_matcher.match_targets_to_proposals(
            #    proposals_per_image, targets_per_image, copied_fields=["labels"]
            #)

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            # fetch the max iou values
            max_ious.append(matched_targets.get_field("matched_max_ious"))
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets, max_ious

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        #print("Type of proposals is {}, Type of targets is {}".format(type(proposals), type(targets)))
        labels, regression_targets, max_ious = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, max_ious)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        if self.reg_loss == 'smooth_l1_loss':
            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,
                beta=1,
            )
        elif self.reg_loss == 'balanced_l1_loss':
            box_loss = balanced_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,
            )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):

    assert(cfg.MODEL.ROI_BOX_HEAD.SAMPLING_METHOD in ('random', 'iou_balanced')), \
        "ROI_BOX_HEAD.SAMPLING_METHOD can only be random or iou_balanced"



    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    if cfg.MODEL.ROI_BOX_HEAD.SAMPLING_METHOD == 'random':
        fg_bg_sampler = BalancedPositiveNegativeSampler(
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        )
    else:
        fg_bg_sampler = IOUBalancedPositiveNegativeSampler(
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
        )

    reg_loss = cfg.MODEL.ROI_BOX_HEAD.REG_LOSS
    assert(reg_loss in ('smooth_l1_loss', 'balanced_l1_loss')), \
        "reg_loss function can only be smooth_l1_loss or balanced_l1_loss"


    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg,
        reg_loss=reg_loss
    )

    return loss_evaluator
