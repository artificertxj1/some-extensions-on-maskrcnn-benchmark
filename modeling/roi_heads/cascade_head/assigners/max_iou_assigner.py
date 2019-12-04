import torch

from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from ..geometry import bbox_overlaps

class MaxIoUAssigner(BaseAssigner):
    def __init__(
            self,
            pos_iou_thr,
            neg_iou_thr,
            min_pos_iou=.0,
            gt_max_assign_all=True
    ):
        assert neg_iou_thr <= pos_iou_thr
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all

    def assign(self, proposal_bboxes, gt_bboxes, gt_labels=None):
        """
        this method assigns a gt bbox to every proposal,
        each proposal will be assigned with -1, 0, or a positive number,
        -1: ignore proposal
        0: negative proposal (background)
        positive: positive proposal
        assign proposals which have max IoU < neg_thr to 0
        else to positive index
        :param proposals: BoxList
        :param targets: BoxList
        :return:
        """
        if proposal_bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError("No gt or proposals")
        overlaps = bbox_overlaps(gt_bboxes, proposal_bboxes)
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result


    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """
        assign proposal bboxes to gt bboxes based on iou
        :param overlaps: Tensor overlaps between k gt_bboxes and n proposals
        :param gt_labels: labels of k gt_bboxes shape(k, 1)
        :return:
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        #2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0


        #3 assign positive
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        #4 assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()

            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1
                ]
        else:
            assigned_labels = None
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)








