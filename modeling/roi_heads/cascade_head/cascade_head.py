from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from ModelCore.structures.bounding_box import BoxList

from .utils import bbox2roi, bbox2result
from .samplers import RandomSampler
from .assigners import MaxIoUAssigner
from .cascade_feature_extractor import SingleRoIExtractor
from .cascade_bbox_heads import BBoxHead

class CascadeHeads(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CascadeHeads, self).__init__()


        self.cfg = cfg
        ## number of cascade stages
        self.num_stages = cfg.MODEL.CASCADE_RCNN.NUM_STAGES
        ## regression weights in each stage delta2bbox
        self.stage_target_means = cfg.MODEL.CASCADE_RCNN.STAGE_TARGET_MEANS
        self.stage_target_stds  = cfg.MODEL.CASCADE_RCNN.STAGE_TARGET_STDS
        ## regression stage loss weights
        self.stage_loss_weights = cfg.MODEL.CASCADE_RCNN.STAGE_LOSS_WEIGHTS
        ## iou threshold used in each stage
        self.stage_ious = cfg.MODEL.CASCADE_RCNN.STAGE_IOUS

        assert len(self.stage_target_means) == \
               len(self.stage_target_stds) == \
               len(self.stage_loss_weights) == \
               len(self.stage_ious) == \
               self.num_stages

        ## sampler and assigner is built on-flying, save the parameters for later use
        ## sampling parameters
        self.sampling_pos_fraction = cfg.MODEL.CASCADE_RCNN.SAMPLING_POS_FRACTION
        self.sampling_num = cfg.MODEL.CASCADE_RCNN.SAMPLING_BATCH_SIZE
        self.sampling_add_gt = cfg.MODEL.CASCADE_RCNN.SAMPLING_ADD_GT
        ## box heads
        self.bbox_roi_extractors = nn.ModuleList()
        self.bbox_heads          = nn.ModuleList()

        self.scales = cfg.MODEL.CASCADE_RCNN.POOLING_SCALES

        for i in range(self.num_stages):
            self.bbox_roi_extractors.append(
                SingleRoIExtractor(
                    output_size=cfg.MODEL.CASCADE_RCNN.POOLING_OUTSIZE,
                    scales=cfg.MODEL.CASCADE_RCNN.POOLING_SCALES,
                    sampling_ratio=cfg.MODEL.CASCADE_RCNN.POOLING_SAMPLE_RATIO,
                )
            )
            self.bbox_heads.append(
                BBoxHead(
                    cfg=cfg,
                    target_means=self.stage_target_means[i],
                    target_stds=self.stage_target_stds[i],
                )
            )

    def forward(self, features, proposals, targets=None):
        proposal_list = [proposal.bbox for proposal in proposals]
        if self.training:
            return self._forward_train(features, proposal_list, targets)
        else:
            return self._forward_test(features, proposal_list, targets)


    def _forward_train(self, features, proposal_list, targets):
        """
        training forward
        :param features: list[Tensor] feature-maps from backbone
        :param proposals: list[BoxList] proposal boxes from rpn
        :param targets: list[BoxList] ground truth bboxes
        :return:
        """
        #assert targets is not None

        ## Assuming all imgs in a batch have same dimension
        img_shape = targets[0].size

        losses = dict()

        for i in range(self.num_stages):

            lw = self.stage_loss_weights[i]

            bbox_assigner = MaxIoUAssigner(
                pos_iou_thr=self.stage_ious[i],
                neg_iou_thr=self.stage_ious[i],
                min_pos_iou=self.stage_ious[i]
            )
            bbox_sampler = RandomSampler(
                num=self.sampling_num,
                pos_fraction=self.sampling_pos_fraction,
                add_gt_as_proposals=self.sampling_add_gt
            )

            num_imgs = len(proposal_list)
            sampling_results = []
            for j in range(num_imgs):
                #if targets[j] is not None:
                assign_result = bbox_assigner.assign(
                        proposal_list[j], targets[j].bbox,
                        targets[j].get_field("labels").long()
                )
                sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        targets[j].bbox,
                        targets[j].get_field("labels").long()
                )
                sampling_results.append(sampling_result)

            bbox_roi_extractor = self.bbox_roi_extractors[i]
            bbox_head = self.bbox_heads[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])


            bbox_feats = bbox_roi_extractor(features[:len(self.scales)], rois)

            cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_targets = bbox_head.get_target(sampling_results)

            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)

            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value
                )

            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts,
                        img_shape=img_shape
                    )
        return {}, proposal_list, losses

    def _forward_test(self, features, proposal_list, targets=None):

        rescale = self.cfg.MODEL.CASCADE_RCNN.TEST_RESCALE

        ## presuming that there is only one input image
        rois = bbox2roi(proposal_list)

        ## multi-stage results
        ms_scores = []

        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractors[i]
            bbox_head = self.bbox_heads[i]

            #bbox_feats = bbox_roi_extractor(features[:bbox_roi_extractor.num_inputs], rois)
            bbox_feats = bbox_roi_extractor(features[:len(self.scales)], rois)

            cls_score, bbox_pred = bbox_head(bbox_feats)

            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred)

        cls_score = sum(ms_scores) / self.num_stages

        det_bboxes, det_labels = self.bbox_heads[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            rescale=rescale
        )
        print(det_bboxes.shape)
        img_shape = self.cfg.MODEL.CASCADE_RCNN.DEFAULT_IMG_SHAPE
        bboxes = det_bboxes[:, :4]
        bbox_results = BoxList(bboxes, img_shape, mode='xyxy')
        bbox_results.add_field("labels", det_labels)
        bbox_results.add_field("scores", det_bboxes[:, 4])
        return {}, [bbox_results], {}


















        



