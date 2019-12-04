
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F

from ModelCore.structures.bounding_box import BoxList
from ModelCore.structures.boxlist_ops import boxlist_nms
from ModelCore.structures.boxlist_ops import cat_boxlist
from ModelCore.modeling.box_coder import BoxCoder


class PostProcessor(torch.nn.Module):
#class PostProcessor(torch.jit.ScriptModule):

    #__constants__ = ['detections_per_img']

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):

        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

        #self.onnx_export = False

    #def prepare_onnx_export(self):
    #    self.onnx_export = True
    """
    @torch.jit.script_method
    def detections_to_keep(self, scores):
        number_of_detections = scores.size(0)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            image_thresh, _ = torch.kthvalue(
                scores, number_of_detections - self.detections_per_img + 1
            )
            keep = scores >= image_thresh  # remove for jit compat... .item()
            # keep = torch.nonzero(keep).squeeze(1)
        else:
            keep = torch.ones(scores.shape, device=scores.device, dtype=torch.uint8)
        return keep

    def detections_to_keep_onnx(self, scores):
        from torch.onnx import operators
        number_of_detections = operators.shape_as_tensor(scores)
        number_to_keep = torch.min(
            torch.cat(
                (torch.tensor([self.detections_per_img], dtype=torch.long),
                 number_of_detections), 0))

        _, keep = torch.topk(
            scores, number_to_keep, dim=0, sorted=True)

        return keep
    """
    def forward(self, x, boxes):

        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        #if self.onnx_export:
        #    assert len(boxes_per_image) == 1, "ONNX exporting only supports batch size 1 now"
        #    proposals = (proposals, )
        #    class_prob = (class_prob, )
        #else:
        #    proposals = proposals.split(boxes_per_image, dim=0)
        #    class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):

        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            #boxlist_for_class.add_field(
                # we use full_like to allow tracing with flexible shape
            #    "labels", torch.full_like(boxlist_for_class.bbox[:, 0], j, dtype=torch.int64)
            #)
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]

        ## this part is changed for onnx export
        #scores = result.get_field('scores')
        #if self.onnx_export:
        #    keep = self.detections_to_keep_onnx(scores)
        #else:
        #    keep = self.detections_to_keep(scores)
        #result = result[keep]


        return result


        #number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        #if number_of_detections > self.detections_per_img > 0:
        #    cls_scores = result.get_field("scores")
        #    image_thresh, _ = torch.kthvalue(
        #        cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
         #   )
         #   keep = cls_scores >= image_thresh.item()
        #    keep = torch.nonzero(keep).squeeze(1)
        #    result = result[keep]
        #return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled
    )
    return postprocessor


