import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .utils import delta2bbox, bbox_target, CascadeHeadConvModule
from .loss import accuracy

from ModelCore.layers import smooth_l1_loss
from ModelCore.layers import nms as _box_nms




class BBoxHead(nn.Module):
    def __init__(
            self,
            cfg,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
    ):
        super(BBoxHead, self).__init__()
        assert(cfg is not None), "cfg cannot be None"
        self.cfg = cfg
        self.with_avg_pool = cfg.MODEL.CASCADE_RCNN.HEAD_WITH_AVG_POOL
        self.with_cls = cfg.MODEL.CASCADE_RCNN.HEAD_WITH_CLS
        self.with_reg = cfg.MODEL.CASCADE_RCNN.HEAD_WITH_REG
        assert self.with_cls or self.with_reg, \
            "If you don't want a class or a reg head, why do you even add this object?"
        if isinstance(cfg.MODEL.CASCADE_RCNN.POOLING_OUTSIZE, tuple):
            self.roi_feat_size = cfg.MODEL.CASCADE_RCNN.POOLING_OUTSIZE
        else:
            self.roi_feat_size = _pair(cfg.MODEL.CASCADE_RCNN.POOLING_OUTSIZE)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = cfg.MODEL.CASCADE_RCNN.POOLING_OUT_CHANNELS
        self.num_classes = cfg.MODEL.CASCADE_RCNN.NUM_CLASSES
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = cfg.MODEL.CASCADE_RCNN.HEAD_CLASS_AGNOSTIC

        self.nms_thr = cfg.MODEL.CASCADE_RCNN.TEST_NMS_THR
        self.scr_thr = cfg.MODEL.CASCADE_RCNN.TEST_SCORE_THR
        self.max_det_num = cfg.MODEL.CASCADE_RCNN.TEST_MAX_DET_NUM
        self.default_img_shape = cfg.MODEL.CASCADE_RCNN.DEFAULT_IMG_SHAPE

        self.fc_out_channels = cfg.MODEL.CASCADE_RCNN.HEAD_FC_OUT_CHANNELS
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        ## two shared fc layers
        self.shared_fcs = nn.ModuleList()
        self.shared_fcs.append(
            nn.Linear(in_channels, self.fc_out_channels),
        )
        self.shared_fcs.append(
            nn.Linear(self.fc_out_channels, self.fc_out_channels)
        )

        if self.with_cls:
            self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = nn.Linear(self.fc_out_channels, out_dim_reg)

        self.relu = nn.ReLU(inplace=True)

        ## initialize weights
        self.init_weights()

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
        for m in self.shared_fcs:
            if isinstance(m , nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes

        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def loss(
            self,
            cls_score,
            bbox_pred,
            labels,
            bbox_targets,
    ):
        losses = dict()

        if cls_score is not None:
            cls_loss = F.cross_entropy(cls_score, labels)
            losses['loss_cls'] = cls_loss
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                # make the tensor [N, 4*num_classes] to [pos_inds.numel(), 4]
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds, labels[pos_inds]]

            bbox_loss = smooth_l1_loss(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                size_average=False,
            )
            bbox_loss = bbox_loss / labels.numel()
            losses['loss_reg'] = bbox_loss
        return losses



    def get_det_bboxes(
            self,
            rois,
            cls_score,
            bbox_pred,
            scale_factor=None,
            rescale=False,
            img_shape=None
    ):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if img_shape is None:
            img_shape = self.default_img_shape

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale and scale_factor is not None:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)


        device = bboxes.device

        dets, labels = [],  []
        inds_all = scores > self.scr_thr
        for j in range(1, self.num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = bboxes[inds, j * 4 : (j + 1) * 4]
            keep = _box_nms(boxes_j, scores_j, self.nms_thr)
            if self.max_det_num > 0:
                keep = keep[: self.max_det_num]
            dets_j = torch.cat([boxes_j[keep, :4], scores_j[keep, None]], dim=1)
            dets.append(dets_j)

            num_labels = keep.size(0)
            dets_labels = torch.full((num_labels,), j, dtype=torch.int64, device=device)
            labels.append(dets_labels)

        if dets:
            dets = torch.cat(dets)
            labels = torch.cat(labels)
            if self.max_det_num > 0 and dets.shape[0] > self.max_det_num:
                _, inds = dets[:, -1].sort(descending=True)
                inds = inds[:self.max_det_num]
                dets = dets[inds]
                labels = labels[inds]
        else:
            dets = bboxes.new_zeros((0,5))
            labels = bboxes.new_zeros((0, ), dtype=torch.long)

        return dets, labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_shape=None):
        img_ids = rois[:, 0].long().unique(sorted=True)
        if img_shape is not None:
            ## I simply assume all images in a batch have same shapes
            assert(type(img_shape) in (tuple, list,) and len(img_shape) == 2)
        else:
            img_shape = self.default_img_shape
        bboxes_list = []
        for i in range(img_ids.numel()):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()
            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_,
                                           bbox_pred_, img_shape)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])
        return bboxes_list




    def regress_by_class(self, rois, label, bbox_pred, img_shape=None):
        assert rois.size(1) == 4 or rois.size(1) == 5
        img_shape = img_shape if img_shape is not None else self.default_img_shape
        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_shape)
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)
        return new_rois



class ConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
       separated branches.
                                   /-> cls convs -> cls fcs -> cls
       shared convs -> shared fcs
                                   \-> reg convs -> reg fcs -> reg
       """
    def __init__(
            self,
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            conv_out_channels=256,
            fc_out_channels=1024,
            use_gn=True,
            *args,
            **kwargs
    ):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0

        #other parameters
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.use_gn = use_gn
        ## add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                is_shared=True
            )
        self.shared_out_channels = last_layer_dim


        ## add cls head
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels
            )
        
        ## add rg head
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels
            )

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)


    def _add_conv_fc_branch(
            self,
            num_branch_convs,
            num_branch_fcs,
            in_channels,
            is_shared=False
        ):

        last_layer_dim = in_channels

        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    CascadeHeadConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        use_gn=self.use_gn
                    )
                )

            last_layer_dim = self.conv_out_channels

        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared or self.num_shared_fcs == 0) and \
                not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels
                )

                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels)
                )
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_reg = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)

        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

class SharedFCBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 num_fcs=2,
                 fc_out_channels=1024,
                 *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs
        )











