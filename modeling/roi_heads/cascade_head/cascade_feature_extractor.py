from __future__ import division

import torch
import torch.nn as nn

from ModelCore.layers import ROIAlign

class SingleRoIExtractor(nn.Module):
    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            finest_scale=56,
    ):
        super(SingleRoIExtractor, self).__init__()
        self.scales = scales
        self.finest_scale = finest_scale
        self.output_size = output_size
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
            )
        self.poolers = nn.ModuleList(poolers)

    def num_inputs(self):
        return len(self.scales)

    def map_roi_levels(self, rois, num_levels):
        area = torch.sqrt((rois[:, 3] - rois[:, 1] + 1) * (
                rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(area / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.poolers[0](feats[0], rois)

        num_levels = len(feats)
        num_channels = feats[0].shape[1]
        output_size = self.output_size[0]

        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(
            rois.size(0), num_channels, output_size, output_size
        )
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.poolers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t

        return roi_feats

