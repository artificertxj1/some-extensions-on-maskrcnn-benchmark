import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelCore.utils.registry import Registry




hrnet_w40_fpn = dict(
    IN_CHANNELS=(40, 80, 160, 320),
    OUT_CHANNELS=256,
    POOLING="AVG",
    NUM_OUTS=5,
    SHARING_CONV=False,
    ACTIVATION=False,
)
hrnet_w32_fpn = dict(
    IN_CHANNELS=(32, 64, 128, 256),
    OUT_CHANNELS=256,
    POOLING="AVG",
    NUM_OUTS=5,
    SHARING_CONV=False,
    ACTIVATION=False,
)

hrnet_w18_fpn = dict(
    IN_CHANNELS=(18, 36, 72, 144),
    OUT_CHANNELS=256,
    POOLING="AVG",
    NUM_OUTS=5,
    SHARING_CONV=False,
    ACTIVATION=False,
)


_FPN_SPECS = Registry({
    "HRNET-W40-FPN": hrnet_w40_fpn,
    "HRNET-W18-FPN": hrnet_w18_fpn,
    "HRNET-W32-FPN": hrnet_w32_fpn,
    "HRNET-W40-LIBRA": hrnet_w40_fpn,
    "HRNET-W18-LIBRA": hrnet_w18_fpn,
    "HRNET-W32-LIBRA": hrnet_w32_fpn,
})

class HRFPN(nn.Module):
    def __init__(self, cfg):
        super(HRFPN, self).__init__()

        specs = _FPN_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        self.pooling_type = specs["POOLING"]
        self.num_outs = specs["NUM_OUTS"]
        self.in_channels = specs["IN_CHANNELS"]
        self.out_channels = specs["OUT_CHANNELS"]
        self.num_ins = len(self.in_channels)
        assert isinstance(self.in_channels, (list, tuple))

        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(self.in_channels),
                      out_channels=self.out_channels,
                      kernel_size=1),
        )
        self.fpn_conv = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_conv.append(nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1
            ))
        if self.pooling_type == "MAX":
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []
        for i in range(self.num_outs):
            outputs.append(self.fpn_conv[i](outs[i]))
        return tuple(outputs)
