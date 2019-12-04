import torch
import torch.nn.functional as F
from torch import nn



from ModelCore.layers import Conv2d, BatchNorm2d, NonLocal2D

class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)
        BFP takes multi-level features as inputs and gather them into a single one,
        then refine the gathered feature and scatter the refined results to
        multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
        https://arxiv.org/pdf/1904.02701.pdf for details.
    """
    def __init__(
            self,
            in_channels,
            num_levels,
            refine_level=2,
            refine_type=None,
    ):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']
        assert 0 <= refine_level < num_levels

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.refine_level = refine_level
        self.refine_type = refine_type

        if self.refine_type == 'conv':
            self.refine = nn.Sequential(
                Conv2d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),
            )
            for m in self.refine:
                if isinstance(m, Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
            )

    def forward(self, inputs):
        assert len(inputs) == self.num_levels, \
            "number of input features must match self.num_levels"

        # step 1: gather multi-level features by resize and average
        # C4 level size is used as the intermedia level fusion size
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size
                )
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest'
                )
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine the gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi_levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])
        return tuple(outs)

