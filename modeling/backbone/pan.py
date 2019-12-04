import torch
import torch.nn.functional as F
from torch import nn

from .fpn import LastLevelMaxPool, LastLevelP6P7

## path aggregation net backbone


class PAN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(PAN, self).__init__()
        self.inner_blocks = []
        self.mid_blocks= []
        self.outer1st_blocks = []
        self.outer2nd_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "pa_inner{}".format(idx)
            mid_block = "pa_mid{}".format(idx)

            if in_channels == 0:
                continue

            if idx < len(in_channels_list):
                outer1st_block = "pa_outer1st{}".format(idx)
                outer2nd_block = "pa_outer2nd{}".format(idx)
                outer1st_block_module = conv_block(out_channels, out_channels, 3, 2)
                outer2nd_block_module = conv_block(out_channels, out_channels, 3, 1)
                self.add_module(outer1st_block, outer1st_block_module)
                self.add_module(outer2nd_block, outer2nd_block_module)
                self.outer1st_blocks.append(outer1st_block)
                self.outer2nd_blocks.append(outer2nd_block)

            inner_block_module = conv_block(in_channels, out_channels, 1)
            mid_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(mid_block, mid_block_module)
            self.inner_blocks.append(inner_block)
            self.mid_blocks.append(mid_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.mid_blocks[-1])(last_inner))
        for feature, inner_block, mid_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.mid_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, mid_block)(last_inner))

        pa_results = []
        low_feature = results[0]
        pa_results.append(low_feature)
        for mid_feature, outer1st_conv, outer2nd_conv in zip(results[1:], self.outer1st_blocks, self.outer2nd_blocks):
            reduced_feature = getattr(self, outer1st_conv)(low_feature)
            lateral_opt = reduced_feature + mid_feature
            outer_feature = getattr(self, outer2nd_conv)(lateral_opt)
            pa_results.append(outer_feature)
            low_feature = outer_feature

        results = pa_results

        if isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)



