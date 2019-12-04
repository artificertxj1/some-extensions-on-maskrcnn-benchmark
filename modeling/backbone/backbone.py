from collections import OrderedDict

from torch import nn

from ModelCore.modeling import registry
from ModelCore.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import pan as pan_module
from . import bfp as bfp_module
from . import resnet
from . import hrnet
from . import hrfpn
from . import senet




@registry.BACKBONES.register("HRNET-W18-FPN")
@registry.BACKBONES.register("HRNET-W32-FPN")
@registry.BACKBONES.register("HRNET-W40-FPN")
def build_hrnet_fpn_backbone(cfg):
    body = hrnet.HRNet(cfg)
    neck = hrfpn.HRFPN(cfg)
    model = nn.Sequential(OrderedDict([('body', body), ('neck', neck)]))
    model.out_channels = 256
    ##TODO make this available to user defined  out-channel number
    return model

@registry.BACKBONES.register("HRNET-W18-LIBRA")
@registry.BACKBONES.register("HRNET-W32-LIBRA")
@registry.BACKBONES.register("HRNET-W40-LIBRA")
def build_resnet_libra_backbone(cfg):
    body = hrnet.HRNet(cfg)
    neck = hrfpn.HRFPN(cfg)
    bfp = bfp_module.BFP(
        in_channels=256,
        num_levels= cfg.MODEL.LIBRA.NUM_LEVELS,
        refine_level=cfg.MODEL.LIBRA.REFINE_LEVEL,
        refine_type=cfg.MODEL.LIBRA.REFINE_TYPE,
    )
    model = nn.Sequential(OrderedDict([("body", body), ("neck", neck), ("bfp", bfp)]))
    model.out_channels = 256
    return model

@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("R-50-PAN")
@registry.BACKBONES.register("R-101-PAN")
@registry.BACKBONES.register("R-152-PAN")
def build_resnet_pan_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    pan = pan_module.PAN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.PAN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("pan", pan)]))
    model.out_channels = out_channels
    return model



@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model



@registry.BACKBONES.register("SE-R-50-FPN")
@registry.BACKBONES.register("SE-R-101-FPN")
@registry.BACKBONES.register("SE-X-50-FPN")
@registry.BACKBONES.register("SE-X-101-FPN")
def build_senet_fpn_backbone(cfg):
    body = senet.build_senet(cfg)

    in_channels_stage2 = cfg.MODEL.SENET.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.SENET.BACKBONE_OUT_CHANNELS

    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("SE-R-50-PAN")
@registry.BACKBONES.register("SE-R-101-PAN")
@registry.BACKBONES.register("SE-X-50-PAN")
@registry.BACKBONES.register("SE-X-101-PAN")
def build_senet_pan_backbone(cfg):
    body = senet.build_senet(cfg)
    in_channels_stage2 = cfg.MODEL.SENET.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.SENET.BACKBONE_OUT_CHANNELS
    pan = pan_module.PAN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.PAN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("pan", pan)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("R-50-LIBRA")
@registry.BACKBONES.register("R-101-LIBRA")
@registry.BACKBONES.register("R-152-LIBRA")
def build_resnet_libra_backbone(cfg):
    body = senet.build_senet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    bfp = bfp_module.BFP(
        in_channels=out_channels,
        num_levels= cfg.MODEL.LIBRA.NUM_LEVELS,
        refine_level=cfg.MODEL.LIBRA.REFINE_LEVEL,
        refine_type=cfg.MODEL.LIBRA.REFINE_TYPE,
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn), ("bfp", bfp)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("SE-R-50-LIBRA")
@registry.BACKBONES.register("SE-R-101-LIBRA")
@registry.BACKBONES.register("SE-X-50-LIBRA")
@registry.BACKBONES.register("SE-X-101-LIBRA")
def build_senet_libra_backbone(cfg):
    body = senet.build_senet(cfg)
    in_channels_stage2 = cfg.MODEL.SENET.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.SENET.BACKBONE_OUT_CHANNELS
    fpn = pan_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.PAN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    bfp = bfp_module.BFP(
        in_channels=out_channels,
        num_levels=cfg.MODEL.LIBRA.NUM_LEVELS,
        refine_level=cfg.MODEL.LIBRA.REFINE_LEVEL,
        refine_type=cfg.MODEL.LIBRA.REFINE_TYPE,
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn), ("bfp", bfp)]))
    model.out_channels = out_channels
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)