# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.models.necks.ssd_neck import L2Norm

@MODELS.register_module()
class SSDFPNNeck(BaseModule):
    """Extra layers of SSD backbone to generate multi-scale feature maps.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (Sequence[int]): Number of output channels per scale.
        level_strides (Sequence[int]): Stride of 3x3 conv per level.
        level_paddings (Sequence[int]): Padding size of 3x3 conv per level.
        l2_norm_scale (float|None): L2 normalization layer init scale.
            If None, not use L2 normalization on the first input feature.
        last_kernel_size (int): Kernel size of the last conv layer.
            Default: 3.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 level_strides,
                 level_paddings,
                 l2_norm_scale=20.,
                 last_kernel_size=3,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=[
                     dict(
                         type='Xavier', distribution='uniform',
                         layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                 ]):
        super(SSDFPNNeck, self).__init__(init_cfg)
        assert len(out_channels) > len(in_channels)
        assert len(out_channels) - len(in_channels) == len(level_strides)
        assert len(level_strides) == len(level_paddings)
        assert in_channels == out_channels[:len(in_channels)]

        if l2_norm_scale:
            self.l2_norm = L2Norm(in_channels[0], l2_norm_scale)
            self.init_cfg += [
                dict(
                    type='Constant',
                    val=self.l2_norm.scale,
                    override=dict(name='l2_norm'))
            ]

        self.extra_layers = nn.ModuleList()
        extra_layer_channels = out_channels[len(in_channels):]
        second_conv = DepthwiseSeparableConvModule if \
            use_depthwise else ConvModule

        for i, (out_channel, stride, padding) in enumerate(
                zip(extra_layer_channels, level_strides, level_paddings)):
            kernel_size = last_kernel_size \
                if i == len(extra_layer_channels) - 1 else 3
            per_lvl_convs = nn.Sequential(
                ConvModule(
                    out_channels[len(in_channels) - 1 + i],
                    out_channel // 2,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                second_conv(
                    out_channel // 2,
                    out_channel,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.extra_layers.append(per_lvl_convs)

        self.target_channels = 256
        self.channel_reduction_convs = nn.ModuleList(
            [nn.Conv2d(c, self.target_channels, kernel_size=1) for c in out_channels if c != self.target_channels]
        )
    
        

    def forward(self, inputs):
        """Forward function."""
        outs = [feat for feat in inputs]
        if hasattr(self, 'l2_norm'):
            outs[0] = self.l2_norm(outs[0])

        feat = outs[-1]
        for layer in self.extra_layers:
            feat = layer(feat)
            outs.append(feat)
        # for o in outs:
        #     print(o.shape)
        p1,p2,p3,p4,p5,p6 = outs
        # print(self.channel_reduction_convs)
        p1 = self.channel_reduction_convs[0](p1)
        p2 = self.channel_reduction_convs[1](p2)
        p3 = self.channel_reduction_convs[2](p3)
        p6 = self.channel_reduction_convs[3](p6)
        # print(p1.shape, p2.shape,p3.shape, p4.shape,p5.shape, p6.shape)
        # print(self.channel_reduction_convs)
        p5 = p5 + F.interpolate(p6, scale_factor=2, mode='nearest')
        p4 = p4 + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = p3 + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = p2 + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = p1 + F.interpolate(p2, scale_factor=2, mode='nearest')
        
        # return tuple(outs)
        return tuple([p1,p2,p3,p4,p5,p6])