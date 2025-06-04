# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale as convert_grayscale
import sys, os, importlib
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from slowfast_modules import head_helper, resnet_helper, stem_helper  # noqa
from slowfast_modules import weight_init_helper as init_helper
from slowfast_modules.checkpoint import load_checkpoint
from fvcore.common.config import CfgNode
import yaml
import pdb
from torchvision.transforms.functional import rgb_to_grayscale, resize, gaussian_blur


# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ]
}

_POOL1 = {
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


class SlowFastLoader(nn.Module):
    def __init__(self, alpha):
        super(SlowFastLoader, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        assert(len(x.shape) == 5)

        x_f = x[:]
        x_s = x[:,:,::self.alpha]

        return (x_s, x_f)


class MotionAwareSlowFastLoader(nn.Module):
    def __init__(self, alpha, feature_size=28):
        super(MotionAwareSlowFastLoader, self).__init__()
        self.alpha = alpha
        self.feature_size = feature_size

    def forward(self, x):
        B, C, T, H, W = x.shape
        k = T // self.alpha
        indices = torch.arange(T, device=x.device)
        segments = torch.tensor_split(indices, k)
        
        x_s_list = []
        for b in range(B):
            selected_frames = []
            for seg in segments:
                segment_frames = x[b, :, seg, :, :]
                L = segment_frames.size(1)
                gray_frames = rgb_to_grayscale(segment_frames.permute(1, 0, 2, 3))
                resized_frames = resize(gray_frames, [self.feature_size]*2)
                
                features = resized_frames.view(L, -1)
                centroid = features.mean(dim=0)
                distances = torch.norm(features - centroid, dim=1)
                min_idx = torch.argmin(distances)
                selected_frame = segment_frames[:, min_idx, :, :]
                selected_frames.append(selected_frame)

            x_s = torch.stack(selected_frames, dim=1)
            x_s_list.append(x_s)
        
        x_s = torch.stack(x_s_list, dim=0)
        x_f = x
        
        return (x_s, x_f)


class FrameDifferenceSampling(nn.Module):
    def __init__(self, alpha, feature_size=28):
        super(FrameDifferenceSampling, self).__init__()
        self.alpha = alpha
        self.feature_size = feature_size

    def forward(self, x):
        B, C, T, H, W = x.shape
        k = T // self.alpha
        indices = torch.arange(T, device=x.device)
        segments = torch.tensor_split(indices, k)
        
        x_s_list = []
        for b in range(B):
            selected_frames = []
            for seg in segments:
                segment_frames = x[b, :, seg, :, :]
                L = segment_frames.size(1)
                flat = segment_frames.permute(1,0,2,3).reshape(L, -1)  # (L, C*H*W)
                dists = torch.zeros(L, device=flat.device)
                dists[1:] = torch.norm(flat[1:] - flat[:-1], dim=1)
                selected_idx = torch.argmax(dists)
                selected_frame = segment_frames[:, selected_idx, :, :]
                selected_frames.append(selected_frame)
            x_s = torch.stack(selected_frames, dim=1)
            x_s_list.append(x_s)
        
        x_s = torch.stack(x_s_list, dim=0)
        x_f = x
        
        return (x_s, x_f)


class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg, multi=False):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = nn.BatchNorm3d
        fuse_helper = importlib.import_module('slowfast_modules.fuse_helper')
        self.fuse = getattr(fuse_helper, cfg.SLOWFAST.FUSE)
        self.cfg = cfg
        self.num_pathways = 2
        # self.loader = SlowFastLoader(cfg.SLOWFAST.ALPHA)
        # self.loader = FrameDifferenceSampling(cfg.SLOWFAST.ALPHA)
        self.loader = MotionAwareSlowFastLoader(cfg.SLOWFAST.ALPHA)
        self._construct_network(cfg, multi)
        init_helper.init_weights(self)

    def _construct_network(self, cfg, multi):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=[3, 3],
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = self.fuse(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )
        self.head1 = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group,
                2 * width_per_group // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            pool_size=[
                [1, 56, 56],
                [1, 56, 56],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            multi=multi,
            cfg=cfg,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group,
                2 * width_per_group // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = self.fuse(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )
        self.head2 = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group * 4,
                2 * width_per_group * 4 // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            pool_size=[
                [1, 56, 56],
                [1, 56, 56],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            multi=multi,
            cfg=cfg,
        )
        
        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group * 4,
                2 * width_per_group * 4 // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = self.fuse(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )
        self.head3 = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group * 8,
                2 * width_per_group * 8 // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            pool_size=[
                [1, 28, 28],
                [1, 28, 28],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            multi=multi,
            cfg=cfg,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group * 8,
                2 * width_per_group * 8 // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = self.fuse(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )
        self.head4 = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group * 16,
                2 * width_per_group * 16 // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            pool_size=[
                [1, 14, 14],
                [1, 14, 14],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            multi=multi,
            cfg=cfg,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group * 16,
                2 * width_per_group * 16 // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )
        self.head5 = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32 + width_per_group * 32 // out_dim_ratio if cfg.SLOWFAST.FUSE in ('FSInteraction', 'FuseBiCat')
                else width_per_group * 32,
                2 * width_per_group * 32 // cfg.SLOWFAST.BETA_INV if cfg.SLOWFAST.FUSE in ('FuseBiCat')
                else width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            pool_size=[
                [1, 7, 7],
                [1, 7, 7],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            multi=multi,
            cfg=cfg,
        )

    def forward(self, x):
        x = self.loader(x)
        x = self.s1(x)
        x = self.s1_fuse(x)
        stage1 = self.head1(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        stage2 = self.head2(x)
        x = self.s3(x)
        x = self.s3_fuse(x)
        stage3 = self.head3(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        stage4 = self.head4(x)
        x = self.s5(x)
        x = self.head5(x)
        return torch.cat([stage1, stage2, stage3], 1), stage4, x

def slowfast18(slowfast_path, slowfast_config='SLOWFAST_8x8_R18.yaml', slowfast_args=None, pretrained=False):
    cfg = yaml.load(open(slowfast_path + '/configs/' + slowfast_config, 'r'), Loader=yaml.FullLoader)
    cfg = CfgNode(cfg)
    if slowfast_args is not None:
        cfg.merge_from_list(slowfast_args)
    model = SlowFast(cfg)
    return model

def slowfast50(slowfast_config='SLOWFAST_8x8_R50.yaml', slowfast_args=[], load_pkl=True, multi=False):
    cfg = yaml.load(open('slowfast_modules/configs/' + slowfast_config, 'r'), Loader=yaml.FullLoader)
    cfg = CfgNode(cfg)
    if len(slowfast_args) > 0:
        cfg.merge_from_list(slowfast_args)
    model = SlowFast(cfg, multi)
    if load_pkl:
        load_checkpoint('ckpt/SLOWFAST_8x8_R50.pkl', model)
    return model

def slowfast101(slowfast_config='SLOWFAST_64x2_R101_50_50.yaml', slowfast_args=[], load_pkl=True, multi=False):
    cfg = yaml.load(open('slowfast_modules/configs/' + slowfast_config, 'r'), Loader=yaml.FullLoader)
    cfg = CfgNode(cfg)
    if len(slowfast_args) > 0:
        cfg.merge_from_list(slowfast_args)
    model = SlowFast(cfg, multi)
    if load_pkl:
        load_checkpoint('ckpt/SLOWFAST_64x2_R101_50_50.pkl', model)
    return model

def test():
    cfg = yaml.load(open('slowfast_modules/configs/SLOWFAST_64x2_R101_50_50.yaml', 'r'), Loader=yaml.FullLoader)
    cfg = CfgNode(cfg)
    model = SlowFast(cfg)
    load_checkpoint('ckpt/SLOWFAST_64x2_R101_50_50.pkl', model)
    print(model)
    y = model(torch.randn(2, 3, 208, 224, 224))
    print(y.shape)
