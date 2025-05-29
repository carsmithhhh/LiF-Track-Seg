import torch.nn as nn
import torch
import sys
sys.path.append('..')
from pointnet_layers import MaskedBatchNorm1d

# WORKS - for now lol
class MaskedMiniPointNet(nn.Module):
    def __init__(self, channels: int, feature_dim: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv1d(channels, 128, 1, bias=False)
        self.norm1 = nn.LayerNorm(128)
        self.conv2 = nn.Conv1d(128, 256, 1)

        self.conv3 = nn.Conv1d(512, 512, 1, bias=False)
        self.norm2 = nn.LayerNorm(512)
        self.conv4 = nn.Conv1d(512, feature_dim, 1)

        ######### Reference Code #################
        # self.first_conv = nn.Sequential(
        #     nn.Conv1d(channels, 128, 1, bias=False),
        #     MaskedBatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, 256, 1),
        # )

        # self.second_conv = nn.Sequential(
        #     nn.Conv1d(512, 512, 1, bias=False),
        #     MaskedBatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, feature_dim, 1),
        # )

    def forward(self, points, mask) -> torch.Tensor:
        # input points: (B, N, C)
        # mask: (B, 1, N)
        #PIPER ADD
        visible_mask = mask.squeeze(1).bool()
        # PIPER ADD chat seems to think that this line should go after the norm
        feature = points.transpose(2, 1)  # (B, C, N)
        out0 = self.conv1(feature)

        # extract visible tokens - original mask is 'mask' but dimensionality has changed
        out0_t = out0.transpose(1, 2) 
        out0_visible = torch.zeros_like(out0_t)

        for b in range(out0_t.shape[0]):
            vis_idx = visible_mask[b]
            out0_visible[b, vis_idx] = self.norm1(out0_t[b, vis_idx])
        out1 = out0_visible.transpose(1, 2)  # (B, 128, N)

        out2 = self.relu(out1)
        out3 = self.conv2(out2)

        # # (B, feature_dim, N) --> (B, feature_dim, 1)
        feature_global = torch.max(out3, dim=2, keepdim=True).values  # (B, feature_dim, 1)
        # # concating global features to each point features
        dist_feature = torch.cat(
            [feature_global.expand(-1, -1, out3.shape[2]), out3], dim=1
        )  # (B, feature_dim * 2, N)

        out4 = self.conv3(dist_feature)
        # extract visible tokens - original mask is 'mask' but dimensionality has changed
        out4_t = out4.transpose(1, 2) 
        out4_visible = torch.zeros_like(out4_t)

        for b in range(out4_t.shape[0]):
            vis_idx = visible_mask[b]
            out4_visible[b, vis_idx] = self.norm2(out4_t[b, vis_idx])
        out5 = out4_visible.transpose(1, 2)  # (B, 512, N)

        out6 = self.relu(out5)
        out7 = self.conv4(out6)

        # (B, feature_dim, N) --> (B, feature_dim)
        feature_global_final = torch.max(out7, dim=2).values  # (B, feature_dim)
        return feature_global_final

       ######## Reference Code Below ##########
        # feature = points.transpose(2, 1)  # (B, C, N)
        # for layer in self.first_conv:
        #     if isinstance(layer, MaskedBatchNorm1d):
        #         feature = layer(feature, mask)
        #     else:
        #         feature = layer(feature)

        # # (B, feature_dim, N) --> (B, feature_dim, 1)
        # feature_global = torch.max(feature, dim=2, keepdim=True).values  # (B, feature_dim, 1)
        # # concating global features to each point features
        # feature = torch.cat(
        #     [feature_global.expand(-1, -1, feature.shape[2]), feature], dim=1
        # )  # (B, feature_dim * 2, N)

        # for layer in self.second_conv:
        #     if isinstance(layer, MaskedBatchNorm1d):
        #         feature = layer(feature, mask)
        #     else:
        #         feature = layer(feature)

        # # (B, feature_dim, N) --> (B, feature_dim)
        # feature_global = torch.max(feature, dim=2).values  # (B, feature_dim)
        # return feature_global