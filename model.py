# Parametric Networks for 3D Point Cloud Classification
import torch
import torch.nn as nn
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from utils.modelpn_utils import *
from Model_Base import DGCNN, class_classifier, pyramid_Decoder
from PCG import DeepPCG


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()  # B, S
        lc_xyz = index_points(xyz, fps_idx)  # B, S, 3; "lc" mean local center
        lc_x = index_points(x, fps_idx)  # B, S, C

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)  # B, S, K
        knn_xyz = index_points(xyz, knn_idx)  # B, S, K, 3
        knn_x = index_points(x, knn_idx)  # B, S, K, C

        return lc_xyz, lc_x, knn_xyz, knn_x


# PosE for Raw-point Embedding
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        return position_embed


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):  # in_dim = 3
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape  # [16, 3, [512->256->128->64], 40]
        feat_dim = self.out_dim // (self.in_dim * 2)  # 72,144,288,288 // (3 * 2) = 12,24,48,48

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)  # [16, 3, 512, 40, 12])

        sin_embed = torch.sin(div_embed)  # [16, 3, 512, 40, 12])
        cos_embed = torch.cos(div_embed)  # [16, 3, 512, 40, 12])
        position_embed = torch.cat([sin_embed, cos_embed], -1)  # [16, 3, 512, 40, 24])
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()  # [16, 3, 24, 512, 40]
        position_embed = position_embed.view(B, self.out_dim, G, K)  # [16, 72, 512, 40]

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, block_num, dim_expansion):
        super().__init__()
        self.type = 'mn'
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        self.linear1 = Linear1Layer(out_dim * expand, out_dim, bias=False)
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(Linear2Layer(out_dim, bias=True))
        self.linear2 = nn.Sequential(*self.linear2)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalization
        if self.type == 'mn':
            mean_xyz = lc_xyz.unsqueeze(dim=-2)
            std_xyz = torch.std(knn_xyz - mean_xyz)
            knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        elif self.type == 'scan':
            knn_xyz = knn_xyz.permute(0, 3, 1, 2)
            knn_xyz -= lc_xyz.permute(0, 2, 1).unsqueeze(-1)
            knn_xyz /= torch.abs(knn_xyz).max(dim=-1, keepdim=True)[0]
            knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B, G, K, 2C

        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)  # B, 3, G, K
        knn_x = knn_x.permute(0, 3, 1, 2)  # B, 2C, G, K
        knn_x = self.linear1(knn_x.reshape(B, -1, G * K)).reshape(B, -1, G, K)  # B, C, G, K

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)  # B, C, G, K

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)  # B, [C->(C/2)->C], G, K

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return lc_x


# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels / 2),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels / 2)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels / 2), out_channels=in_channels,
                      kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


# Parametric Encoder
class EncP(nn.Module):
    def __init__(self, in_channels, input_points, num_stages, embed_dim, proj_dim,
                 k_neighbors, alpha, beta, LGA_block, dim_expansion):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        # self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)  # non-parametric
        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList()  # FPS, kNN
        self.LGA_list = nn.ModuleList()  # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList()  # Pooling

        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta, LGA_block[i], dim_expansion[i]))
            self.Pooling_list.append(Pooling(out_dim))

        self.projector = Linear1Layer(out_dim, self.proj_dim, bias=False)

    def forward(self, x):
        # Raw-point Embedding
        xyz = x  # B, N, 3
        x = x.permute(0, 2, 1)  # B, 3, N
        x = self.raw_point_embed(x)  # B, C, N

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)  # B, C, S, K
            # Pooling
            x = self.Pooling_list[i](knn_x_w)  # B, C, S

            xyz_list.append(xyz)
            x_list.append(x)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)

        # Projector
        x = x.unsqueeze(-1)
        x = self.projector(x)

        return x.squeeze(), xyz_list, x_list


# Parametric Decoder
class DecP(nn.Module):
    def __init__(self, embed_dim, proj_dim, dim_expansion, num_stages, de_neighbors):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        out_dim = self.embed_dim
        out_dim_list = [out_dim]
        self.linear_list = []
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            out_dim_list.append(out_dim)

        mid_dim = out_dim_list[-1]
        for i in range(self.num_stages):
            mid_dim += out_dim_list[-2 - i]
            self.linear = Linear1Layer(mid_dim, mid_dim, bias=False).cuda()
            self.linear_list.append(self.linear)

        self.str_dec = nn.Sequential(Linear1Layer(sum(out_dim_list), 256, bias=False),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(256, 1, kernel_size=1, bias=False)
                                     )

    def propagate(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)

            index_points(xyz1, idx)
            interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)

        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return new_points

    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()

        x = x_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x)
            x = self.linear_list[i](x)

        out = self.str_dec(x)
        return out.squeeze()


class Point_PN(nn.Module):
    def __init__(self, args, in_channels=3, input_points=1024, num_stages=4, embed_dim=36, proj_dim=2048,
                 k_neighbors=40, alpha=1000, beta=100, LGA_block=[2, 2, 1, 1], dim_expansion=[2, 2, 2, 1]):
        super().__init__()
        # Parametric Encoder
        self.encoder_dg = DGCNN(args.cls_num, 1024)
        self.encoder = EncP(in_channels, input_points, num_stages, embed_dim, proj_dim,
                            k_neighbors, alpha, beta, LGA_block, dim_expansion)
        self.out_channel = proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, args.cls_num)
        )

        self.local_encoder = EncP(in_channels, args.patch_size, 2, embed_dim, proj_dim,
                                  k_neighbors, alpha, beta, LGA_block, dim_expansion)
        self.local_classifier = class_classifier(2048, args.cls_num)
        self.decoder_geo = pyramid_Decoder()
        self.decoder_str = DecP(embed_dim, proj_dim, dim_expansion, num_stages, de_neighbors=6)


class Deep_PCG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder_dg = DGCNN(args.cls_num, 1024)
        self.classifier_dg = class_classifier(2048, args.cls_num)

        self.pcg = DeepPCG(16)
        self.classifier_pcg = class_classifier(2048, args.cls_num)

