import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
import numpy as np
from PCG import DeepPCG


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def cos_sim(x1, x2):
    scores = torch.acos(torch.cosine_similarity(x1, x2, dim=1)) / np.pi * 180
    return scores.mean()


def l2_norm(input, axit=1):
    norm = torch.norm(input, 2, axit, True)
    output = torch.div(input, norm)
    return output


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    # device = torch.device('cuda')
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


def trig_function(pc):
    B, C, N = pc.shape
    device = pc.device
    feat_dim = 36 // (3 * 2)
    feat_range = torch.arange(feat_dim).float().to(device)
    dim_embed = torch.pow(1000, feat_range / feat_dim)
    div_embed = torch.div(100 * pc.unsqueeze(-1), dim_embed)

    sin_embed = torch.sin(div_embed)
    cos_embed = torch.cos(div_embed)
    position_embed = torch.cat([sin_embed, cos_embed], -1)
    position_embed = position_embed.permute(0, 1, 3, 2).contiguous()
    position_embed = position_embed.view(B, -1, N)

    return position_embed


class DGCNN(nn.Module):
    def __init__(self, k, emb_dims):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        num_points = x.shape[2]
        # x = trig_function(x)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x5 = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x_1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x_2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x_1, x_2), 1)  # (batch_size, emb_dims*2)

        x_rep = x.unsqueeze(-1).repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x_seg = torch.cat((x_rep, x1, x2, x3, x4), dim=1)  # (batch_size, 2048+64+64+128+256, num_points)

        return x_seg, x


class PointNet(nn.Module):
    def __init__(self, emb_dims):
        super(PointNet, self).__init__()
        self.emb_dims = emb_dims
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, self.emb_dims, 1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1)
        num_points = x.shape[2]
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x_max_pool, _ = torch.max(x5, dim=2, keepdim=False)
        x_avg_pool = torch.mean(x5, dim=2, keepdim=False)
        x = torch.cat((x_max_pool, x_avg_pool), 1)

        x_rep = x.unsqueeze(-1).repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x_seg = torch.cat((x_rep, x1, x2, x3, x4), dim=1)  # (batch_size, 2048+64+64+128+256, num_points)

        return x_seg, x


class freq_Decoder(nn.Module):
    def __init__(self):
        super(freq_Decoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(nn.Conv1d(2560, 512, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(256, 1, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)  # (batch_size, 2048+64+64+128+256, num_points) -> (batch_size, 512, num_points)
        x = self.conv2(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp(x)
        x = self.conv3(x)  # (batch_size, 256, num_points) -> (batch_size, 1, num_points)
        x = x.squeeze()

        return x


def maxpool(x, dim=1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class pyramid_Decoder(nn.Module):

    def __init__(self):
        super(pyramid_Decoder, self).__init__()

        # Submodules

        self.fc_c = nn.Linear(2051, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        self.block1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.block2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc_out = nn.Linear(128, 4)
        self.bn3 = nn.BatchNorm1d(4)

    def forward(self, c):
        x = F.relu(self.bn0(self.fc_c(c)))  # bnumber,2048+64 -> # bnumber,1024
        x = F.relu(self.bn1(self.block1(x)))
        x = F.relu(self.bn2(self.block2(x)))  # bnumber,512 -> # bnumber,128

        x = F.normalize(self.bn3(self.fc_out(x)), dim=1)

        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                nn.BatchNorm1d(out_ch),
                # nn.LayerNorm(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                self.ac
            )

    def forward(self, x):
        # x = l2_norm(x, 1)
        x = self.fc(x)
        return x


class pos_emb(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(pos_emb, self).__init__()
        self.hidden_dim = 32
        self.output_dim = output_dim

        self.bn = nn.BatchNorm1d(self.hidden_dim)

        self.conv1 = nn.Conv1d(input_dim, self.hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.bn(self.conv1(x)), negative_slope=0.2)
        x = self.conv2(x)
        return x


class MLPNet_relu(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn=True):
        super().__init__()
        list_layers = mlp_layers_relu(nch_input, nch_layers, b_shared, bn_momentum, dropout, if_bn)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


def mlp_layers_relu(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn=True):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
            init.xavier_normal_(weights.weight, gain=1.0)
        else:
            weights = torch.nn.Linear(last, outp)
            init.xavier_normal_(weights.weight, gain=1.0)
        layers.append(weights)
        if if_bn:
            layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if not b_shared and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


class mask_discriminator(nn.Module):
    def __init__(self, input_dim):
        super(mask_discriminator, self).__init__()
        self.of1 = 512
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)

        self.conv1 = nn.Conv1d(input_dim, self.of1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(self.of3, 1, kernel_size=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = self.conv4(x)
        x = torch.sigmoid(x)
        return x


class Reconstruction(nn.Module):
    def __init__(self, args, input_dim):
        super(Reconstruction, self).__init__()
        self.args = args
        self.of1 = 512
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.conv1 = nn.Conv1d(input_dim, self.of1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(self.of3, 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = self.conv4(x)
        return x.permute(0, 2, 1)


class class_classifier(nn.Module):
    def __init__(self, input_dim, num_class=10):
        super(class_classifier, self).__init__()

        self.mlp1 = fc_layer(input_dim, 512, bias=True, activation='leakyrelu', bn=True)
        self.dp1 = nn.Dropout(p=0.5)
        self.mlp2 = fc_layer(512, 256, bias=True, activation='leakyrelu', bn=True)
        self.dp2 = nn.Dropout(p=0.5)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        return logits


class Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.decoder = pyramid_Decoder().to(device)
        self.encoder = DGCNN(10, 1024).to(device)
        self.local_encoder = PointNet(1024).to(device)
        self.classifier = class_classifier(2048, 10).to(device)
        self.pcg_classifier = class_classifier(2048, 10).to(device)
        self.pcg = DeepPCG(16).to(device)
        self.pos_emb = pos_emb(3, 64).to(device)
        self.mask_discrim = mask_discriminator(2112).to(device)


if __name__ == '__main__':
    pc = torch.randn(10, 3, 15).cuda()  # B, C, N
    B, C, N = pc.shape
    feat_dim = 36 // (3 * 2)

    feat_range = torch.arange(feat_dim).float().cuda()
    dim_embed = torch.pow(1000, feat_range / feat_dim)
    div_embed = torch.div(100 * pc.unsqueeze(-1), dim_embed)

    sin_embed = torch.sin(div_embed)
    cos_embed = torch.cos(div_embed)
    position_embed = torch.cat([sin_embed, cos_embed], -1)
    position_embed = position_embed.permute(0, 1, 3, 2).contiguous()
    position_embed = position_embed.view(B, -1, N)
    print(position_embed.shape)