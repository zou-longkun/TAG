import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, DyGraphConv2d
from timm.models.layers import DropPath


class GrapherModule(nn.Module):
    """Grapher module with graph conv and FC layers"""
    def __init__(self, in_channels, hidden_channels, k=9, dilation=1, drop_path=0.0):
        super(GrapherModule, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),)
        self.graph_conv = nn.Sequential(
            DyGraphConv2d(in_channels, hidden_channels, k, dilation, act=None),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class FFNModule(nn.Module):
    """Feed-forward Network"""
    def __init__(self, in_channels, hidden_channels, drop_path=0.0):
        super(FFNModule, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU())
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class ViGBlock(nn.Module):
    """ViG block with Grapher and FFN modules"""
    def __init__(self, channels, k, dilation, drop_path=0.0):
        super(ViGBlock, self).__init__()
        self.grapher = GrapherModule(channels, channels * 2, k, dilation, drop_path)
        self.ffn = FFNModule(channels, channels * 2, drop_path)

    def forward(self, x):
        x = self.grapher(x)
        x = self.ffn(x)
        return x


class DeepPCG(nn.Module):
    def __init__(self, n_nodes):
        super(DeepPCG, self).__init__()
        act = 'gelu'
        norm = 'batch'
        bias = True
        epsilon = 0.2
        stochastic = False
        conv = 'mr'
        drop_path = 0.0
        n_blocks = 4
        channels = [512, 1024, 1024, 1024]
        reduce_ratios = 1
        dilation = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule
        num_knn = [3, 4, 5, 6]  # number of knn's k
        self.n_nodes = n_nodes
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(n_blocks):
            if i > 0:
                self.backbone += [Seq(nn.Conv2d(channels[i - 1], channels[i], 1, stride=1, padding=0),
                                      nn.BatchNorm2d(channels[i]),)]
            else:
                self.backbone += [Seq(Grapher(channels[i], num_knn[idx], dilation, conv, act, norm, bias,
                                              stochastic, epsilon, reduce_ratios, n=self.n_nodes,
                                              drop_path=dpr[idx], relative_pos=False),
                                      FFNModule(channels[i], channels[i] * 2, drop_path=dpr[idx]),)]
            idx += 1
        self.backbone = Seq(*self.backbone)

        self.bottleneck = Seq(nn.Conv2d(2051, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              nn.GELU(),
                              nn.Dropout(0.0),
                              nn.Conv2d(1024, 512, 1, bias=True))

    def forward(self, x):
        x = self.bottleneck(x)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        x1 = F.adaptive_avg_pool2d(x, 1)
        x2 = F.adaptive_max_pool2d(x, 1)
        x = torch.cat((x1, x2), 1)
        return x
