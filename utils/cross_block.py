import torch
import torch.nn as nn
import torch.utils.checkpoint
from timm.models.layers import Mlp, DropPath


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B1, N1, C1 = x.shape
        qkv_x = self.qkv(x).reshape(B1, N1, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        q, _, _ = qkv_x.unbind(0)   # make torchscript happy (cannot use tensor as tuple) [B1, num_heads, N1, C1//num_heads]

        B2, N2, C2 = y.shape
        qkv_y = self.qkv(y).reshape(B2, N2, 3, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        _, k, v = qkv_y.unbind(0)  # make torchscript happy (cannot use tensor as tuple) [B2, num_heads, N2, C2//num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, num_heads, N1, N2]

        # x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)  # attention version

        attn = attn.transpose(-2, -1)  # [B, num_heads, N2, N1]
        x = (attn * v).transpose(1, 2).reshape(B2, N2, C2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class CrossBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y):
        x = x + self.drop_path1(self.ls1(self.cross_attn(self.norm1(x), self.norm1(y))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


if __name__ == '__main__':

    croblk = CrossBlock(dim=2, num_heads=2)
    q = torch.randint(10, (2, 1, 2)).float()
    v = torch.randint(10, (2, 4, 2)).float()
    x = croblk(q, v)
    print(x.shape)

