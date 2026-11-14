"""TDD-YOLO v2 modules.

CSAF2   - fixed CBAM_cor (Channel-Spatial Attention Fusion). Every weight is created in
          __init__ and trained; nothing depends on the feature-map resolution.
CAFE2   - fixed co_FEM (Coordinate Attention Feature Enhancement): both coordinate-attention
          branches are complete (branch2_3, the dilated 3x3, is actually used).
DySample - learned point-sampling upsampler (Liu et al., "Learning to Upsample by Learning
          to Sample", ICCV 2023), static-scope "lp" variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("CSAF2", "CAFE2", "DySample")


# --------------------------------------------------------------------------------------
# CSAF2
# --------------------------------------------------------------------------------------
class CSAF2(nn.Module):
    """Channel-spatial attention fusion with trained, resolution-independent fusion.

    ch0  = MLP1(avgpool x) + MLP1(maxpool x)                       (B,C,1,1) logits
    sp   = conv7x7([max_c x, mean_c x, mean_c(x * sigmoid(ch0))])  (B,1,H,W) logits
    d    = sum_hw softmax_hw(sp) * x                               (B,C,1,1) descriptor
    ch   = ch0 + MLP2(d)          (MLP2's last layer is zero-initialised, so CSAF2 starts as CBAM)
    out  = x * sigmoid(ch) + x * sigmoid(sp)
    """

    def __init__(self, c, reduction=8, k=7):
        super().__init__()
        h = max(c // reduction, 8)
        self.mlp_c = nn.Sequential(nn.Conv2d(c, h, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(h, c, 1, bias=False))
        self.spatial = nn.Conv2d(3, 1, k, padding=k // 2, bias=True)
        self.mlp_s = nn.Sequential(nn.Conv2d(c, h, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(h, c, 1, bias=False))
        nn.init.zeros_(self.mlp_s[-1].weight)

    def forward(self, x):
        b, c = x.shape[:2]
        ch = self.mlp_c(F.adaptive_avg_pool2d(x, 1)) + self.mlp_c(F.adaptive_max_pool2d(x, 1))
        xc = x * torch.sigmoid(ch)
        s_in = torch.cat([x.amax(1, keepdim=True), x.mean(1, keepdim=True), xc.mean(1, keepdim=True)], 1)
        sp = self.spatial(s_in)  # (B,1,H,W)
        w = torch.softmax(sp.flatten(2).float(), dim=-1).to(x.dtype)  # (B,1,HW)
        d = torch.bmm(x.flatten(2), w.transpose(1, 2)).view(b, c, 1, 1)  # (B,C,1,1)
        ch = ch + self.mlp_s(d)
        return x * torch.sigmoid(ch) + x * torch.sigmoid(sp)


# --------------------------------------------------------------------------------------
# CAFE2
# --------------------------------------------------------------------------------------
class BasicConv(nn.Module):
    """Conv-BN(-ReLU), same as the original co_FEM helper."""

    def __init__(self, c1, c2, k, s=1, p=0, d=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-5, momentum=0.01)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CoordGate(nn.Module):
    """Coordinate attention gates (a_h: B,C,H,1 and a_w: B,C,1,W), as in the original CoordAtt."""

    def __init__(self, c, reduction=16):
        super().__init__()
        h = max(c // reduction, 1)
        self.mlp = nn.Sequential(nn.Conv2d(c, h, 1, bias=False), nn.ReLU(), nn.Conv2d(h, c, 1, bias=False))

    def forward(self, x):
        H, W = x.shape[2:]
        y = torch.cat([x.mean(3, keepdim=True), x.mean(2, keepdim=True).transpose(2, 3)], 2)  # (B,C,H+W,1)
        a_h, a_w = torch.split(self.mlp(y), [H, W], dim=2)
        return torch.sigmoid(a_h), torch.sigmoid(a_w.transpose(2, 3))


class CAFE2(nn.Module):
    """co_FEM with both coordinate-attention branches complete.

    branch0: 1x1 -> 3x3
    branch1: 1x1 -> 1x3 -> *a_h -> 3x1 -> *a_w -> dilated 3x3 (d=5)
    branch2: 1x1 -> 3x1 -> *a_w -> 1x3 -> *a_h -> dilated 3x3 (d=5)   <- fixed: uses branch2_3
    out = ReLU(ConvLinear(cat) * scale + shortcut(x))
    """

    def __init__(self, c1, c2, scale=1.0, map_reduce=8):
        super().__init__()
        self.scale = scale
        ci = c1 // map_reduce
        self.branch0 = nn.Sequential(BasicConv(c1, 2 * ci, 1), BasicConv(2 * ci, 2 * ci, 3, p=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(c1, ci, 1), BasicConv(ci, 2 * ci, (1, 3), p=(0, 1)))
        self.branch1_2 = BasicConv(2 * ci, 2 * ci, (3, 1), p=(1, 0))
        self.branch1_3 = BasicConv(2 * ci, 2 * ci, 3, p=5, d=5, relu=False)
        self.branch2 = nn.Sequential(BasicConv(c1, ci, 1), BasicConv(ci, 2 * ci, (3, 1), p=(1, 0)))
        self.branch2_2 = BasicConv(2 * ci, 2 * ci, (1, 3), p=(0, 1))
        self.branch2_3 = BasicConv(2 * ci, 2 * ci, 3, p=5, d=5, relu=False)
        self.ConvLinear = BasicConv(6 * ci, c2, 1, relu=False)
        self.shortcut = BasicConv(c1, c2, 1, relu=False)
        self.corconv1 = BasicConv(c1, 2 * ci, 1)
        self.cohw = CoordGate(2 * ci)
        self.relu = nn.ReLU()

    def forward(self, x):
        a_h, a_w = self.cohw(self.corconv1(x))
        x0 = self.branch0(x)
        x1 = self.branch1_3(self.branch1_2(self.branch1(x) * a_h) * a_w)
        x2 = self.branch2_3(self.branch2_2(self.branch2(x) * a_w) * a_h)
        out = self.ConvLinear(torch.cat((x0, x1, x2), 1))
        return self.relu(out * self.scale + self.shortcut(x))


# --------------------------------------------------------------------------------------
# DySample
# --------------------------------------------------------------------------------------
class DySample(nn.Module):
    """DySample upsampler, "lp" style with static scope factor 0.25.

    A 1x1 conv predicts per-pixel offsets for scale^2 sub-pixel points in each of `groups`
    channel groups; the input is bilinearly resampled at (initial position + offset).
    With zero offset it reduces to bilinear upsampling, so it trains stably from scratch.
    """

    def __init__(self, c, scale=2, groups=4):
        super().__init__()
        assert c % groups == 0, f"DySample: channels {c} must be divisible by groups {groups}"
        self.scale, self.groups = scale, groups
        self.offset = nn.Conv2d(c, 2 * groups * scale * scale, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        s, g = self.scale, self.groups
        offset = (self.offset(x) * 0.25 + self.init_pos.to(x.dtype)).view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")).transpose(1, 2).unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), s)
            .view(B, 2, -1, s * H, s * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        out = F.grid_sample(
            x.reshape(B * g, -1, H, W), coords.to(x.dtype), mode="bilinear", align_corners=False, padding_mode="border"
        )
        return out.view(B, -1, s * H, s * W)
