import torch
from torch import nn
from torch.nn import init


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # = self.bn(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = max_out + avg_out
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        return output


class CBAM_cor(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ChannelAttention = ChannelAttention(channel=channel, reduction=reduction)
        self.SpatialAttention = SpatialAttention(kernel_size=kernel_size)
        self.reduction = reduction  #

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            # 动态获取输入
            B, C, H, W = x.size()
            residual = x


            Channel_x = self.ChannelAttention(x).reshape(B, C, 1, 1)  # (B,C,1,1)


            Spatial_x = self.SpatialAttention(x).reshape(B, H * W, 1, 1)  # (B,1,H,W)-->(B,HW,1,1)

            # 动态拼接
            joint_channel = C + H * W
            MLP = nn.Sequential(
                nn.Conv2d(joint_channel, C // self.reduction, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(C // self.reduction, joint_channel, 1, bias=False)
            ).to(x.device)

            CS_x = torch.cat([Channel_x, Spatial_x], 1)
            CS_xx = MLP(CS_x)


            Channel_x = CS_xx[:, :C, :].reshape(B, C, 1, 1)  # (B,C,1,1)
            Spatial_x = CS_xx[:, C:, :].reshape(B, 1, H, W)  # (B,HW,1,1)-->(B,1,H,W)

            Channel_weight = torch.sigmoid(Channel_x)
            Spatial_weight = torch.sigmoid(Spatial_x)

            # 最终输出

            out = x * Channel_weight +x * Spatial_weight

            return out


if __name__ == '__main__':
    input = torch.randn(1, 64, 16, 16)
    model = CBAM_cor(channel=64, reduction=8, kernel_size=7)
    out = model(input)
    print(out.shape)



