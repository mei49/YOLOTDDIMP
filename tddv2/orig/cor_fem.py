import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class co_FEM(nn.Module):
    def __init__(self, in_planes, out_planesls, stride=1, scale=1.0, map_reduce=8):
        super(co_FEM, self).__init__()

        self.scale = scale
        self.out_channels = in_planes
        out_planes = in_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1))
            )
        self.branch1_2 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0))
            )
        self.branch1_3 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )


        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
                    )
        self.branch2_2 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            )
        self.branch2_3 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
            )


        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

        self.corconv1 = BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride)
        self.cohw = CoordAtt(2 * inter_planes,2 * inter_planes)


    def forward(self, x):
        x0 = self.branch0(x)  # 1*1 3*3

        x_cor = self.corconv1(x)
        h_co, w_co = self.cohw(x_cor)


        x1 = self.branch1(x)
        x1 = x1 *h_co
        x1 = self.branch1_2(x1)
        x1 = x1  * w_co
        x1 = self.branch1_3(x1)



        x2 = self.branch2(x)
        x2 = x2*w_co
        x2 = self.branch2_2(x2)
        x2 = x2*h_co
        x2 = self.branch2_2(x2)


        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short  #
        out = self.relu(out)

        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.MLP = nn.Sequential(
            nn.Conv2d(inp, inp // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inp // reduction, inp, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        B, C, H, W = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)


        y = torch.cat([x_h, x_w], dim=2)
        y = self.MLP(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(x_h)
        a_w = self.sigmoid(x_w)


        return a_h,a_w


if __name__ == '__main__':
    input = torch.randn(1, 64, 16, 16)  # 假设输入大小
    model = co_FEM(64,1024)
    out = model(input)
    print(out.shape)
