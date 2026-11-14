

import torch.nn as nn
import torch

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



def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,stride,padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size,stride=stride,padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),#groups = in_channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)





class Averge_FEM(nn.Module):
    def __init__(self, in_channels,out_channels):  #,scale=1.0
        super(Averge_FEM, self).__init__()
        #self.scale = scale
        self.shortcut = BasicConv2d(in_channels, in_channels//3,1, 1,0)

        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3, 1, bias=False),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(),)
        in_channels = in_channels // 12
        rate1, rate2, rate3 = tuple([2,4,8])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels*4, 1, bias=False),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(),)
            #nn.Dropout(0.5))

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, in_channels, rate1)
        self.layer3 = ASPPConv(in_channels, in_channels, rate2)
        self.layer4 = ASPPConv(in_channels, in_channels, rate3)
        self.relu = nn.ReLU(inplace=False)



    def forward(self, x):
        # x1/x2/x3/x4: (B,C,H,W)
        short_x = x
        short = self.shortcut(short_x)

        x = self.cv1(x)
        x0,x1,x2,x3 = list(torch.chunk(x,4, 1))
        y0 = self.layer1(x0)
        y1 = self.layer2(y0 + x1)
        y2 = self.layer3(y1 + x2)
        y3 = self.layer4(y2 + x3)
        out = torch.cat((y0,y1,y2,y3),1)
        out = self.project(out)

        out =out+short

        return  out

class BackAverge_FEM(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(BackAverge_FEM, self).__init__()
        self.shortcut = BasicConv2d(in_channels, in_channels // 4, 1, 1, 0)

        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(), )
        in_channels = in_channels // 16
        rate1, rate2, rate3 = tuple([2, 4, 8])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, 1, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(), )
        # nn.Dropout(0.5))

        self.layer1 = nn.Sequential(
            BasicConv(in_channels, in_channels // 4, kernel_size=1, ),
            nn.Conv2d(in_channels // 4, in_channels, 3, padding=1, dilation=1, bias=False),
            # groups = in_channels , bias=False
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        self.layer2 = nn.Sequential(
                        ASPPConv(in_channels, in_channels, rate1))
        self.layer3 = nn.Sequential(
                    ASPPConv(in_channels, in_channels, rate2))
        self.layer4 = nn.Sequential(
                    ASPPConv(in_channels, in_channels, rate3))
        self.relu = nn.ReLU(inplace=False)



    def forward(self, x):
        # x1/x2/x3/x4: (B,C,H,W)
        short_x = x
        short = self.shortcut(short_x)


        x = self.cv1(x)
        x0, x1, x2, x3 = list(torch.chunk(x, 4, 1))
        y0 = self.layer1(x0)  
        y1 = self.layer2(y0 + x1)
        y2 = self.layer3(y1 + x2)
        y3 = self.layer4(y2 + x3)
        out = torch.cat((y0, y1, y2, y3), 1)
        out = self.project(out)
        out = out + short
        out = self.relu(out)

        return out

if __name__ == '__main__':
    # (B,C,H,W)
    tensor1 = torch.rand(1, 192, 44, 44)
    Model = Averge_FEM(in_channels=192,out_channels=1024)
    out = Model(tensor1)
    print(out.shape)
