from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return Net(args)

class FeatureNorm(nn.Module):
    def __init__(self, n_channels):
        super(FeatureNorm, self).__init__()
        self.proc = nn.Conv2d(n_channels, n_channels, 3, padding=3//2)
        self.gamma = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 1), nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2, groups=n_channels), nn.Sigmoid(),
        )
        self.beta = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 1), nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2, groups=n_channels),
        )
    def forward(self, x):
        proc = self.proc(x)
        gamma = self.gamma(proc)
        beta = self.beta(proc)
        return x * (1+gamma) + beta

class SEBlock(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 16),
            nn.ReLU(True),
            nn.Linear(input_dim // 16, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Upsample(nn.Module):
    def __init__(self, n_channels, scale=4):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(n_channels, 3 * scale * scale, 3, padding=3 // 2)
        self.up = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class WideConv(nn.Module):
    def __init__(self, n_channels):
        super(WideConv, self).__init__()
        self.conv_3x3 = nn.Conv2d(n_channels, n_channels, 3, padding=3//2)
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2), nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2),
        )
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2), nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2), nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2),
        )
        self.conv_9x9 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2), nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2), nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2), nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2),
        )
        self.conv_1x1 = nn.Conv2d(n_channels*4, n_channels, 1)
        #self.senet = SEBlock(n_channels)
        self.fn = FeatureNorm(n_channels)
    def forward(self, x):
        x3 = self.conv_3x3(x)
        x5 = self.conv_5x5(x3) + x3
        x7 = self.conv_7x7(x5) + x5
        x9 = self.conv_9x9(x7) + x7
        x1 = self.conv_1x1(torch.cat((x3, x5, x7, x9), 1))
        #x1 = self.senet(x1)
        x1 = self.fn(x1)
        return x1+x

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        scale = args.scale[0]
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        n_conv = 8

        module_body = [WideConv(64) for _ in range(n_conv)] + [nn.Conv2d(64, 64, 3, padding=3//2),
                                                               nn.ReLU(True),
                                                               nn.Conv2d(64, 64, 3, padding=3//2),]

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Conv2d(3, 64, 3, padding=3//2)
        self.body = nn.Sequential(*module_body)
        self.tail = nn.Conv2d(64, 64, 3, padding=3//2)
        self.upsample = Upsample(n_channels=64, scale=scale)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        x = self.upsample(x)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

