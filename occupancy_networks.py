import torch
import torch.nn as nn
import numpy as np
from torch_utils import positional_encoding


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        assert stride == 1

        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        identity = x if self.inplanes == self.planes else self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# depths = [32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32]
depths = [32, 32, 32]


class OccupancyNetwork(nn.Module):
    # def __init__(self, layer_depths=[32, 64, 128, 256, 512]):
    # def __init__(self, layer_depths=[512, 512, 512, 512, 512]):
    # def __init__(self, layer_depths=[32, 32, 32]):
    # def __init__(self, layer_depths=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]):
    def __init__(
        self, layer_depths=depths, L=10, input_dims=3, use_positional_encoding=True,
    ):
        super(OccupancyNetwork, self).__init__()
        embedding_size = 2 * input_dims * L if use_positional_encoding else input_dims
        self.blocks = nn.Sequential(
            *tuple(
                BasicBlock(
                    layer_depths[i - 1] if i > 0 else embedding_size, layer_depths[i]
                )
                for i in range(len(layer_depths))
            )
        )
        self.fc = conv1x1(layer_depths[-1], 1, 1)
        self.L = L
        self.input_dims = input_dims
        self.embedding_size = embedding_size
        self.use_positional_encoding = use_positional_encoding

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        bs = x.shape[0]
        x = x.clone().detach().requires_grad_(True)
        num_dims = x.shape[-1]

        if self.use_positional_encoding:
            e = positional_encoding(x)
        xs = e.view(bs, self.embedding_size, 1, 1)

        # x is [BS, (X, Y, Z), WIDTH, HEIGHT]
        assert len(xs.shape) == 4
        # assert xs.shape[1] == self.input_dims

        xs = self.blocks(xs)
        xs = self.fc(xs)
        return xs.squeeze(), x
