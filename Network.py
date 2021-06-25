import torch
import torch.nn as nn

class Mutual_net(nn.Module):
    def __init__(self):
        super(Mutual_net, self).__init__()

        self.fc1 = nn.Linear(10, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 10)
        self.last = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = self.last(x)
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class myBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(myBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = 64
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AutoEncoder(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(AutoEncoder, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        replace_stride_with_dilation = [False, False, False]
        l = [32, 64, 128]
        self.inplanes = l[0]  # 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # self.layer1 = self._make_layer(block, 64, layers[0], n_inplane=64)
        self.layer1 = self._make_layer(block, l[0], layers[0], n_inplane=l[0])

        self.fconv1 = nn.Conv2d(l[0], l[1], kernel_size=5, stride=2, padding=2)

        self.layer2 = self._make_layer(block, l[1], layers[1], n_inplane=l[1],
                                       dilate=replace_stride_with_dilation[0])

        self.fconv2 = nn.Conv2d(l[1], l[2], kernel_size=3, stride=2)

        self.layer3 = self._make_layer(block, l[2], layers[2], n_inplane=l[2],
                                       dilate=replace_stride_with_dilation[0])

        self.fc1 = nn.Linear(128 * 3 * 3, 10)

        self.fc2 = nn.Linear(in_features=10,
                             out_features=1152)
        self.layer4 = self._make_layer(block, l[2], layers[2], n_inplane=l[2],
                                       dilate=replace_stride_with_dilation[0])
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.layer5 = self._make_layer(block, l[1], layers[1], n_inplane=l[1],
                                       dilate=replace_stride_with_dilation[0])
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=1)
        self.layer6 = self._make_layer(block, l[0], layers[0], n_inplane=l[0])
        self.deconv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=2)

    def _make_layer(self, block, planes, blocks, n_inplane, stride=1, dilate=False):
        norm_layer = self.norm_layer
        layers = []
        for _ in range(0, blocks):
            layers.append(block(n_inplane, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.layer1(x)
        x = self.fconv1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = self.fconv2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        x = x.view(-1, 1152)

        u = self.fc1(x)

        x = self.fc2(u)
        x = torch.relu(x)
        x = x.view(-1, 128, 3, 3)

        x = self.deconv1(x)
        x = torch.relu(x)

        x = self.deconv2(x)
        x = torch.relu(x)

        x = self.deconv3(x)
        return x, u
