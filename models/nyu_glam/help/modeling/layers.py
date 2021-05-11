import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution, importing from torchvision doesn't work for earlier versions"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape, activation="log_softmax"):
        super(OutputLayer, self).__init__()
        if not isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape]
        self.output_shape = output_shape
        self.flattened_output_shape = int(np.prod(output_shape))
        self.fc_layer = nn.Linear(in_features, self.flattened_output_shape)
        self.activation = activation

    def forward(self, x, use_activation=True):
        h = self.fc_layer(x)
        if len(self.output_shape) > 1:
            h = h.view(h.shape[0], *self.output_shape)
        if use_activation:
            if self.activation == "log_softmax":
                h = F.log_softmax(h, dim=-1)
            else:
                raise KeyError(self.activation)
        return h


class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckV2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.conv3(out)

        out += residual

        return out


class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        if not self.gaussian_noise_std or not self.training:
            return x

        return {
            "L-CC": self._add_gaussian_noise(x["L-CC"]),
            "L-MLO": self._add_gaussian_noise(x["L-MLO"]),
            "R-CC": self._add_gaussian_noise(x["R-CC"]),
            "R-MLO": self._add_gaussian_noise(x["R-MLO"]),
        }

    def _add_gaussian_noise(self, single_view):
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)


class AllViewsConvLayer(nn.Module):
    """Convolutional layers across all 4 views"""

    def __init__(self, in_channels, number_of_filters=32, filter_size=(3, 3), stride=(1, 1)):
        super(AllViewsConvLayer, self).__init__()
        self.cc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=number_of_filters,
            kernel_size=filter_size,
            stride=stride,
        )
        self.mlo = nn.Conv2d(
            in_channels=in_channels,
            out_channels=number_of_filters,
            kernel_size=filter_size,
            stride=stride,
        )

    def forward(self, x):
        return {
            "L-CC": F.relu(self.cc(x["L-CC"])),
            "L-MLO": F.relu(self.mlo(x["L-MLO"])),
            "R-CC": F.relu(self.cc(x["R-CC"])),
            "R-MLO": F.relu(self.mlo(x["R-MLO"])),
        }

    @property
    def ops(self):
        return {
            "CC": self.cc,
            "MLO": self.mlo,
        }


class AllViewsMaxPool(nn.Module):
    """Max-pool across all 4 views"""

    def __init__(self):
        super(AllViewsMaxPool, self).__init__()

    def forward(self, x, stride=(2, 2), padding=(0, 0)):
        return {
            "L-CC": F.max_pool2d(x["L-CC"], kernel_size=stride, stride=stride, padding=padding),
            "L-MLO": F.max_pool2d(x["L-MLO"], kernel_size=stride, stride=stride, padding=padding),
            "R-CC": F.max_pool2d(x["R-CC"], kernel_size=stride, stride=stride, padding=padding),
            "R-MLO": F.max_pool2d(x["R-MLO"], kernel_size=stride, stride=stride, padding=padding),
        }


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            view_name: self._avg_pool(view_tensor)
            for view_name, view_tensor in x.items()
        }

    @staticmethod
    def _avg_pool(single_view):
        n, c, h, w = single_view.size()
        return single_view.view(n, c, -1).mean(-1)


class AllViewsPad(nn.Module):
    """Pad tensor across all 4 views"""

    def __init__(self):
        super(AllViewsPad, self).__init__()

    def forward(self, x, pad):
        return {
            "L-CC": F.pad(x["L-CC"], pad),
            "L-MLO": F.pad(x["L-MLO"], pad),
            "R-CC": F.pad(x["R-CC"], pad),
            "R-MLO": F.pad(x["R-MLO"], pad),
        }
