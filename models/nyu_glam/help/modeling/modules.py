"""
Module that contains networks for MIL family
"""
import numpy as np
import torch
import torch.nn as nn

import src.modeling.common_functions as common_functions
import src.modeling.resnet_pytorch as resnet_pytorch
from src.modeling import layers

resnet_block_dict = {"basic_block_2by2": resnet_pytorch.BasicBlock2by2,
                     "normal": resnet_pytorch.BasicBlock,
                     "bottleneck": resnet_pytorch.Bottleneck}


class ResNetV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2.
    """

    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        # Diff: padding=SAME vs. padding=0
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        block = self._resolve_block(block_fn)
        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        self.final_bn = nn.BatchNorm2d(
            # current_num_filters // growth_factor
            current_num_filters // growth_factor * block.expansion
        )
        self.relu = nn.ReLU()
        self.initialize()

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.growth_factor = growth_factor
        self.block = block

    def forward(self, x, return_intermediate=False):
        intermediate = []
        h = self.first_conv(x)
        h = self.first_pool(h)

        if return_intermediate:
            intermediate.append(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
            if return_intermediate:
                intermediate.append(h)

        h = self.final_bn(h)
        h = self.relu(h)

        if return_intermediate:
            return h, intermediate
        else:
            return h

    @classmethod
    def _resolve_block(cls, block_fn):
        if block_fn == "normal":
            return layers.BasicBlockV2
        elif block_fn == "bottleneck":
            return layers.BottleneckV2
        else:
            raise KeyError(block_fn)

    def _make_layer(self, block, planes, blocks, stride=1):
        # downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            # nn.BatchNorm2d(planes * block.expansion),
        )

        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)

    def initialize(self):
        for m in self.modules():
            self._layer_init(m)

    @classmethod
    def _layer_init(cls, m):
        if isinstance(m, nn.Conv2d):
            # From original
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @classmethod
    def from_parameters(cls, parameters):
        return cls(
            input_channels=1,
            num_filters=parameters["num_filters"],
            first_layer_kernel_size=parameters["first_layer_kernel_size"],
            first_layer_conv_stride=parameters["first_layer_conv_stride"],
            first_layer_padding=parameters.get("first_layer_padding", 0),
            blocks_per_layer_list=parameters["blocks_per_layer_list"],
            block_strides_list=parameters["block_strides_list"],
            block_fn=parameters["block_fn"],
            first_pool_size=parameters["first_pool_size"],
            first_pool_stride=parameters["first_pool_stride"],
            first_pool_padding=parameters.get("first_pool_padding", 0),
            growth_factor=parameters.get("growth_factor", 2)
        )


class ResNetlocal(nn.Module):
    """
    Class that represents a ResNet with classifier sequence removed.
    """

    def __init__(self, initial_filters, block, layers,
                 input_channels=1, skip_pooling=True, localHR=True):

        self.inplanes = initial_filters
        self.num_layers = len(layers)
        self.skip_pooling = skip_pooling
        super(ResNetlocal, self).__init__()

        # so the first layer would still be the 7*7 with stride 2  input 512*512
        self.conv1 = nn.Conv2d(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.relu = nn.ReLU(inplace=True)

        if not self.skip_pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual sequence
        for i in range(self.num_layers):
            num_filters = initial_filters * pow(2, i)
            if localHR:
                num_stride = 1
            else:
                num_stride = (1 if i == 0 else 2)
            # num_stride = (1 if i == 0 else 2)
            setattr(self, 'layer{0}'.format(i + 1), self._make_layer(block, num_filters, layers[i], stride=num_stride))
        self.num_filter_last_seq = initial_filters * pow(2, self.num_layers - 1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # first sequence
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.skip_pooling:
            x = self.maxpool(x)

        # residual sequences
        for i in range(self.num_layers):
            x = getattr(self, 'layer{0}'.format(i + 1))(x)
        return x


class DownSampleNetworkResNetV2(ResNetV2):
    """
    Downsampling using ResNet V2.
    First conv is 7*7, stride 2, padding 3, cut 1/2 resolution.
    First pooling is skipped (kernel_size=1*1).
    """

    def __init__(self, parameters):
        super(DownSampleNetworkResNetV2, self).__init__(
            1, 16,
            # first conv
            first_layer_kernel_size=(7, 7), first_layer_conv_stride=2, first_layer_padding=3,
            # skip pooling
            first_pool_size=3, first_pool_stride=2, first_pool_padding=0,
            # architecture
            blocks_per_layer_list=[2, 2, 2, 2, 2],
            block_strides_list=[1, 2, 2, 2, 2],
            block_fn="normal",
            growth_factor=2)

    def forward(self, x):
        x_last, h_list = super(DownSampleNetworkResNetV2, self).forward(x, return_intermediate=True)
        # h_list = [after_first_pool, h1, h2, ..., before_last_bn_relu]
        all_level = h_list[1:-1] + [x_last]
        return all_level


class PostProcessingStandard(nn.Module):
    """
    Unit in Global Network that takes in x_out and produces CAM.
    """

    def __init__(self, parameters):
        super(PostProcessingStandard, self).__init__()
        # map all filters to output classes
        # need to add one more class for background in multilabel mode
        number_class = 2

        self.gn_conv_1 = nn.Conv2d(int(256 / 4),
                                   number_class,
                                   (1, 1), bias=False)
        self.gn_conv_2 = nn.Conv2d(int(256 / 2),
                                   number_class,
                                   (1, 1), bias=False)
        self.gn_conv_last = nn.Conv2d(256,
                                      number_class,
                                      (1, 1), bias=False)

        # use to combine three feature map
        self.combine = nn.Conv2d(number_class * 3, number_class, (1, 1), bias=False)

    def forward(self, x_out):
        # the third last feature map size
        _, _, h_h, w_h = x_out[-3].size()

        out_1 = self.gn_conv_1(x_out[-3])

        out_2 = self.gn_conv_2(x_out[-2])

        out_3 = self.gn_conv_last(x_out[-1])

        x_layer1 = out_1
        x_layer1 = torch.sigmoid(x_layer1)

        x_layer2 = common_functions.resize_pytorch_image(out_2, (h_h, w_h), no_grad=False)
        x_layer2 = torch.sigmoid(x_layer2)

        x_layer3 = common_functions.resize_pytorch_image(out_3, (h_h, w_h), no_grad=False)
        x_layer3 = torch.sigmoid(x_layer3)

        out = (x_layer1 + x_layer2 + x_layer3) / 3

        return x_layer1, x_layer2, x_layer3, out


class GlobalNetworkGeneric(nn.Module):
    """
    Global Network Module.
    A generic one that divides the logic flow into downsampling, upsampling, and postprocessing.
    """

    def __init__(self, parameters):
        super(GlobalNetworkGeneric, self).__init__()
        # downsampling-branch
        # self.downsampling_branch = parameters["downsample_network"](parameters)
        self.downsampling_branch = DownSampleNetworkResNetV2(parameters)
        # post-processing
        # self.postprocess_module = parameters["postprocess_network"](parameters)
        self.postprocess_module = PostProcessingStandard(parameters)
        self.parameters = parameters

    def forward(self, x):
        # retrieve results from downsampling network at all 4 levels
        res_all_levels = self.downsampling_branch.forward(x)
        # feed into postprocessing network
        (x_layer1, x_layer2, x_layer3, cam) = self.postprocess_module.forward(res_all_levels)
        return res_all_levels[-1], (x_layer1, x_layer2, x_layer3, cam)


class TopKAggregator:
    """
    An aggregator that uses the CAM to compute the y_cam.
    Uses the sum of topK values.
    """

    def __init__(self, parameters):
        self.percent_k = 0.10 if "percent_k" not in parameters else parameters["percent_k"]

    def forward(self, cam):
        batch_size, num_class, W, H = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_k = int(round(W * H * self.percent_k))
        selected_area = cam_flatten.topk(top_k, dim=2)[0]
        return selected_area.mean(dim=2)


class TopKAggregatorMultiPatch:
    """
    An aggregator that uses the CAM to compute the y_cam.
    Uses the sum of topK values.
    """

    def __init__(self, parameters):
        self.percent_k = 0.20 if "percent_k" not in parameters else parameters["percent_k"]

    def forward(self, cam):
        batch_size, num_class, num_patch, H, W = cam.size()  # (batch,2, num_patch, h, w)
        cam = cam.contiguous().view(batch_size, num_class, -1)
        cam_flatten = cam.view(batch_size, num_class, -1)  # (batch,2, -1)
        top_k = int(round(W * H * num_patch * self.percent_k))
        selected_area = cam_flatten.topk(top_k, dim=2)[0]
        return selected_area.mean(dim=2)


class TopKMIL:
    """
    An aggregator that uses the CAM to aggregate the patch-level prediction.
    Uses the sum of topK values.
    """

    def __init__(self, parameters):
        self.percent_k = 0.20 if "topk_mil" not in parameters else parameters["topk_mil"]

    def forward(self, preds):
        batch_size, num_patch, num_class = preds.size()
        preds = preds.permute(0, 2, 1)
        # cam_flatten = preds.view(batch_size, num_class, -1)
        top_k = int(round(num_patch * self.percent_k))
        selected_area = preds.topk(top_k, dim=2)[0]
        # return torch.sigmoid(selected_area.mean(dim=2))
        return selected_area.mean(dim=2)


class RegionProposalNetworkGreedy:
    """
    A Regional Proposal Network instance that computes the locations of the crops.
    Greedily selects crops with largest sums.
    """

    def __init__(self, parameters):
        self.crop_method = "upper_left"
        self.parameters = parameters
        self.num_crops_per_class = parameters["top_k_per_class"]
        self.crop_shape = parameters["crop_shape"]
        # self.pooling_logic = parameters["pooling_logic"]
        self.pooling_logic = 'avg'

    def forward(self, x_original, cam_size, h_small, device='cpu'):
        """
        Function that uses the low-res image to determine the position of the high-res crops.
        :param x_original: N, C, H, W pytorch tensor
        :param cam_size: (h, w)
        :param h_small: N, C, h_h, w_h pytorch tensor
        :param device: cpu or gpu
        :return: N, num_classes*k, 2 numpy matrix; returned coordinates are corresponding to x_small
        """
        # retrieve parameters
        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()

        # make sure that the size of h_small == size of cam_size
        assert h_h == h, "h_h!=h"
        assert w_h == w, "w_h!=w"

        # adjust crop_shape since crop shape is based on the original image
        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

        # greedily find the box with max sum of weights
        current_images = h_small
        all_max_position = []
        # if self.parameters["combined_channel"]:
        max_vals = current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)
        for _ in range(self.num_crops_per_class):
            max_pos = common_functions.get_max_window(current_images, crop_shape_adjusted, self.pooling_logic)
            all_max_position.append(max_pos)
            mask = common_functions.generate_mask_uplft(current_images, crop_shape_adjusted, max_pos, device=device)
            current_images = current_images * mask
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()


class DetectionNetworkResNetv3(nn.Module):
    """
    A Detection Network unit instance that takes a crop and computes its hidden representation.
    Uses ResNet.
    """

    def __init__(self, parameters):
        super(DetectionNetworkResNetv3, self).__init__()
        # resnet 34 based
        self.dn_resnet = ResNetlocal(64, resnet_pytorch.BasicBlock, [3, 4, 6, 3], 3, False, True)

        self.gn_conv_last = nn.Conv2d(512, 2, (1, 1), bias=False)

        self.Detectpooling = TopKAggregatorMultiPatch({'percent_k': parameters['detection_pooling_percent_k']})

        self.parameters = parameters

    def forward(self, x_crop):
        """
        Function that takes in a single crop and returns the hidden representation.
        :param x_crop: (N,C,h,w)
        :return: hidden representation.
        """

        last_featuremap = self.dn_resnet(x_crop.expand(-1, 3, -1, -1))  # the local level CAM, now use the pretrained local branch

        num_patch = self.parameters['top_k_per_class']

        CAM = torch.sigmoid(self.gn_conv_last(last_featuremap))  # (batch*num_patch, 2, h, w)
        raw_CAM = CAM
        batch_mul_num, _, h, w = CAM.size()

        batch = int(batch_mul_num / num_patch)

        CAM = CAM.view(batch, num_patch, 2, h, w)  # (batch, num_patch,2, h, w)
        CAM = CAM.permute(0, 2, 1, 3, 4)  # (batch,2, num_patch, h, w)
        # global average pooling
        # res = CAM.mean(dim=2).mean(dim=2)
        res = self.Detectpooling.forward(CAM)  # the class prediction result of the 2 dimension class vector (batch,2)
        return raw_CAM, res
