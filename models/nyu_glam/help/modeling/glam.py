"""
Module that contains networks for MIL family
"""
import logging
import numpy as np
import torch
import torch.nn as nn

import src.modeling.common_functions as common_functions
import src.modeling.modules as m


class MILSingleImageModel(nn.Module):
    def __init__(self, parameters):
        super(MILSingleImageModel, self).__init__()

        self.cam_size = parameters["cam_size"]

        # save parameters
        self.experiment_parameters = parameters

        # construct networks
        # global network
        self.global_network = m.GlobalNetworkGeneric(self.experiment_parameters)

        # aggregator
        self.aggregator = m.TopKAggregator(self.experiment_parameters)

        # region proposal network
        self.region_proposal_network = m.RegionProposalNetworkGreedy(self.experiment_parameters)

        # local module
        self.detection_network = m.DetectionNetworkResNetv3(self.experiment_parameters)

        if (parameters["device_type"] == "gpu") and torch.has_cudnn:
            device = torch.device("cuda:{}".format(parameters["gpu_number"]))
            self.device_str = 'gpu'
        else:
            device = torch.device("cpu")
            self.device_str = 'cpu'
        self.device = device

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original.
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()
        n, k, _ = crops_x_small.shape

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _combine_crop_cam(self, camlocal, crop_positions, x_original_pytorch):

        # camlocal: batch, numcrop, 2, h, w
        # the size of original image

        _, _, _, h_local, w_local = camlocal.size()
        _, _, H, W = x_original_pytorch.size()

        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        batch_size, num_crops, _ = crop_positions.shape

        # the combined cam will be interpolated to the same size as the input image size

        cam_combined = torch.zeros((batch_size, 2, H, W))

        for i in range(batch_size):
            # the ith batch
            for j in range(num_crops):
                # the ith crop
                crop_x, crop_y = int(crop_positions[i, j, 0]), int(crop_positions[i, j, 1])
                cam_resize = common_functions.resize_pytorch_image(camlocal[i, j:j + 1, :, :, :], (crop_h, crop_w), no_grad=True)
                cam_combined[i, :, crop_x:(crop_x + crop_h), crop_y:(crop_y + crop_w)] = cam_resize[0, :, :, :]

        return cam_combined

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method, device):
        """
        Function that takes in the original image and cropping position and returns the crops.
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :param device: either cpu or gpu
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        for i in range(batch_size):
            for j in range(num_crops):
                common_functions.crop_pytorch(x_original_pytorch[i, 0, :, :],
                                              self.experiment_parameters["crop_shape"],
                                              crop_positions[i, j, :],
                                              output[i, j, :, :],
                                              method=crop_method)
        if device == 'gpu':
            output = output.cuda()
        return output

    def _retrieve_crop_featuremap(self, feature_map, crop_positions, crop_size):
        """
        Function that takes in feature map and cropping position and returns the crops.
        Gradients are perserved!
        :param feature_map: PyTorch Tensor array (N,C,H,W)
        :param crop_positions: (N,k,2)
        :return: N,k,C,h,w
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = crop_size

        all_featuremap_crops = []
        for i in range(batch_size):
            current_featuremap_crops = []
            for j in range(num_crops):
                current_crop = feature_map[i, :, crop_positions[i, j, 0]:crop_positions[i, j, 0] + crop_h,
                               crop_positions[i, j, 1]:crop_positions[i, j, 1] + crop_w]
                current_crop = current_crop.unsqueeze(0).unsqueeze(0)
                current_featuremap_crops.append(current_crop)
            all_featuremap_crops.append(torch.cat(current_featuremap_crops, dim=1))
        return torch.cat(all_featuremap_crops, dim=0)

    def forward(self, x_original):
        """
        :param x_original: N,H,W,C numpy matrix
        :return:
        """
        N, _, H, W = x_original.size()

        # global network: x_original -> class activation map
        # class activation map should have the same dimension with x_original
        last_feature_map, (x_layer1, x_layer2, x_layer3, out) = self.global_network.forward(x_original)
        self.downsampled_map = out

        # self.class_activation_map_global = out
        self.saliency_map_global = 0.2 * x_layer3 + 0.6 * x_layer2 + 0.2 * x_layer1

        y_cam_logits_1 = self.aggregator.forward(x_layer1)
        y_cam_logits_2 = self.aggregator.forward(x_layer2)
        y_cam_logits_3 = self.aggregator.forward(x_layer3)
        y_cam_logits_4 = self.aggregator.forward(out)

        y_cam_logits = (y_cam_logits_1, y_cam_logits_2, y_cam_logits_3, y_cam_logits_4)

        self.y_cam_logits = y_cam_logits
        self.y_cam_pred = y_cam_logits

        # add some intermediate layers to get the score for individual layers
        self.x_layer1 = x_layer1
        self.x_layer2 = x_layer2
        self.x_layer3 = x_layer3
        self.y_global = (y_cam_logits_1 + y_cam_logits_2 + y_cam_logits_3) / 3

        # select the patch location
        patch_select_cam = self.saliency_map_global
        small_x_locations = self.region_proposal_network.forward(x_original, self.cam_size, patch_select_cam, device=self.device_str)

        # patch
        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        # cropping retriever
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.region_proposal_network.crop_method, device=self.device_str)
        self.patches = crops_variable.data.cpu().numpy()

        # local module to form the local segmentation map
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        # batch * num_crops, 1, H, W

        local_CAMs, _ = self.detection_network.forward(crops_variable)
        _, _, h_local, w_local = local_CAMs.size()
        self.raw_local_CAM = local_CAMs  # local_CAMs: batch*num_crops, 2 , h_local, w_local

        local_CAMs = local_CAMs.view(batch_size, num_crops, 2, h_local, w_local)  # local_CAMs: batch, num_crops, 2, h_local, w_local

        # the CAM for the local module
        combined_cam = self._combine_crop_cam(local_CAMs, self.patch_locations, x_original)  # batch， 2， H，W
        self.saliency_map_local = combined_cam.to(self.device)

        self.patch_pred = None

        # to combine the global and local saliency maps
        _, _, h_h, w_h = self.saliency_map_local.size()
        map_to_fuse = self.saliency_map_global
        map_to_fuse = common_functions.resize_pytorch_image(map_to_fuse, (h_h, w_h), no_grad=False)
        self.saliency_map = (self.saliency_map_local + map_to_fuse) / 2

        return self.y_global
