"""
Module that includes common functions that could be shared across different files
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


def random_rotation_scale_patch(img_tensor, rotation_range, scale_limits):
    """
    Function that randomly rotates and scales an image.
    """
    compos_func = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomAffine(rotation_range, scale=scale_limits),
                                      transforms.ToTensor()])
    orgin_max = img_tensor.max()
    orgin_min = img_tensor.min()
    standarized_img = (img_tensor - orgin_min) / (orgin_max - orgin_min)
    res = compos_func(standarized_img)
    rescale_img = res * (orgin_max - orgin_min) + orgin_min
    return rescale_img


def make_sure_in_range(val, min_val, max_val):
    """
    Function that makes sure that min < val < max; otherwise returns the limit value.
    """
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


def loc_normalize(val, constant):
    """
    Function that normalizes a value by y = (x-c)/c.
    """
    return (val - constant) / constant


def build_grid(img_size, crop_size):
    # calculate the scale
    # F.grid_sample assumes inputs in [-1, 1]
    H, W = img_size
    h, w = crop_size
    h_delta = (h - 1) / (H / 2)
    w_delta = (w - 1) / (W / 2)
    # create the 2d linspace from 0 to corresponding length
    # need to +1 to correct the numerical issue
    xv, yv = torch.meshgrid([torch.linspace(0, h_delta, steps=h),
                             torch.linspace(0, w_delta, steps=w)])
    # concat the results
    xv.unsqueeze_(2)
    yv.unsqueeze_(2)

    # set cuda
    if torch.cuda.is_available():
        return torch.cat([yv, xv], dim=2).cuda()
    else:
        return torch.cat([yv, xv], dim=2)


def crop_minibatch_pytorch(original_img_pytorch, crop_shape, crop_position, method="center"):
    """
    Function that takes a crop on the original image.
    It uses PyTorch and performs batch operation. Very fast!
    :param original_img_pytorch: (N,C,H,W) PyTorch Tensor
    :param crop_shape: (h, w) integer tuple
    :param crop_position: (N, K, 2) numpy ndarray, K is # of class - 1 (background class) *  # of crops per class
    :param method: supported in ["center", "upper_left"]
    :return: (N, K, h, w) PyTorch Tensor
    """
    # fill the original image into GPU
    if torch.cuda.is_available():
        numpy2tensor = lambda x: torch.from_numpy(x).cuda()
    else:
        numpy2tensor = lambda x: torch.from_numpy(x)

    # retrieve inputs
    _, _, H, W = original_img_pytorch.shape
    half_H = H / 2
    half_W = W / 2
    x_delta, y_delta = crop_shape
    N, K, D = crop_position.shape
    assert D == 2, "The third dimension of crop_position is {0}, should be 2".format(D)
    # crop_position_stacked = crop_position.reshape(N*K, D)

    # locate the four corners and normalize (-1 = upper left, 1 = bottom right)
    if method == "center":
        left_x = loc_normalize(crop_position[:, :, 0] - x_delta / 2, half_H)
        lower_y = loc_normalize(crop_position[:, :, 1] - y_delta / 2, half_W)
    elif method == "upper_left":
        left_x = numpy2tensor(loc_normalize(crop_position[:, :, 0], half_H)).unsqueeze(2)
        lower_y = numpy2tensor(loc_normalize(crop_position[:, :, 1], half_W)).unsqueeze(2)
    else:
        raise NotImplementedError("{0} method is not implemented.".format(method))
    # calculate the corresponding grid
    prototype_grid = build_grid((H, W), crop_shape)
    prototype_grid.unsqueeze_(0)
    # repeat the prototype grid for tensor of shape (N, h, w, 2)
    zero_grid = prototype_grid.repeat(N, 1, 1, 1)
    # iterate over each of the k crops for each image
    ouputs = []
    for i in range(K):
        # adjust the offset
        # note that the order for the grid is y,x somehow
        offset_mat = torch.cat([lower_y[:, i, :], left_x[:, i, :]], dim=1).unsqueeze_(1).unsqueeze_(1)
        grid = zero_grid.float() + offset_mat.float()
        crops = F.grid_sample(original_img_pytorch, grid)
        ouputs.append(crops)
    return torch.cat(ouputs, dim=1)


def crop_pytorch(original_img_pytorch, crop_shape, crop_position, out,
                 method="center", background_val="min"):
    """
    Function that take a crop on the original image.
    Use PyTorch to do this.
    :param original_img_pytorch: H,W PyTorch Tensor
    :param crop_shape: (h, w) integer tuple
    :param method: supported in ["center", "upper_left"]
    :param background_val:
    :return: (N, K, h, w) PyTorch Tensor
    """
    # retrieve inputs
    H, W = original_img_pytorch.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    # locate the four corners
    if method == "center":
        left_x = int(np.round(crop_x - x_delta / 2))
        right_x = int(np.round(crop_x + x_delta / 2))
        lower_y = int(np.round(crop_y - y_delta / 2))
        upper_y = int(np.round(crop_y + y_delta / 2))
    elif method == "upper_left":
        left_x = int(np.round(crop_x))
        right_x = int(np.round(crop_x + x_delta))
        lower_y = int(np.round(crop_y))
        upper_y = int(np.round(crop_y + y_delta))

    # make sure that the crops are in range
    left_x = make_sure_in_range(left_x, 0, H)
    right_x = make_sure_in_range(right_x, 0, H)
    lower_y = make_sure_in_range(lower_y, 0, W)
    upper_y = make_sure_in_range(upper_y, 0, W)

    # somehow background is normalized to this number
    if background_val == "min":
        out[:, :] = original_img_pytorch.min()
    else:
        out[:, :] = background_val
    real_x_delta = right_x - left_x
    real_y_delta = upper_y - lower_y
    origin_x = crop_shape[0] - real_x_delta
    origin_y = crop_shape[1] - real_y_delta
    out[origin_x:, origin_y:] = original_img_pytorch[left_x:right_x, lower_y:upper_y]


def crop_pytorch_multichannel(original_img_pytorch, crop_shape, crop_position, out,
                              method="center", background_val="min"):
    """
    Function that take a crop on the original image.
    Use PyTorch to do this.
    :param original_img_pytorch: C,H,W PyTorch Tensor
    :param crop_shape: (h, w) integer tuple
    :param method: supported in ["center", "upper_left"]
    """
    # retrieve inputs
    C, H, W = original_img_pytorch.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    # locate the four corners
    if method == "center":
        left_x = int(np.round(crop_x - x_delta / 2))
        right_x = int(np.round(crop_x + x_delta / 2))
        lower_y = int(np.round(crop_y - y_delta / 2))
        upper_y = int(np.round(crop_y + y_delta / 2))
    elif method == "upper_left":
        left_x = int(np.round(crop_x))
        right_x = int(np.round(crop_x + x_delta))
        lower_y = int(np.round(crop_y))
        upper_y = int(np.round(crop_y + y_delta))

    # make sure that the crops are in range
    left_x = make_sure_in_range(left_x, 0, H)
    right_x = make_sure_in_range(right_x, 0, H)
    lower_y = make_sure_in_range(lower_y, 0, W)
    upper_y = make_sure_in_range(upper_y, 0, W)

    # somehow background is normalized to this number
    if background_val == "min":
        out[:, :, :] = original_img_pytorch.min()
    else:
        out[:, :, :] = background_val
    real_x_delta = right_x - left_x
    real_y_delta = upper_y - lower_y
    origin_x = crop_shape[0] - real_x_delta
    origin_y = crop_shape[1] - real_y_delta
    out[:, origin_x:, origin_y:] = original_img_pytorch[:, left_x:right_x, lower_y:upper_y]


def crop(original_img, crop_shape, crop_position, method="center",
         in_place=False, background_val="min"):
    """
    Function that take a crop on the original image.
    This function must staty in numpy since original_img should not be loaded into Pytorch during the network time.
    original_img is large and would consume lots of GPU memory.
    :param original_img:
    :param crop_shape:
    :param crop_position:
    :param method: supported in ["center", "upper_left"]
    :param in_place: if in_place, the effective pixels in the crop will be flagged (1.0) in the original_img
    :param background_val:
    :return:
    """
    # retrieve inputs
    I, J = original_img.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    # locate the four corners
    if method == "center":
        left_x = int(np.round(crop_x - x_delta / 2))
        right_x = int(np.round(crop_x + x_delta / 2))
        lower_y = int(np.round(crop_y - y_delta / 2))
        upper_y = int(np.round(crop_y + y_delta / 2))
    elif method == "upper_left":
        left_x = int(np.round(crop_x))
        right_x = int(np.round(crop_x + x_delta))
        lower_y = int(np.round(crop_y))
        upper_y = int(np.round(crop_y + y_delta))

    # make sure that the crops are in range
    left_x = make_sure_in_range(left_x, 0, I)
    right_x = make_sure_in_range(right_x, 0, I)
    lower_y = make_sure_in_range(lower_y, 0, J)
    upper_y = make_sure_in_range(upper_y, 0, J)

    # if in_place, flag the original inputs
    if in_place:
        original_img[left_x:right_x, lower_y:upper_y] = 1.0
    # else make the new matrix
    else:
        # somehow background is normalized to this number
        if background_val == "min":
            output = np.ones(crop_shape) * np.min(original_img)
        else:
            output = np.ones(crop_shape) * background_val
        real_x_delta = right_x - left_x
        real_y_delta = upper_y - lower_y
        origin_x = crop_shape[0] - real_x_delta
        origin_y = crop_shape[1] - real_y_delta
        output[origin_x:, origin_y:] = original_img[left_x:right_x, lower_y:upper_y]
        return output


def get_max_window(input_image, window_shape, pooling_logic="avg"):
    """
    Function that makes a sliding window of size window_shape over the
    input_image and return the UPPER_LEFT corner index with max sum.
    :param input_image: N*C*H*W
    :param window_shape: h*w
    :return: N*C*2
    """
    N, C, H, W = input_image.size()
    if pooling_logic == "avg":
        # use average pooling to locate the window sums
        pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)

    elif pooling_logic == "LPPool1.5":
        # Applies a 2D power-average pooling over an input signal composed of several input planes.
        pool_map = torch.nn.functional.lp_pool2d(input_image, norm_type=1.5, kernel_size=window_shape, stride=1)

    elif pooling_logic == "LPPool2":
        # Applies a 2D power-average pooling over an input signal composed of several input planes.
        pool_map = torch.nn.functional.lp_pool2d(input_image, norm_type=2.0, kernel_size=window_shape, stride=1)

    elif pooling_logic == "LPPool3":
        # Applies a 2D power-average pooling over an input signal composed of several input planes.
        pool_map = torch.nn.functional.lp_pool2d(input_image, norm_type=3.0, kernel_size=window_shape, stride=1)

    elif pooling_logic == "LPPool5":
        # Applies a 2D power-average pooling over an input signal composed of several input planes.
        pool_map = torch.nn.functional.lp_pool2d(input_image, norm_type=5.0, kernel_size=window_shape, stride=1)

    elif pooling_logic == "LPPool10":
        # Applies a 2D power-average pooling over an input signal composed of several input planes.
        pool_map = torch.nn.functional.lp_pool2d(input_image, norm_type=10.0, kernel_size=window_shape, stride=1)

    elif pooling_logic == "LPPool20":
        # Applies a 2D power-average pooling over an input signal composed of several input planes.
        pool_map = torch.nn.functional.lp_pool2d(input_image, norm_type=20.0, kernel_size=window_shape, stride=1)

    elif pooling_logic in ["std", "avg_entropy"]:
        # create sliding windows
        output_size = (H - window_shape[0] + 1, W - window_shape[1] + 1)
        sliding_windows = F.unfold(input_image, kernel_size=window_shape).view(N, C, window_shape[0] * window_shape[1], -1)
        # apply aggregation function on each sliding windows
        if pooling_logic == "std":
            agg_res = sliding_windows.std(dim=2, keepdim=False)
        elif pooling_logic == "avg_entropy":
            agg_res = -sliding_windows * torch.log(sliding_windows) - (1 - sliding_windows) * torch.log(1 - sliding_windows)
            agg_res = agg_res.mean(dim=2, keepdim=False)
        # merge back
        pool_map = F.fold(agg_res, kernel_size=(1, 1), output_size=output_size)
    _, _, _, W_map = pool_map.size()
    # transform to linear and get the index of the max val locations
    _, max_linear_idx = torch.max(pool_map.view(N, C, -1), -1)
    # convert back to 2d index
    max_idx_x = max_linear_idx / W_map
    max_idx_y = max_linear_idx - max_idx_x * W_map
    # put together the 2d index
    upper_left_points = torch.cat([max_idx_x.unsqueeze(-1), max_idx_y.unsqueeze(-1)], dim=-1)
    return upper_left_points


def generate_mask_uplft(input_image, window_shape, upper_left_points, device='cpu'):
    """
    Function that generates mask that sets crops given upper_left corners to 0.
    """
    N, C, H, W = input_image.size()
    window_h, window_w = window_shape
    # get the positions of masks
    mask_x_min = upper_left_points[:, :, 0]
    mask_x_max = upper_left_points[:, :, 0] + window_h
    mask_y_min = upper_left_points[:, :, 1]
    mask_y_max = upper_left_points[:, :, 1] + window_w
    # generate masks
    if device.lower() == 'gpu':
        mask_x = Variable(torch.arange(0, H).cuda().view(-1, 1).repeat(N, C, 1, W))
        mask_y = Variable(torch.arange(0, W).cuda().view(1, -1).repeat(N, C, H, 1))
    else:
        mask_x = Variable(torch.arange(0, H).view(-1, 1).repeat(N, C, 1, W))
        mask_y = Variable(torch.arange(0, W).view(1, -1).repeat(N, C, H, 1))
    # TODO: for the mismatched version of pytorch on skynet
    x_gt_min = mask_x >= mask_x_min.unsqueeze(-1).unsqueeze(-1)
    x_ls_max = mask_x < mask_x_max.unsqueeze(-1).unsqueeze(-1)
    y_gt_min = mask_y >= mask_y_min.unsqueeze(-1).unsqueeze(-1)
    y_ls_max = mask_y < mask_y_max.unsqueeze(-1).unsqueeze(-1)

    # since logic operation is not supported for variable
    # I used * for logic ANd
    selected_x = x_gt_min * x_ls_max
    selected_y = y_gt_min * y_ls_max
    selected = selected_x * selected_y
    mask = 1 - selected.float()
    return mask


def resize_numpy_image(original_images, target_shape):
    """
    Function that resizes and interpolates numpy images.
    :param original_images: (minibatch_size, I, J, 1)
    :param target_shape:
    :return:
    """
    # TODO: check if there is some quick bulk resizing
    minibatch_size = original_images.shape[0]
    output = np.zeros([minibatch_size, target_shape[0], target_shape[1], 1])
    for i in range(minibatch_size):
        # somehow, the resize shape argument for cv2 shall be reversed
        output[i, :, :, 0] = cv2.resize(original_images[i, :, :, :], (target_shape[1], target_shape[0]))
    return output


def resize_pytorch_image(original_images, target_shape, no_grad=True):
    """
    Function that uses pytorch to resize images by nearest interpolation.
    :param original_images: pytorch Variable with size (N,C,H,W)
    :param target_shape: int tuple (H_out, W_out)
    :return: pytorch Variable with size (N,C,H_out,W_out)
    """
    # Current on Skynet 0.4.0a0+200fb22
    # With env setting 0.4.1
    # TODO: use interpolate() when pytorch is updated to 0.4.1
    upsample_func = lambda x: F.interpolate(x, size=target_shape)
    # execute upsampling
    if no_grad:
        with torch.no_grad():
            return upsample_func(original_images)
    else:
        return upsample_func(original_images)


def pretty_print_matrix(matrix, H, W, precision=1):
    """
    Function that pretty prints a matrix.
    :param matrix:
    :return: None
    """
    output = ""
    for i in range(H):
        output += "{0} \t".format(i)
        for j in range(W):
            output += ("{0:." + str(precision) + "f} ").format(matrix[i, j])
        output += "\n"
    print(output)


def two_boxes_iou(boxA, boxB):
    """
    Function that calculates the IoU between two bounding boxes.
    :param boxA: [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    :param boxB: [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    :return:
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_crop_mask(loc, crop_shape, image_shape, method, indicator=True):
    """
    Function that generates the mask.
    :param loc:
    :param crop_shape:
    :param image_shape:
    :param method:
    :return:
    """
    crop_map = np.zeros(image_shape)
    for crop_loc in loc:
        # this is the indicator for point of crop
        if indicator:
            crop_map[int(crop_loc[0]), int(crop_loc[1])] = 999.0
        # fill in 1.0 in the cropped regions
        crop(crop_map, crop_shape, crop_loc, method=method, in_place=True)
    return crop_map


def percent_roi_covered(input_mask, gold_mask):
    """
    Function that calculates the % of ROI region covered by the mask.
    :param input_mask:
    :param gold_mask:
    :return: -1 if gold_mask doesn't have any ROI
    """
    # return NaN if gold_mask is not available
    if gold_mask is None:
        return np.NaN

    # sanity check
    assert np.max(input_mask) <= 1.0, "np.max(input_mask) > 1.0"
    assert np.min(input_mask) >= 0.0, "np.min(input_mask) < 0.0"
    assert np.max(gold_mask) <= 1.0, "np.max(gold_mask) > 1.0"
    assert np.min(gold_mask) >= 0.0, "np.min(gold_mask) < 0.0"

    # return -1 if the gold mask doesn't have any ROI
    if np.sum(gold_mask) == 0:
        return np.NaN
    # calculate the %
    output = np.sum(input_mask * gold_mask) / np.sum(gold_mask)
    # sanity check
    assert output <= 1.0, "output > 1.0"
    assert output >= 0.0, "output < 0.0"
    return output
