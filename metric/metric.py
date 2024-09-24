import torch
from torch import nn
import torch.utils.data

# import pyiqa
import numpy as np
# import lpips
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from loguru import logger
from util.calc_PSNR import calculate_psnr


def mae(input_tensor, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input_tensor, target)
    return output


# input_tensor and target tensor should be 0 ~ 1
class PSNR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.mse = nn.MSELoss()
        self.data_range = 1.0

    def forward(self, input_tensor, target):
        mse = torch.mean((input_tensor - target)**2, dim=[-3, -2, -1])
        return torch.mean(10. * torch.log10(self.data_range / mse)).item()

        # err = self.mse(input_tensor, target).item()
        # return 10 * np.log10((self.data_range ** 2) / err)
        # return peak_signal_noise_ratio(np.array(input_tensor.float().cpu()), np.array(target.float().cpu()), data_range=1)


# ssim_func = reference_pyiqa_metric_fun_generator('ssim')
class SSIM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_tensor, target):
        h, w = input_tensor.shape[-2:]
        input_tensor = input_tensor.view(-1, h, w)
        target = target.view(-1, h, w)
        return structural_similarity(input_tensor.float().cpu().numpy(), target.float().cpu().numpy(),
                                     channel_axis=0, data_range=1)


class packLPIPS(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_alex = lpips.LPIPS(net='alex').cuda()

    """
    Input of tensors are [0, 1]
    """
    def forward(self, input_tensor, target):
        with torch.no_grad():
            # convert to [-1, 1]
            tensor_a = (input_tensor - 0.5) * 2
            tensor_b = (target - 0.5) * 2
            result = self.loss_fn_alex(tensor_a, tensor_b)

            result_sum = 0
            if len(result.shape) == 0:
                return float(result)

            for i in result:
                result_sum += i

        return float(result_sum / len(result))


class NIQE(nn.Module):
    def __init__(self, is_reverse, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_metric = pyiqa.create_metric('niqe').cuda()
        self.is_reverse = is_reverse

    def forward(self, input_tensor, target):
        if len(input_tensor.shape) <= 3:
            reshape_input_tensor = reshape_tensor(input_tensor)
            # reshape_target = reshape_tensor(target)
        else:
            reshape_input_tensor = input_tensor
            # reshape_target = target

        result = self.method_metric(reshape_input_tensor)

        return result if not self.is_reverse else -result


def reference_pyiqa_metric_fun_generator(metric_name):
    def metric_fun(input_tensor, target):
        method_metric = pyiqa.create_metric(metric_name).cuda()
        if len(input_tensor.shape) <= 3:
            reshape_input_tensor = reshape_tensor(input_tensor)
            reshape_target = reshape_tensor(target)
        else:
            reshape_input_tensor = input_tensor
            reshape_target = target

        result = method_metric(reshape_input_tensor, reshape_target)

        result_sum = 0
        if len(result.shape) == 0:
            return result

        for i in result:
            result_sum += i

        return result_sum / len(result)

    return metric_fun


def reshape_tensor(input_tensor):
    with torch.no_grad():
        input_tensor = input_tensor.clamp_(0, 1).detach()
        # norm_input_tensor = input_tensor + torch.ones_like(input_tensor)
        # norm_input_tensor = norm_input_tensor / 2.0
    return input_tensor.unsqueeze(0)


class YPSNR(nn.Module):
    def __init__(self):
        super().__init__()
        logger.warning('PSNR is calculated only on Y channel!!!')

    def forward(self, input_tensor, target):
        input_tensor = torch.flatten(input_tensor, start_dim=0, end_dim=-4)
        target = torch.flatten(target, start_dim=0, end_dim=-4)
        b = target.shape[0]
        input_tensor_np = input_tensor.cpu().numpy()
        target_np = target.cpu().numpy()
        psnr_sum = 0
        for i in range(b):
            psnr_sum += calculate_psnr(input_tensor_np[i], target_np[i], crop_border=0, input_order='CHW', test_y_channel=True)
        return psnr_sum / b


class YSSIM(nn.Module):
    def __init__(self):
        super().__init__()
        logger.warning('SSIM is calculated only on Y channel!!!')
        self.ssim_fun = reference_pyiqa_metric_fun_generator('ssim')

    def forward(self, input_tensor, target):
        input_tensor = torch.flatten(input_tensor, start_dim=0, end_dim=-4)
        target = torch.flatten(target, start_dim=0, end_dim=-4)
        return self.ssim_fun(input_tensor, target)
        # return structural_similarity(np.array(torch.squeeze(input_tensor.float().cpu())), np.array(torch.squeeze(target.float().cpu())),
        #                              channel_axis=0, data_range=1, calculate_on_Y=True)
