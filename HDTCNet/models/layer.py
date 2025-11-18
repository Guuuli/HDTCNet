import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from functools import partial

import pywt
import pywt.data
from einops.layers.torch import Rearrange



class Group_Linear(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()

        self.out_channels = out_channels
        self.groups = groups

        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.group_mlp.reset_parameters()


    def forward(self, x: Tensor, is_reshape: False):
        """
        Args:
            x (Tensor): [B, C, N, F] (if not is_reshape), [B, C, G, N, F//G] (if is_reshape)
        """
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups

        if not is_reshape:
            # x: [B, C_in, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C_in, N, F//G]
        x = x.transpose(1, 2).reshape(B, G*C, N, -1)

        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)

        # out: [B, C_out, G, N, F//G]
        return out


class Dense_TimeDiffPool2d(nn.Module):

    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()

        # TODO: add Normalization
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))

        self.re_param = Parameter(Tensor(kern_size, 1))

    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')


    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)

        # s: [ N^(l+1), N^l, 1, K ]
        # s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # TODO: fully-connect, how to decrease time complexity

        return out


# class Conv2d_cd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding,
#                  stride=1, dilation=1, groups=1, bias=False, theta=1.0):
#         super(Conv2d_cd, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding),
#                               dilation=dilation, groups=groups, bias=bias)
#         self.theta = theta

#     def get_weight(self,kernel_size):
#         conv_weight = self.conv.weight  # 获取卷积层的权重，形状为 (out_channels, in_channels, 1, 9)
#         conv_shape = conv_weight.shape

#         # 将权重变形为 (out_channels, in_channels, 9)
#         conv_weight = conv_weight.view(conv_shape[0], conv_shape[1], conv_shape[3])

#         # 创建一个新的权重矩阵，形状与原始权重矩阵相同
#         conv_weight_cd = torch.cuda.FloatTensor(conv_shape).fill_(0)

#         # 设置中心差分的权重矩阵
#         conv_weight_cd[:, :, 0, :] = conv_weight[:, :, :]
#         conv_weight_cd[:, :, 0, (kernel_size-1)//2] = conv_weight[:, :, (kernel_size-1)//2] - conv_weight.sum(2)

#         return conv_weight_cd, self.conv.bias


# class Conv2d_ad(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding,
#                  stride=1, dilation=1, groups=1, bias=False, theta=1.0):
#         super(Conv2d_ad, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.theta = theta

#     def get_weight(self, kernel_size):
#         conv_weight = self.conv.weight  # 获取卷积层的权重，形状为 (out_channels, in_channels, 1, kernel_size)
#         conv_shape = conv_weight.shape

#         # 将权重变形为 (out_channels, in_channels, kernel_size)
#         conv_weight = conv_weight.view(conv_shape[0], conv_shape[1], conv_shape[3])

#         # 创建一个新的权重矩阵，形状与原始权重矩阵相同
#         conv_weight_ad = torch.cuda.FloatTensor(conv_shape).fill_(0)

#         # 设置角差分的权重矩阵
#         center_index = kernel_size // 2
#         conv_weight_ad[:, :, 0, :] = conv_weight[:, :, :] - self.theta * conv_weight[:, :, [center_index] + list(
#             range(center_index)) + list(range(center_index + 1, kernel_size))]

#         # 将权重变形回原始形状
#         conv_weight_ad = conv_weight_ad.view(conv_shape)

#         return conv_weight_ad, self.conv.bias


# class Conv2d_hd(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding,
#                  stride=1, dilation=1, groups=1, bias=False):

#         super(Conv2d_hd, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation, groups=groups, bias=bias)

#     def get_weight(self, kernel_size):
#         conv_weight = self.conv.weight  # 获取卷积层的权重，形状为 (out_channels, in_channels, 1, 9)
#         conv_shape = conv_weight.shape
#         # 创建一个新的权重矩阵，形状与原始权重矩阵相同
#         conv_weight_hd = torch.cuda.FloatTensor(conv_shape).fill_(0)
#         # 设置水平差分的权重矩阵
#         conv_weight_hd[:, :, 0, 0] = conv_weight[:, :, 0, 0]    # 最左边的权重
#         conv_weight_hd[:, :, 0, kernel_size - 1] = -conv_weight[:, :, 0, 0]   # 最右边的权重取反
#         return conv_weight_hd, self.conv.bias


# class DEConv(nn.Module):
#     def __init__(self, in_dim, out_dim, kernel_size, padding):
#         super(DEConv, self).__init__()
#         self.conv1_1 = Conv2d_hd(in_dim, out_dim, kernel_size, padding, bias=True)
#         self.conv1_2 = Conv2d_cd(in_dim, out_dim, kernel_size, padding, bias=True)
#         self.conv1_3 = Conv2d_ad(in_dim, out_dim, kernel_size, padding, bias=True)
#         self.conv1_4 = nn.Conv2d(in_dim, out_dim, (1, kernel_size), (0, padding), bias=True)

#     def forward(self, x, kernel_size, padding):
#         w1, b1 = self.conv1_1.get_weight(kernel_size)
#         w2, b2 = self.conv1_2.get_weight(kernel_size)
#         # w3, b3 = self.conv1_3.get_weight(kernel_size)
#         w4, b4 = self.conv1_4.weight, self.conv1_4.bias

#         w = w1 + w2 + w4
#         b = b1 + b2 + b4
#         res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=(0,padding), groups=1)

#         return res


class Conv2d_cd_k3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd_k3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)

        # conv_weight_cd = conv_weight - conv_weight[:, :, [4, 4, 4, 4, 4, 4, 4, 4, 4]]

        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_cd_k5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd_k5, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 12] = conv_weight[:, :, 12] - conv_weight[:, :, :].sum(2)

        # indices = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
        # conv_weight_cd = conv_weight - conv_weight[:, :, indices]

        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_cd_k9(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 padding=4, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd_k9, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 9 * 9).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 40] = conv_weight[:, :, 40] - conv_weight[:, :, :].sum(2)

        # indices = [
        #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        # ]
        # conv_weight_cd = conv_weight - conv_weight[:, :, indices]

        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias



class Conv2d_ad_k3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_ad_k3, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias



class Conv2d_ad_k5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_ad_k5, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        indices = [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, indices]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias
    

class Conv2d_ad_k9(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 padding=4, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_ad_k9, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        indices = [
            40, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80
        ]
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, indices]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias




class Conv2d_hd_k3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_hd_k3, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
        return conv_weight_hd, self.conv.bias



class Conv2d_hd_k5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_hd_k5, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape

        # 创建5x5卷积核的特定结构的权重张量
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
        
        # 设置横向差分
        conv_weight_hd[:, :, [0, 5, 10, 15, 20]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [4, 9, 14, 19, 24]] = -conv_weight[:, :, :]
        
        # 重新排列权重为5x5结构
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=5, k2=5)(conv_weight_hd)
        
        return conv_weight_hd, self.conv.bias
    


class Conv2d_hd_k9(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 padding=4, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_hd_k9, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        
        # 创建9x9卷积核的特定结构的权重张量
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 9 * 9).fill_(0)
        
        
        # 设置横向差分
        conv_weight_hd[:, :, [0, 9, 18, 27, 36, 45, 54, 63, 72]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [8, 17, 26, 35, 44, 53, 62, 71, 80]] = -conv_weight[:, :, :]
        
        # 重新排列权重为9x9结构
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=9, k2=9)(conv_weight_hd)
        
        return conv_weight_hd, self.conv.bias


class Conv2d_vd_k3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd_k3, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class Conv2d_vd_k5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1, bias=False):
        super(Conv2d_vd_k5, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2, 3, 4]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [20, 21, 22, 23, 24]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias



class Conv2d_vd_k9(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 padding=4, dilation=1, groups=1, bias=False):
        super(Conv2d_vd_k9, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 9 * 9).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [72, 73, 74, 75, 76, 77, 78, 79, 80]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias



class DEConv_k3(nn.Module):
    def __init__(self, dim, out_dim):
        super(DEConv_k3, self).__init__()
        self.conv1_1 = Conv2d_cd_k3(dim, out_dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd_k3(dim, out_dim, 3, bias=True)
        self.conv1_3 = Conv2d_ad_k3(dim, out_dim, 3, bias=True)
        self.conv1_4 = Conv2d_vd_k3(dim, out_dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, out_dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        # Vanilla
        # w = w5
        # b = b5

        # Vanilla+H
        # w = w2 + w5
        # b = b2 + b5

        # Vanilla+V
        # w = w4 + w5
        # b = b4 + b5

        # Vanilla+C
        # w = w1 + w5
        # b = b1 + b5

        # Vanilla+H+V
        w = w2 + w4 + w5
        b = b2 + b4 + b5

        # Vanilla+H+C
        # w = w1 + w2 + w5
        # b = b1 + b2 + b5

        # Vanilla+V+C
        # w = w1 + w4 + w5
        # b = b1 + b4 + b5

        # Vanilla+H+V+C
        # w = w1 + w2 + w4 + w5
        # b = b1 + b2 + b4 + b5

        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res
    
class DEConv_k5(nn.Module):
    def __init__(self, dim, out_dim):
        super(DEConv_k5, self).__init__()
        self.conv1_1 = Conv2d_cd_k5(dim, out_dim, 5, bias=True)
        self.conv1_2 = Conv2d_hd_k5(dim, out_dim, 5, bias=True)
        self.conv1_3 = Conv2d_ad_k5(dim, out_dim, 5, bias=True)
        self.conv1_4 = Conv2d_vd_k5(dim, out_dim, 5, bias=True)
        self.conv1_5 = nn.Conv2d(dim, out_dim, 5, padding=2, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        # Vanilla
        # w = w5
        # b = b5

        # Vanilla+H
        # w = w2 + w5
        # b = b2 + b5

        # Vanilla+V
        # w = w4 + w5
        # b = b4 + b5

        # Vanilla+C
        # w = w1 + w5
        # b = b1 + b5

        # Vanilla+H+V
        w = w2 + w4 + w5
        b = b2 + b4 + b5

        # Vanilla+H+C
        # w = w1 + w2 + w5
        # b = b1 + b2 + b5

        # Vanilla+V+C
        # w = w1 + w4 + w5
        # b = b1 + b4 + b5

        # Vanilla+H+V+C
        # w = w1 + w2 + w4 + w5
        # b = b1 + b2 + b4 + b5

        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=2, groups=1)

        return res
    
class DEConv_k9(nn.Module):
    def __init__(self, dim, out_dim):
        super(DEConv_k9, self).__init__() 
        self.conv1_1 = Conv2d_cd_k9(dim, out_dim, 9, bias=True)
        self.conv1_2 = Conv2d_hd_k9(dim, out_dim, 9, bias=True)
        self.conv1_3 = Conv2d_ad_k9(dim, out_dim, 9, bias=True)
        self.conv1_4 = Conv2d_vd_k9(dim, out_dim, 9, bias=True)
        self.conv1_5 = nn.Conv2d(dim, out_dim, 9, padding=4, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        # Vanilla
        # w = w5
        # b = b5

        # Vanilla+H
        # w = w2 + w5
        # b = b2 + b5

        # Vanilla+V
        # w = w4 + w5
        # b = b4 + b5

        # Vanilla+C
        # w = w1 + w5
        # b = b1 + b5

        # Vanilla+H+V
        w = w2 + w4 + w5
        b = b2 + b4 + b5

        # Vanilla+H+C
        # w = w1 + w2 + w5
        # b = b1 + b2 + b5

        # Vanilla+V+C
        # w = w1 + w4 + w5
        # b = b1 + b4 + b5

        # Vanilla+H+V+C
        # w = w1 + w2 + w4 + w5
        # b = b1 + b2 + b4 + b5

        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=4, groups=1)

        return res





def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])

    # Ensure that the number of filters matches the input and output channels
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)  # Shape: [num_filters, filter_length]
    dec_filters = dec_filters.unsqueeze(1).repeat(in_size, 1, 1)  # Shape: [num_filters, in_channels, filter_length]

    # Ensure that the number of filters matches the input and output channels
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)  # Shape: [num_filters, filter_length]
    rec_filters = rec_filters.unsqueeze(1).repeat(out_size, 1, 1)  # Shape: [num_filters, out_channels, filter_length]

    return dec_filters, rec_filters



def wavelet_transform(x, filters):
    b, c, l = x.shape
    pad = (filters.shape[2] // 2 - 1,)
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad[0])
    x = x.reshape(b, c, 2, l // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, l_half = x.shape
    pad = (filters.shape[2] // 2 - 1,)
    x = x.reshape(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad[0])
    return x

class WTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv1d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', stride=1, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1], init_scale=1.0)

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size, padding='same', stride=1, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 2, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv1d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if curr_shape[2] % 2 > 0:
                curr_pads = (0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 2, shape_x[3])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    
    def forward(self, x):
        return torch.mul(self.weight, x)


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=4):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = torch.transpose(x, 1, 2)
#         b, c, h, w = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         y = y.view(b, c, 1, 1)
#         x = x * y
#         x = torch.transpose(x, 1, 2)
#         return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        x = x * y
        x = torch.transpose(x, 1, 2)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x



