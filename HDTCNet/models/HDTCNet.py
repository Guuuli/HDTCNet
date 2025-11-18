import torch
from torch import nn
import torch.nn.functional as F
import math
from models.layer import *


class HDTCNet(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, num_layers, pool_ratio, kern_size,
                 in_dim, hidden_dim, out_dim,
                 num_nodes, num_classes, dropout=0.5, activation=nn.ReLU()):

        super().__init__()

        # TODO: Sparsity Analysis
        self.num_nodes = num_nodes
        heads = 1

        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
        paddings = [(k - 1) // 2 for k in kern_size]


        self.left_num_nodes = []
        for layer in range(num_layers + 1):
            left_node = round(num_nodes * (1 - (pool_ratio * layer)))
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)


        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(heads * out_dim, num_classes)

        self.reset_parameters()


        self.act_class = F.gelu

        self.class_dropout = nn.Dropout(0.1)

        self.head_class = nn.Linear(32000, 4)

        ##############################

        self.Conv_D1 = DEConv_k9(1, in_dim)
        self.Conv_D2 = DEConv_k5(heads * in_dim, hidden_dim)
        self.Conv_D3 = DEConv_k3(heads * hidden_dim, out_dim)

        self.WT_Conv1 = WTConv1d(num_nodes, num_nodes, 5, stride=1, bias=True, wt_levels=4)

        self.Dense_timepool1 = Dense_TimeDiffPool2d(self.left_num_nodes[0], self.left_num_nodes[0], kern_size[0],
                                                    paddings[0])
        self.Dense_timepool2 = Dense_TimeDiffPool2d(self.left_num_nodes[0], self.left_num_nodes[0], kern_size[1],
                                                    paddings[1])
        self.Dense_timepool3 = Dense_TimeDiffPool2d(self.left_num_nodes[0], self.left_num_nodes[0], kern_size[-1],
                                                    paddings[-1])

        self.bn1 = nn.BatchNorm2d(heads * in_dim)
        self.bn2 = nn.BatchNorm2d(heads * hidden_dim)
        self.bn3 = nn.BatchNorm2d(heads * out_dim)

        self.ReLU = nn.ReLU()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, inputs: Tensor):

        x = inputs

        x = torch.squeeze(x, dim=1)

        x = self.WT_Conv1(x)

        x = x.unsqueeze(1)
        x = self.Conv_D1(x)  # (kernel_size=9)
        x = self.Dense_timepool1(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.Conv_D2(x)  # (kernel_size=5)
        x = self.Dense_timepool2(x)
        x = self.bn2(x)
        x = self.ReLU(x)

        x = self.Conv_D3(x)  # (kernel_size=3)
        x = self.Dense_timepool3(x)
        x = self.bn3(x)
        x = self.ReLU(x)

        out = self.global_pool(x)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # hyper param
        self.num_layers = configs.num_layers
        self.pool_ratio = configs.pool_ratio
        self.kern_size = configs.kern_size
        self.in_dim = configs.in_dim
        self.hidden_dim = configs.hidden_dim
        self.out_dim = configs.out_dim
        self.num_nodes = configs.enc_in
        self.num_classes = configs.num_class



        self.model = HDTCNet(num_layers=self.num_layers, pool_ratio=self.pool_ratio, kern_size=self.kern_size,
                     in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim, num_nodes=self.num_nodes, num_classes=self.num_classes)

    def forward(self, x, x_mark_enc=None):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.model(x)
        return x
