import math
import torch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.module import Module

def is_uninitialized_parameter(x) -> bool:
    if not hasattr(torch.nn.parameter, 'UninitializedParameter'):
        return False
    return isinstance(x, UninitializedParameter)

class GCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if in_features > 0:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.weight = UninitializedParameter()

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if is_uninitialized_parameter(self.weight):
            return
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        if is_uninitialized_parameter(self.weight):
            self.in_features = x[0].size(-1)
            self.weight.materialize((self.in_features, self.out_features), dtype=torch.float)
            self.reset_parameters()
    
        support = torch.mm(x, self.weight)
        output = torch.spmm(edge_index, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'