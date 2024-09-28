import math
import mindspore
from mindspore import nn, Parameter, Tensor
import mindspore.ops as ops
from mindspore.common.initializer import Normal

class GCNM(nn.Cell):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GCNM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight
        if in_features > 0:
            self.weight = Parameter(Tensor(shape=(in_features, out_features), dtype=mindspore.float32, init=Normal()))
        else:
            self.weight = None  # Will be initialized later
        
        # Initialize bias
        if bias:
            self.bias = Parameter(Tensor(shape=(out_features), dtype=mindspore.float32, init=Normal()))
        else:
            self.bias = None
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is None:
            return
        stdv = Tensor(1. / math.sqrt(self.weight.shape[1]), dtype=mindspore.float32)
        self.weight.set_data(Tensor(ops.uniform(self.weight.shape, -stdv, stdv), mindspore.float32))
        if self.bias is not None:
            self.bias.set_data(Tensor(ops.uniform(self.bias.shape, -stdv, stdv), mindspore.float32))

    def construct(self, x, edge_index):
        # Dynamically initialize weight if needed
        if self.weight is None:
            self.in_features = x.shape[-1]
            self.weight = Parameter(Tensor(shape=(self.in_features, self.out_features), dtype=mindspore.float32, init=Normal()))
            self.reset_parameters()

        # Support computation: x * W
        support = ops.MatMul()(x, self.weight)
        
        # Output computation: A * (x * W)
        spmm = ops.SparseTensorDenseMatmul()
        output = spmm(edge_index, support)
        
        # Add bias if exists
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
