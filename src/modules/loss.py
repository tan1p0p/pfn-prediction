import numpy as np

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check

class MeanAbsoluteErrorWithWeight(function_node.FunctionNode):
    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        w = np.where(x1 <= 0, 0.001, 1).astype(np.float32)
        self.diff = (x0 - x1) * w
        diff = self.diff.ravel()
        return np.array(abs(diff).sum() / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        w = np.where(x1 <= 0, 0.001, 1).astype(np.float32)
        self.diff = (x0 - x1) * w
        diff = self.diff.ravel()
        return abs(diff).sum() / diff.dtype.type(diff.size),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        coeff = gy * gy.data.dtype.type(1. / self.diff.size)
        coeff = chainer.functions.broadcast_to(coeff, self.diff.shape)
        gx0 = coeff * backend.get_array_module(gy.data).sign(self.diff)
        return gx0, -gx0

def mean_absolute_error_with_weight(x0, x1):
    return MeanAbsoluteErrorWithWeight().apply((x0, x1))[0]