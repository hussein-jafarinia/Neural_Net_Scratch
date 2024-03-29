import numpy as np
from module import Module


class ReLU(Module):
    def __init__(self, name):
        super(ReLU, self).__init__(name)

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of ReLU function for input x.
        **Save whatever you need for backward pass in self.cache.
        """
        out = None
        # todo: implement the forward propagation for ReLU module
        # if x > 0:
        #     out = x
        # else:
        #     out = 0
        rel = lambda x: x * (x > 0).astype(float)
        out = rel(x)
        # out = 12
        self.cache = x
        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        dx = None
        # todo: implement the backward propagation for ReLU module.
        dx = dout * ( self.cache >= 0)
  
        return dx
