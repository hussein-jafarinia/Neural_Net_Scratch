import numpy as np
from module import Module


class SoftmaxCrossentropy(Module):
    def __init__(self, name):
        super(SoftmaxCrossentropy, self).__init__(name)

    def forward(self, x, **kwargs):
        y = kwargs.pop('y', None)
        """
        x: input array.
        y: real labels for this input.
        probs: probabilities of labels for this input.
        loss: cross entropy loss between probs and real labels.
        **Save whatever you need for backward pass in self.cache.
        """
        probs = None
        loss = None
        # todo: implement the forward propagation for probs and compute cross entropy loss
        # NOTE: implement a numerically stable version.If you are not careful here
        # it is easy to run into numeric instability!
        #softmax
        # e_x = np.exp(x - np.max(x))
        # probs = e_x / np.sum(e_x, axis=0)
        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        probs = exp_x/sum_exp_x
        #cross-entropy
        y_one_hut = np.zeros((y.size, y.max() + 1))
        y_one_hut[np.arange(y.size), y] = 1
        loss = np.sum(y_one_hut * -(np.log(probs)))
        # print(loss)
        # loss = loss / float(probs.shape[1])
        loss /= probs.shape[0]
        print(loss)
        
        self.cache = np.copy(probs), np.copy(y_one_hut)
        return loss, probs

    def backward(self, dout=0):
        dx = None
        dx = (self.cache[0] - self.cache[1]) / self.cache[1].shape[0] #dont know why devide by self.cache[1].shape[0]
        return dx