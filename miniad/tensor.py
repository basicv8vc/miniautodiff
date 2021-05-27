# encoding: utf-8
import numpy as np
import miniad.functional as F


class Tensor:
    def __init__(self, data, children=None, grad_fn=None):
        if isinstance(data, np.ndarray):
            self.data = data
        if isinstance(data, (list, tuple)):
            self.data = np.array(data)
        else:
            self.data = data  # scalar is ok

        self.data = data
        self.grad = None
        self.children = children
        self.grad_fn = grad_fn
        self.save_for_backward = None  # something for backward, like exponent in power function, or Tensor + scalar

    def __add__(self, other):
        out = F.add(self, other)

        return out

    def __radd__(self, other):
        out = F.add(self, other)

        return out

    def __minus__(self, other):
        out = F.minus(self, other)

        return out

    def __rminus__(self, other):
        out = F.minus(self, other)

        return out

    def __mul__(self, other):
        out = F.multiply(self, other)

        return out

    def __rmul__(self, other):
        out = F.multiply(self, other)

        return out

    def __div__(self, other):
        out = F.divison(self, other)

        return out

    def __rdiv__(self, other):
        out = F.divison(self, other)

        return out

    def backward(self, out_grad=None):
        '''Backpropagtaion
        Only root node can call backward function
        
        Parameters
        ----------
        out_grad:    float
        '''
        self.grad = np.ones_like(self.data) if self.grad is None else self.grad
        self.grad = self.grad if out_grad is None else self.grad + np.ones_like(
            self.data) * out_grad
        candidates = [self]
        while candidates:
            curr_node = candidates.pop()
            if curr_node.grad_fn is not None:
                curr_node.children = curr_node.grad_fn(curr_node,
                                                       *curr_node.children)
                for item in curr_node.children:
                    candidates.append(item)
