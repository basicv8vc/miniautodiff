# encoding:utf-8

import numpy as np
import miniad
import miniad.utils as utils


class Function:
    '''
    Base class for Function
    '''
    @staticmethod
    def forward():
        raise NotImplementedError

    @staticmethod
    def backward():
        raise NotImplementedError


class Add(Function):
    @staticmethod
    def forward(left, right):
        '''
        Add function

        Parameters
        ----------
        left:	Tensor
            left operand
        right:	Tensor or float
            right operand

        Returns
        -------
        Tensor
        '''
        is_right_tensor = True if isinstance(right, miniad.Tensor) else False
        out = miniad.Tensor(data=left.data +
                            right.data if is_right_tensor else left.data +
                            right)
        out.children = (left, right) if is_right_tensor else (left, )
        out.grad_fn = Add.backward

        return out

    @staticmethod
    def backward(out, left, right=None):
        '''
        out = left + right
        left_grad += out_grad
        right += out_grad
        '''
        out_grad = out.grad  # Do not revise out
        if out_grad is None:
            out_grad = np.ones_like(left.data)
        left.grad = out_grad if left.grad is None else left.grad + out_grad
        if right is None:  # Tensor + float = Tensor
            return (left, )
        else:
            right.grad = out_grad if right.grad is None else right.grad + out_grad
            return (left, right)


class Minus(Function):
    @staticmethod
    def forward(left, right):
        '''
        Minus function

        Parameters
        ----------
        left:	Tensor
            left operand
        right:	Tensor or float
            right operand

        Returns
        -------
        Tensor
        '''
        is_right_tensor = True if isinstance(right, miniad.Tensor) else False
        out = miniad.Tensor(data=left.data -
                            right.data if is_right_tensor else left.data -
                            right)
        out.children = (left, right) if is_right_tensor else (left, )
        out.grad_fn = Minus.backward

        return out

    @staticmethod
    def backward(out, left, right=None):
        '''
        out = left - right
        left_grad += out_grad
        right -= out_grad
        '''
        out_grad = out.grad  # Do not revise out
        if out_grad is None:
            out_grad = np.ones_like(left.data)
        left.grad = out_grad if left.grad is None else left.grad + out_grad
        if right is None:  # Tensor + float = Tensor
            return (left, )
        else:
            right.grad = -out_grad if right.grad is None else right.grad - out_grad
            return (left, right)


class Multiply(Function):
    @staticmethod
    def forward(left, right):
        '''Element-wise Multiply function

        Parameters
        ----------
        left:	Tensor
            left operand
        right:	Tensor or float
            right operand

        Returns
        -------
        Tensor
        '''
        is_right_tensor = True if isinstance(right, miniad.Tensor) else False
        out = miniad.Tensor(data=left.data *
                            right.data if is_right_tensor else left.data *
                            right)
        out.children = (left, right) if is_right_tensor else (left, )
        out.grad_fn = Multiply.backward
        out.save_for_backward = None if is_right_tensor else right

        return out

    @staticmethod
    def backward(out, left, right=None):
        '''
        out = left * right
        left_grad += out_grad * right.data
        right += out_grad * left.data
        '''
        out_grad = out.grad  # Do not revise out
        if out_grad is None:
            out_grad = np.ones_like(left.data)
        if right is None:
            right_data = out.save_for_backward
            left.grad = out_grad * right_data if left.grad is None else left.grad + \
                out_grad * right_data
            return (left, )
        else:
            left.grad = out_grad * right.data if left.grad is None else left.grad + \
                out_grad * right.data
            right.grad = out_grad * \
                left.data if right.grad is None else right.grad + out_grad * left.data

            return (left, right)


class Division(Function):
    @staticmethod
    def forward(numerator, denominator):
        '''
        Division function

        Parameters
        ----------
        numerator:	Tensor
            numerator operand
        denominator:	Tensor or float
            denominator operand

        Returns
        -------
        Tensor
        '''
        is_deno_tensor = True if isinstance(denominator,
                                            miniad.Tensor) else False
        EPS = 1e-12
        if is_deno_tensor:
            deno_data = utils.clip(denominator.data, EPS)
            out = miniad.Tensor(data=numerator.data / deno_data)
            out.children = (numerator, denominator)
        else:
            deno_data = utils.clip(denominator, EPS)
            out = miniad.Tensor(data=numerator.data / deno_data)
            out.children = (numerator, )
            out.save_for_backward = deno_data

        out.grad_fn = Division.backward

        return out

    @staticmethod
    def backward(out, numerator, denominator=None):
        '''
        out = nume / deno
        nume_grad += out_grad * (1/deno.data)
        deno_grad -= out_grad * nume.data / (deno.data * deno.data)
        '''

        out_grad = out.grad  # Do not revise out
        if out_grad is None:
            out_grad = np.ones_like(numerator.data)
        EPS = 1e-12
        if denominator is None:
            deno_data = out.save_for_backward
            numerator.grad = out_grad / deno_data if numerator.grad is None else numerator.grad + out_grad / deno_data

            return (numerator, )
        else:
            deno_data = utils.clip(denominator.data, EPS)
            numerator.grad = out_grad / deno_data if numerator.grad is None else numerator.grad + out_grad / deno_data
            deno_data = utils.clip(denominator.data**2, EPS)
            denominator.grad = -out_grad * numerator.data / deno_data if denominator.grad is None else denominator.grad - out_grad * numerator.data / deno_data

            return (numerator, denominator)


class Power(Function):
    @staticmethod
    def forward(base, exponent):
        '''
        Power function

        Parameters
        ----------
        base:	Tensor
            base operand
        exponent:	float
            exponent operand

        Returns
        -------
        Tensor
        '''
        out = miniad.Tensor(data=np.power(base.data, exponent))
        out.children = (base, )
        out.grad_fn = Power.backward
        out.save_for_backward = exponent

        return out

    @staticmethod
    def backward(out, base):
        '''
        out = base ^ {exponent}
        base_grad += exponent * base ^ {exponent - 1}
        '''
        out_grad = out.grad
        if out_grad is None:
            out_grad = np.ones_like(base.data)
        exponent = out.save_for_backward
        base_grad = out_grad * exponent * np.power(base.data, exponent - 1)
        base.grad = base_grad if base.grad is None else base.grad + base_grad

        return (base, )


# TODO bias can be None
class Linear(Function):
    @staticmethod
    def forward(x, weight, bias):
        '''Linear function
        y = x @ weight + bias
        Parameters
        ----------
        x:	Tensor
            input
        weight:	Tensor
            weight parameter
        bias:    Tensor
            bias parameter

        Returns
        -------
        Tensor
        '''
        out = miniad.Tensor(data=x.data @ weight.data + bias.data)
        out.children = (x, weight, bias)
        out.grad_fn = Linear.backward

        return out

    @staticmethod
    def backward(out, x, weight, bias):
        '''
        y = x @ w + b

        dx = dy @ w.T
        dw = x.T @ dy
        db = dy.sum(0)

        Note:
        x: (B, M)
        weight: (M, N)
        bias: (N)
        '''
        out_grad = out.grad
        if out_grad is None:
            out_grad = np.ones(
                (x.data.shape[0], weight.data.shape[1]))  # (B, N)

        x_grad = out_grad @ weight.data.T  # (B, M)
        w_grad = x.data.T @ out_grad  # (M, N)
        b_grad = np.sum(out_grad, 0)  # (N)

        x.grad = x_grad if x.grad is None else x_grad + x.grad
        weight.grad = w_grad if weight.grad is None else w_grad + weight.grad
        bias.grad = b_grad if bias.grad is None else b_grad + bias.grad

        return (x, weight, bias)


class ReLU(Function):
    @staticmethod
    def forward(x):
        '''ReLU function
        y = relu(x) = max(x, 0)

        Parameters
        ----------
        x:	Tensor
            input

        Returns
        -------
        Tensor
        '''
        out = miniad.Tensor(data=np.where(x.data >= 0, x.data, 0))
        out.children = (x, )
        out.grad_fn = ReLU.backward

        return out

    @staticmethod
    def backward(out, x):
        '''
        y = relu(x) = x if x >= 0 else 0
        x_grad = 1 if x >= 0 else 0
        '''
        out_grad = out.grad
        if out_grad is None:
            out_grad = np.ones_like(x.data)
        x_grad = np.where(x.data >= 0, 1., 0.)
        x_grad *= out_grad
        x.grad = x_grad if x.grad is None else x_grad + x.grad

        return (x, )


class Sigmoid(Function):
    @staticmethod
    def forward(x):
        '''Sigmoid function
        y = sigmoid(x) = 1/(1 + exp(-x))

        Parameters
        ----------
        x:	Tensor
            input

        Returns
        -------
        Tensor
        '''
        out = miniad.Tensor(data=1. / (np.exp(-x.data) + 1))
        out.children = (x, )
        out.grad_fn = Sigmoid.backward

        return out

    @staticmethod
    def backward(out, x):
        '''
        y = sigmoid(x)
        x_grad = sigmoid(x)(1 - sigmoid(x))
        '''
        out_grad = out.grad
        if out_grad is None:
            out_grad = np.ones_like(x.data)

        x_grad = (1. / (np.exp(-x.data) + 1)) * (1 - 1. /
                                                 (np.exp(-x.data) + 1))
        x_grad *= out_grad
        x.grad = x_grad if x.grad is None else x.grad + x_grad

        return (x, )


# TODO. dim can be sequence
class Squeeze(Function):
    @staticmethod
    def forward(x, dim=None):
        '''Squeeze function
        Remove dim of length one from x

        Parameters
        ----------
        x:	Tensor
            input
        dim:    int or None
            which dim to squeeze. If dim is None, squeeze all length one dims.
            

        Returns
        -------
        Tensor
        '''
        if dim is None:
            squeezed_data = np.squeeze(x.data)
        else:
            squeezed_data = np.squeeze(x.data, axis=dim)
        out = miniad.Tensor(data=squeezed_data)
        out.children = (x, )
        out.grad_fn = Squeeze.backward

        return out

    @staticmethod
    def backward(out, x):
        '''
        y = squeeze(x)
        x_grad = y_grad
        '''
        out_grad = out.grad
        if out_grad is None:
            out_grad = np.ones_like(x.data)
        out_grad = out_grad.reshape(x.data.shape)
        x.grad = out_grad if x.grad is None else x.grad + out_grad

        return (x, )


class BinaryCrossEntropyLoss(Function):
    @staticmethod
    def forward(y_hat, y):
        '''Binary Cross Entropy function

        L = -sum([ylog y_hat + (1-y)log(1-y_hat)]) / N

        Parameters
        ----------
        y_hat:	Tensor
            predicts
        y:	numpy.ndarray
            labels

        Returns
        -------
        Tensor
        '''
        EPS = 1e-12
        pos_indexs = np.where(y == 1)
        neg_indexs = np.where(y == 0)
        pos_y_hat = utils.clip(y_hat.data[pos_indexs], EPS)
        neg_y_hat = utils.clip(1 - y_hat.data[neg_indexs], EPS)

        loss_data = -(np.sum(np.log(pos_y_hat)) +
                      np.sum(np.log(neg_y_hat))) / len(y)
        loss = miniad.Tensor(data=loss_data)
        loss.children = (y_hat, )
        loss.grad_fn = BinaryCrossEntropyLoss.backward
        loss.save_for_backward = y

        return loss

    @staticmethod
    def backward(out, y_hat):
        '''
        y_hat_grad = if y == 1, -1/(N* y_hat)
        y_hat_grad = if y == 0,  1/(N * (1 - y_hat))
        '''
        out_grad = out.grad
        if out_grad is None:
            out_grad = 1  # loss is always a scalar
        y = out.save_for_backward
        EPS = 1e-12
        N = len(y)

        pos_indexs = np.where(y == 1)
        neg_indexs = np.where(y == 0)
        pos_y_hat = utils.clip(y_hat.data[pos_indexs] * N, EPS)
        neg_y_hat = utils.clip(N * (1 - y_hat.data[neg_indexs]), EPS)

        dx_pos = -1. / pos_y_hat
        dx_neg = 1. / neg_y_hat
        dx = np.zeros_like(y_hat.data)
        dx[pos_indexs] = dx_pos
        dx[neg_indexs] = dx_neg
        dx *= out_grad

        y_hat.grad = dx if y_hat.grad is None else y_hat.grad + dx

        return (y_hat, )


class MSELoss(Function):
    @staticmethod
    def forward(y_hat, y):
        '''Mean Squared Error function

        L = sum((y_hat - y) ** 2) / N

        Parameters
        ----------
        y_hat:	Tensor
            predicts
        y:	numpy.ndarray
            labels

        Returns
        -------
        Tensor
        '''
        loss_data = np.sum((y_hat.data - y)**2) / len(y)
        loss = miniad.Tensor(data=loss_data)
        loss.children = (y_hat, )
        loss.grad_fn = MSELoss.backward
        loss.save_for_backward = y

        return loss

    @staticmethod
    def backward(out, y_hat):
        '''
        l = sum((y_hat - y) ** 2) / N
        y_hat_grad = 2 * (y_hat - y) / N
        '''
        out_grad = out.grad
        if out_grad is None:
            out_grad = 1  # loss is always a scalar
        y = out.save_for_backward
        N = len(y)
        dx = 2 * (y_hat.data - y) / N
        dx *= out_grad
        
        y_hat.grad = dx if y_hat.grad is None else y_hat.grad + dx

        return (y_hat, )


add = Add.forward
minus = Minus.forward
multiply = Multiply.forward
division = Division.forward
power = Power.forward
linear = Linear.forward
relu = ReLU.forward
sigmoid = Sigmoid.forward
squeeze = Squeeze.forward
binary_cross_entropy_loss = BinaryCrossEntropyLoss.forward
mse_loss = MSELoss.forward
