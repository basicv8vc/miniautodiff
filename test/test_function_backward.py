# encoding:utf-8
import unittest
import copy
import random
import numpy as np
import miniad
import miniad.functional as F
random.seed(123)
np.random.seed(123)

EPS = 1e-5


class TestAddBackward(unittest.TestCase):
    def test_tensor_add_scalar(self):
        '''tensor + scalar, test left operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(input_data)
        right = 2
        output = F.add(left, right)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            left_p = copy.deepcopy(input_data)
            left_p[i] += EPS
            left_m = copy.deepcopy(input_data)
            left_m[i] -= EPS

            left_p = miniad.Tensor(left_p)
            left_m = miniad.Tensor(left_m)

            appro_p = F.add(left_p, right)
            appro_m = F.add(left_m, right)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_tensor_add_tensor_left(self):
        '''tensor + tensor, test left operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(input_data)
        right = miniad.Tensor(np.array([-1., 100.9, 1e-5]))
        output = F.add(left, right)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            left_p = copy.deepcopy(input_data)
            left_p[i] += EPS
            left_m = copy.deepcopy(input_data)
            left_m[i] -= EPS

            left_p = miniad.Tensor(left_p)
            left_m = miniad.Tensor(left_m)

            appro_p = F.add(left_p, right)
            appro_m = F.add(left_m, right)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_tensor_add_tensor_right(self):
        '''tensor + tensor, test right operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(np.array([-1., 100.9, 1e-5]))
        right = miniad.Tensor(input_data)
        output = F.add(left, right)
        output.backward()
        grad = output.children[1].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            right_p = copy.deepcopy(input_data)
            right_p[i] += EPS
            right_m = copy.deepcopy(input_data)
            right_m[i] -= EPS

            right_p = miniad.Tensor(right_p)
            right_m = miniad.Tensor(right_m)

            appro_p = F.add(left, right_p)
            appro_m = F.add(left, right_m)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestMinusBackward(unittest.TestCase):
    def test_tensor_minus_scalar(self):
        '''tensor - scalar, test left operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(input_data)
        right = 2
        output = F.minus(left, right)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            left_p = copy.deepcopy(input_data)
            left_p[i] += EPS
            left_m = copy.deepcopy(input_data)
            left_m[i] -= EPS

            left_p = miniad.Tensor(left_p)
            left_m = miniad.Tensor(left_m)

            appro_p = F.minus(left_p, right)
            appro_m = F.minus(left_m, right)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_tensor_minus_tensor_left(self):
        '''tensor - tensor, test left operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(input_data)
        right = miniad.Tensor(np.array([-1., 100.9, 1e-5]))
        output = F.minus(left, right)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            left_p = copy.deepcopy(input_data)
            left_p[i] += EPS
            left_m = copy.deepcopy(input_data)
            left_m[i] -= EPS

            left_p = miniad.Tensor(left_p)
            left_m = miniad.Tensor(left_m)

            appro_p = F.minus(left_p, right)
            appro_m = F.minus(left_m, right)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_tensor_minus_tensor_right(self):
        '''tensor - tensor, test right operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(np.array([-1., 100.9, 1e-5]))
        right = miniad.Tensor(input_data)
        output = F.minus(left, right)
        output.backward()
        grad = output.children[1].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            right_p = copy.deepcopy(input_data)
            right_p[i] += EPS
            right_m = copy.deepcopy(input_data)
            right_m[i] -= EPS

            right_p = miniad.Tensor(right_p)
            right_m = miniad.Tensor(right_m)

            appro_p = F.minus(left, right_p)
            appro_m = F.minus(left, right_m)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestMultiplyBackward(unittest.TestCase):
    def test_tensor_multiply_scalar(self):
        '''tensor * scalar, test left operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(input_data)
        right = 2.
        output = F.multiply(left, right)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            left_p = copy.deepcopy(input_data)
            left_p[i] += EPS
            left_m = copy.deepcopy(input_data)
            left_m[i] -= EPS

            left_p = miniad.Tensor(left_p)
            left_m = miniad.Tensor(left_m)

            appro_p = F.multiply(left_p, right)
            appro_m = F.multiply(left_m, right)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_tensor_multiply_tensor_left(self):
        '''tensor * tensor, test left operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(input_data)
        right = miniad.Tensor(np.array([-1., 100.9, 1e-5]))
        output = F.multiply(left, right)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            left_p = copy.deepcopy(input_data)
            left_p[i] += EPS
            left_m = copy.deepcopy(input_data)
            left_m[i] -= EPS

            left_p = miniad.Tensor(left_p)
            left_m = miniad.Tensor(left_m)

            appro_p = F.multiply(left_p, right)
            appro_m = F.multiply(left_m, right)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_tensor_multiply_tensor_right(self):
        '''tensor * tensor, test right operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(np.array([-1., 100.9, 1e-5]))
        right = miniad.Tensor(input_data)
        output = F.multiply(left, right)
        output.backward()
        grad = output.children[1].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            right_p = copy.deepcopy(input_data)
            right_p[i] += EPS
            right_m = copy.deepcopy(input_data)
            right_m[i] -= EPS

            right_p = miniad.Tensor(right_p)
            right_m = miniad.Tensor(right_m)

            appro_p = F.multiply(left, right_p)
            appro_m = F.multiply(left, right_m)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestDivisionBackward(unittest.TestCase):
    def test_tensor_division_scalar(self):
        '''tensor / scalar '''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(input_data)
        right = 2.
        output = F.division(left, right)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            left_p = copy.deepcopy(input_data)
            left_p[i] += EPS
            left_m = copy.deepcopy(input_data)
            left_m[i] -= EPS

            left_p = miniad.Tensor(left_p)
            left_m = miniad.Tensor(left_m)

            appro_p = F.division(left_p, right)
            appro_m = F.division(left_m, right)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_tensor_division_tensor(self):
        '''tensor / tensor, test left operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(input_data)
        right = miniad.Tensor(np.array([-1., 100.9, 1e-5]))
        output = F.division(left, right)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            left_p = copy.deepcopy(input_data)
            left_p[i] += EPS
            left_m = copy.deepcopy(input_data)
            left_m[i] -= EPS

            left_p = miniad.Tensor(left_p)
            left_m = miniad.Tensor(left_m)

            appro_p = F.division(left_p, right)
            appro_m = F.division(left_m, right)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_tensor_division_tensor_right(self):
        '''tensor / tensor, test right operand'''
        input_data = np.array([1., 2., 3.])
        left = miniad.Tensor(np.array([-1., 100.9, 1e-5]))
        right = miniad.Tensor(input_data)
        output = F.division(left, right)
        output.backward()
        grad = output.children[1].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            right_p = copy.deepcopy(input_data)
            right_p[i] += EPS
            right_m = copy.deepcopy(input_data)
            right_m[i] -= EPS

            right_p = miniad.Tensor(right_p)
            right_m = miniad.Tensor(right_m)

            appro_p = F.division(left, right_p)
            appro_m = F.division(left, right_m)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestPowerBackward(unittest.TestCase):
    def test_power_backward_function(self):
        input_data = np.array([1., 2., 3.])
        base = miniad.Tensor(input_data)
        exponent = 2.5
        output = F.power(base, exponent)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(len(input_data)):

            base_p = copy.deepcopy(input_data)
            base_p[i] += EPS
            base_m = copy.deepcopy(input_data)
            base_m[i] -= EPS

            base_p = miniad.Tensor(base_p)
            base_m = miniad.Tensor(base_m)

            appro_p = F.power(base_p, exponent)
            appro_m = F.power(base_m, exponent)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestLinearBackward(unittest.TestCase):
    def test_linear_backward_function_x(self):
        '''x @ weight + bias, test x operand

        Note:
        x: (B, M)
        weight: (M, N)
        bias: (N)
        '''
        B = 32
        M = 16
        N = 64
        input_data = np.random.randn(B, M)
        x = miniad.Tensor(input_data)
        weight = miniad.Tensor(np.random.randn(M, N))
        bias = miniad.Tensor(np.random.randn(N))
        output = F.linear(x, weight, bias)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(B):
            for j in range(M):
                x_p = copy.deepcopy(input_data)
                x_p[i][j] += EPS
                x_m = copy.deepcopy(input_data)
                x_m[i][j] -= EPS

                x_p = miniad.Tensor(x_p)
                x_m = miniad.Tensor(x_m)

                appro_p = F.linear(x_p, weight, bias)
                appro_m = F.linear(x_m, weight, bias)
                appro_grad[i][j] = np.sum(
                    (appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_linear_backward_function_weight(self):
        '''x @ weight + bias, test weight operand

        Note:
        x: (B, M)
        weight: (M, N)
        bias: (N)
        '''
        B = 32
        M = 16
        N = 64
        input_data = np.random.randn(M, N)
        x = miniad.Tensor(np.random.randn(B, M))
        weight = miniad.Tensor(input_data)
        bias = miniad.Tensor(np.random.randn(N))
        output = F.linear(x, weight, bias)
        output.backward()
        grad = output.children[1].grad

        appro_grad = np.zeros_like(grad)
        for i in range(M):
            for j in range(N):
                weight_p = copy.deepcopy(input_data)
                weight_p[i][j] += EPS
                weight_m = copy.deepcopy(input_data)
                weight_m[i][j] -= EPS

                weight_p = miniad.Tensor(weight_p)
                weight_m = miniad.Tensor(weight_m)

                appro_p = F.linear(x, weight_p, bias)
                appro_m = F.linear(x, weight_m, bias)
                appro_grad[i][j] = np.sum(
                    (appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_linear_backward_function_bias(self):
        '''x @ weight + bias, test bias operand

        Note:
        x: (B, M)
        weight: (M, N)
        bias: (N)
        '''
        B = 32
        M = 16
        N = 64
        input_data = np.random.randn(N)
        x = miniad.Tensor(np.random.randn(B, M))
        weight = miniad.Tensor(np.random.randn(M, N))
        bias = miniad.Tensor(input_data)
        output = F.linear(x, weight, bias)
        output.backward()
        grad = output.children[2].grad

        appro_grad = np.zeros_like(grad)
        for i in range(N):
            bias_p = copy.deepcopy(input_data)
            bias_p[i] += EPS
            bias_m = copy.deepcopy(input_data)
            bias_m[i] -= EPS

            bias_p = miniad.Tensor(bias_p)
            bias_m = miniad.Tensor(bias_m)

            appro_p = F.linear(x, weight, bias_p)
            appro_m = F.linear(x, weight, bias_m)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestReLUBackward(unittest.TestCase):
    def test_relu_backward_function(self):
        '''relu(x)'''
        input_data = np.random.randn(16, 8)
        x = miniad.Tensor(input_data)
        output = F.relu(x)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(16):
            for j in range(8):
                x_p = copy.deepcopy(input_data)
                x_p[i][j] += EPS
                x_m = copy.deepcopy(input_data)
                x_m[i][j] -= EPS

                x_p = miniad.Tensor(x_p)
                x_m = miniad.Tensor(x_m)

                appro_p = F.relu(x_p)
                appro_m = F.relu(x_m)
                appro_grad[i][j] = np.sum(
                    (appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestSigmoidBackward(unittest.TestCase):
    def test_sigmoid_backward_function(self):
        '''sigmoid(x)'''
        input_data = np.random.randn(16, 8)
        x = miniad.Tensor(input_data)
        output = F.sigmoid(x)
        output.backward()
        grad = output.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(16):
            for j in range(8):
                x_p = copy.deepcopy(input_data)
                x_p[i][j] += EPS
                x_m = copy.deepcopy(input_data)
                x_m[i][j] -= EPS

                x_p = miniad.Tensor(x_p)
                x_m = miniad.Tensor(x_m)

                appro_p = F.sigmoid(x_p)
                appro_m = F.sigmoid(x_m)
                appro_grad[i][j] = np.sum(
                    (appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestSqueezeBackward(unittest.TestCase):
    def test_squeeze_all(self):
        '''squeeze(x, dim=None)'''
        x = miniad.Tensor(
            np.random.randn(16, 8).reshape((1, 16, 1, 1, 8, 1, 1)))
        output = F.squeeze(x)
        output.backward()

        x_grad = output.children[0].grad
        output_grad = output.grad.reshape(x_grad.shape)

        difference = np.linalg.norm(x_grad - output_grad) / (
            np.linalg.norm(x_grad) + np.linalg.norm(output_grad))
        self.assertAlmostEqual(difference, 0, places=7)

    def test_squeeze_some_dim(self):
        '''squeeze(x, dim=3)'''
        x = miniad.Tensor(
            np.random.randn(16, 8).reshape((1, 16, 1, 1, 8, 1, 1)))
        output = F.squeeze(x, dim=3)
        output.backward()

        x_grad = output.children[0].grad
        output_grad = output.grad.reshape(x_grad.shape)

        difference = np.linalg.norm(x_grad - output_grad) / (
            np.linalg.norm(x_grad) + np.linalg.norm(output_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestBinaryCrossEntropyLossBackward(unittest.TestCase):
    def test_bce_backward_function(self):
        '''binary_cross_entropy(y_hat, y)'''
        input_data = np.random.uniform(size=(10, ))
        y_hat = miniad.Tensor(input_data)
        y = np.random.randint(2, size=(10, ))
        loss = F.binary_cross_entropy_loss(y_hat, y)
        loss.backward()

        grad = loss.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(10):
            x_p = copy.deepcopy(input_data)
            x_p[i] += EPS
            x_m = copy.deepcopy(input_data)
            x_m[i] -= EPS

            x_p = miniad.Tensor(x_p)
            x_m = miniad.Tensor(x_m)

            appro_p = F.binary_cross_entropy_loss(x_p, y)
            appro_m = F.binary_cross_entropy_loss(x_m, y)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


class TestMSELossBackward(unittest.TestCase):
    def test_mse_backward_function(self):
        '''mse(y_hat, y)'''
        input_data = np.random.randn(10)
        y_hat = miniad.Tensor(input_data)
        y = np.random.randn(10)
        loss = F.mse_loss(y_hat, y)
        loss.backward()

        grad = loss.children[0].grad

        appro_grad = np.zeros_like(grad)
        for i in range(10):
            x_p = copy.deepcopy(input_data)
            x_p[i] += EPS
            x_m = copy.deepcopy(input_data)
            x_m[i] -= EPS

            x_p = miniad.Tensor(x_p)
            x_m = miniad.Tensor(x_m)

            appro_p = F.mse_loss(x_p, y)
            appro_m = F.mse_loss(x_m, y)
            appro_grad[i] = np.sum((appro_p.data - appro_m.data) / (2 * EPS))

        difference = np.linalg.norm(grad - appro_grad) / (
            np.linalg.norm(grad) + np.linalg.norm(appro_grad))
        self.assertAlmostEqual(difference, 0, places=7)


if __name__ == '__main__':
    unittest.main()
