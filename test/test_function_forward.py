# encoding:utf-8ß
import unittest
import numpy as np
import miniad
import miniad.functional as F


class TestAddForward(unittest.TestCase):
    def test_tensor_add_scalar(self):
        '''tensor + scalar'''
        left = miniad.Tensor(np.array([1, 2, 3]))
        right = 1
        self.assertTrue(
            np.allclose(F.add(left, right).data,
                        np.array([2, 3, 4]),
                        rtol=1e-05,
                        atol=1e-08))

    def test_tensor_add_tensor(self):
        '''tensor + tensor'''
        left = miniad.Tensor(np.array([1.5, 2, 5]))
        right = miniad.Tensor(np.array([1, 2, 3]))
        self.assertTrue(
            np.allclose(F.add(left, right).data,
                        np.array([2.5, 4, 8]),
                        rtol=1e-05,
                        atol=1e-08))


class TestMinusForward(unittest.TestCase):
    def test_tensor_minus_scalar(self):
        '''tensor - scalar'''
        left = miniad.Tensor(np.array([1, 2, 3]))
        right = 1
        self.assertTrue(
            np.allclose(F.minus(left, right).data,
                        np.array([0, 1, 2]),
                        rtol=1e-05,
                        atol=1e-08))

    def test_tensor_minus_tensor(self):
        '''tensor - tensor'''
        left = miniad.Tensor(np.array([1.5, 2, 5]))
        right = miniad.Tensor(np.array([1, 2, 3]))
        self.assertTrue(
            np.allclose(F.minus(left, right).data,
                        np.array([0.5, 0, 2]),
                        rtol=1e-05,
                        atol=1e-08))


class TestMultiplyForward(unittest.TestCase):
    def test_tensor_multiply_scalar(self):
        '''tensor * scalar'''
        left = miniad.Tensor(np.array([1, 2, 3]))
        right = 1
        self.assertTrue(
            np.allclose(F.multiply(left, right).data,
                        np.array([1, 2, 3]),
                        rtol=1e-05,
                        atol=1e-08))

    def test_tensor_multiply_tensor(self):
        '''tensor * tensor'''
        left = miniad.Tensor(np.array([1.5, 2, 5]))
        right = miniad.Tensor(np.array([1, 2, 3]))
        self.assertTrue(
            np.allclose(F.multiply(left, right).data,
                        np.array([1.5, 4, 15]),
                        rtol=1e-05,
                        atol=1e-08))


class TestDivisionForward(unittest.TestCase):
    def test_tensor_division_scalar(self):
        '''tensor / scalar'''
        left = miniad.Tensor(np.array([1, 2, 3]))
        right = 1.
        self.assertTrue(
            np.allclose(F.division(left, right).data,
                        np.array([1, 2, 3]),
                        rtol=1e-05,
                        atol=1e-08))

    def test_tensor_division_tensor(self):
        '''tensor / tensor'''
        left = miniad.Tensor(np.array([1.5, 2, 5]))
        right = miniad.Tensor(np.array([1, 2, 3]))
        self.assertTrue(
            np.allclose(F.division(left, right).data,
                        np.array([1.5, 1, 5 / 3]),
                        rtol=1e-05,
                        atol=1e-08))


class TestPowerForward(unittest.TestCase):
    def test_power_forward_function(self):
        base = miniad.Tensor(1.5)
        exponent = 2.

        self.assertEqual(F.power(base, exponent).data, 2.25, "Should be 2.25")


class TestLinearForward(unittest.TestCase):
    def test_linear_forward_function(self):
        '''
        batch_size = 32
        M = 128
        N = 16
        '''
        x = miniad.Tensor(np.arange(32 * 128).reshape(32, 128))
        weight = miniad.Tensor(np.arange(128 * 16).reshape(128, 16))
        bias = miniad.Tensor(np.arange(16))

        self.assertTrue(
            np.allclose(F.linear(x, weight, bias).data,
                        np.arange(32 * 128).reshape(32, 128) @ np.arange(
                            128 * 16).reshape(128, 16) + np.arange(16),
                        rtol=1e-05,
                        atol=1e-08))


class TestReLUForward(unittest.TestCase):
    def test_relu_forward_function(self):
        x = miniad.Tensor(np.array([-0.1, 0.1, 0, 1.1]))

        self.assertTrue(
            np.allclose(F.relu(x).data,
                        np.array([0, 0.1, 0, 1.1]),
                        rtol=1e-05,
                        atol=1e-08))


class TestSigmoidForward(unittest.TestCase):
    def test_sigmoid_forward_function(self):
        x = miniad.Tensor(np.array([-0.1, 0.1, 0, 1.1]))

        def sigmoid(x):
            return 1. / (1 + np.exp(-x))

        self.assertTrue(
            np.allclose(F.sigmoid(x).data,
                        sigmoid(x.data),
                        rtol=1e-05,
                        atol=1e-08))


class TestSqueezeForward(unittest.TestCase):
    def test_squeeze_forward_function_all(self):
        data = np.arange(12).reshape(1, 1, 3, 1, 1, 4, 1, 1)
        x = miniad.Tensor(data)

        self.assertTrue(
            np.allclose(F.squeeze(x).data,
                        np.arange(12).reshape(3, 4),
                        rtol=1e-05,
                        atol=1e-08))

    def test_squeeze_forward_function_dim(self):
        data = np.arange(12).reshape(1, 1, 3, 1, 1, 4, 1, 1)
        x = miniad.Tensor(data)

        self.assertTrue(
            np.allclose(F.squeeze(x, dim=0).data,
                        np.arange(12).reshape(1, 3, 1, 1, 4, 1, 1),
                        rtol=1e-05,
                        atol=1e-08))


class TestBinaryCrossEntropyLossForward(unittest.TestCase):
    def test_bce_forward_function(self):
        y = np.array([1, 0, 1, 0, 1])
        y_hat = np.array([0.1, 0.9, 0.5, 0.6, 0.3])
        y_hat = miniad.Tensor(y_hat)

        def bce(y, y_hat):
            loss = 0.
            for yl, yp in zip(y, y_hat):
                if yl > 0:
                    loss += (-np.log(yp))
                else:
                    loss += (-np.log(1 - yp))
            return loss / len(y)

        self.assertAlmostEqual(F.binary_cross_entropy_loss(y_hat, y).data,
                               bce(y, y_hat.data),
                               places=4)


class TestMSELossForward(unittest.TestCase):
    def test_mse_forward_function(self):
        y = np.array([1, 0, 1, 0, 1])
        y_hat = np.array([0.1, 0.9, 0.5, 0.6, 0.3])
        y_hat = miniad.Tensor(y_hat)

        def mse(y, y_hat):
            loss = 0.
            for yl, yp in zip(y, y_hat):
                loss += (yl - yp)**2
            return loss / len(y)

        self.assertAlmostEqual(F.mse_loss(y_hat, y).data,
                               mse(y, y_hat.data),
                               places=4)


if __name__ == '__main__':
    unittest.main()
