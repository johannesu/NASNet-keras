import unittest

import nasnet


class NumParametersTest(unittest.TestCase):
    def check_parameter_count(self, model, target_in_m):
        count = model.count_params() / 10 ** 6
        msg = '{} params #{}M suppose to be #{}M.'.format(model.name, count, target_in_m)
        self.assertAlmostEqual(target_in_m, count, msg=msg, delta=0.1)

    def test_cifar_10(self):
        self.check_parameter_count(nasnet.cifar10(), 3.3)
