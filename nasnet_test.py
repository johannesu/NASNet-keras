from __future__ import division
import logging

logging.getLogger('tensorflow').setLevel(logging.WARNING)

import unittest
import nasnet
import keras.backend as K


class ModelTest(unittest.TestCase):
    def check_parameter_count(self, model, target_in_m):
        count = model.count_params() / 10 ** 6
        msg = '{} params #{}M suppose to be #{}M.'.format(model.name, count, target_in_m)
        self.assertAlmostEqual(target_in_m, count, msg=msg, delta=0.1)

    def check_penultimate_shape(self, model, target_shape):
        layer = model.get_layer('last_relu')

        if K.image_data_format() == 'channels_first':
            shape = layer.input_shape[2:]
        else:
            shape = layer.input_shape[1:3]

        self.assertEqual(shape, target_shape)

    def test_cifar_10(self):
        model = nasnet.cifar10()
        self.check_parameter_count(model, 3.3)
        self.check_penultimate_shape(model, (8, 8))

        aux_model = nasnet.cifar10(add_aux_output=True)
        self.check_parameter_count(aux_model, 4.9)
        self.assertEqual(len(aux_model.output), 2)

    def test_mobile(self):
        model = nasnet.mobile()
        self.check_parameter_count(model, 5.3)
        self.check_penultimate_shape(model, (7, 7))

        aux_model = nasnet.mobile(add_aux_output=True)
        self.check_parameter_count(aux_model, 7.7)
        self.assertEqual(len(aux_model.output), 2)

    def test_large(self):
        model = nasnet.large()
        self.check_parameter_count(model, 88.9)
        self.check_penultimate_shape(model, (11, 11))

        aux_model = nasnet.large(add_aux_output=True)
        self.check_parameter_count(aux_model, 93.5)
        self.assertEqual(len(aux_model.output), 2)
