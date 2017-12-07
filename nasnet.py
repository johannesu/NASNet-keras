"""Implementation of NASNet-A"""
from __future__ import division

import os


import tensorflow as tf
from keras import Input, Model, layers
from keras import backend as K
from keras.engine import get_source_inputs
from keras.layers import Activation, SeparableConv2D, BatchNormalization, Dropout
from keras.layers import AveragePooling2D, MaxPooling2D, Add, Concatenate
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense
from keras.layers import ZeroPadding2D, GlobalMaxPooling2D, Cropping2D
from keras.utils import get_file, Progbar


def preprocess(image, size):
    with tf.Session():
        x = preprocess_tf(image, size).eval()

    return x


def preprocess_tf(image, size=224, central_fraction=0.875):
    """Used to train the weights

    From:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=central_fraction)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [size, size], align_corners=False)
    image = tf.squeeze(image, [0])

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image


class CifarStem:
    def __init__(self, stem_filters, filters):
        self.stem_filters = stem_filters

    def __call__(self, x):
        with K.name_scope('cifar-stem'):
            x = Convolution2D(self.stem_filters, 3, kernel_initializer='he_normal', padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)

        return None, x


class ImagenetStem:
    def __init__(self, stem_filters, filters):
        self.stem_filters = stem_filters
        self.filters = filters

    def __call__(self, x):
        with K.name_scope('imagenet-stem'):
            x = Convolution2D(self.stem_filters, 3, strides=2,
                              kernel_initializer='he_normal', padding='valid', use_bias=False,
                              name='conv0')(x)
            x = BatchNormalization(name='conv0_bn')(x)

            prev = ReductionCell(self.filters // 4, prefix='cell_stem_0')(None, x)
            cur = ReductionCell(self.filters // 2, prefix='cell_stem_1')(x, prev)

        return prev, cur


class Separable:
    def __init__(self, filters, kernel_size, prefix, strides=1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.prefix = prefix
        self.strides = strides

    def __call__(self, x):
        with K.name_scope('separable_{0}x{0}_strides_{1}'.format(self.kernel_size, self.strides)):
            for repeat in range(1, 3):
                strides = self.strides if repeat == 1 else 1

                x = Activation('relu')(x)

                name = '{0}/separable_{1}x{1}_{2}'.format(self.prefix, self.kernel_size, repeat)
                x = SeparableConv2D(self.filters,
                                    kernel_size=self.kernel_size,
                                    kernel_initializer='he_normal',
                                    strides=strides,
                                    padding='same',
                                    use_bias=False,
                                    name=name)(x)

                name = '{0}/bn_sep_{1}x{1}_{2}'.format(self.prefix, self.kernel_size, repeat)
                x = BatchNormalization(name=name)(x)

        return x


class SqueezeChannels:
    """Use 1x1 convolutions to squeeze the input channels to match the cells filter count"""

    def __init__(self, filters, prefix, conv_suffix='1x1', bn_suffix='beginning_bn'):
        self.filters = filters
        self.conv_name = '{}/{}'.format(prefix, conv_suffix)
        self.bn_name = '{}/{}'.format(prefix, bn_suffix)

    def __call__(self, x):
        with K.name_scope('filter_squeeze'):
            x = Activation('relu')(x)
            x = Convolution2D(self.filters, 1, kernel_initializer='he_normal', use_bias=False,
                              name=self.conv_name)(x)
            x = BatchNormalization(name=self.bn_name)(x)

            return x


class Fit:
    """Make the cell outputs compatible"""

    def __init__(self, filters, target_layer, prefix):
        self.filters = filters
        self.target_layer = target_layer
        self.prefix = prefix

    def __call__(self, x):
        if x is None:
            return self.target_layer

        elif int(x.shape[2]) != int(self.target_layer.shape[2]):
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

            with K.name_scope('reduce_shape'):
                x = Activation('relu')(x)

                p1 = AveragePooling2D(pool_size=1, strides=(2, 2), padding='valid')(x)
                p1 = Convolution2D(self.filters // 2,
                                   kernel_size=1,
                                   kernel_initializer='he_normal',
                                   padding='same',
                                   use_bias=False,
                                   name='{}/path1_conv'.format(self.prefix))(p1)

                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
                p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = AveragePooling2D(pool_size=1, strides=2, padding='valid')(p2)

                p2 = Convolution2D(self.filters // 2,
                                   kernel_size=1,
                                   kernel_initializer='he_normal',
                                   padding='same',
                                   use_bias=False,
                                   name='{}/path2_conv'.format(self.prefix))(p2)

                x = Concatenate(axis=concat_axis)([p1, p2])
                x = BatchNormalization(name='{}/final_path_bn'.format(self.prefix))(x)

                return x
        else:
            return SqueezeChannels(self.filters, prefix=self.prefix, conv_suffix='prev_1x1', bn_suffix='prev_bn')(x)


class NormalCell:
    def __init__(self, filters, prefix):
        self.filters = filters
        self.prefix = prefix

    def __call__(self, prev, cur):
        with K.name_scope('normal'):
            cur = SqueezeChannels(self.filters, self.prefix)(cur)
            prev = Fit(self.filters, cur, self.prefix)(prev)
            output = [prev]

            with K.name_scope('comb_iter_0'):
                prefix = '{}/comb_iter_0'.format(self.prefix)
                output.append(Add()([Separable(self.filters, 5, prefix='{}/left'.format(prefix))(cur),
                                     Separable(self.filters, 3, prefix='{}/right'.format(prefix))(prev)]))

            with K.name_scope('comb_iter_1'):
                prefix = '{}/comb_iter_1'.format(self.prefix)
                output.append(Add()([Separable(self.filters, 5, prefix='{}/left'.format(prefix))(prev),
                                     Separable(self.filters, 3, prefix='{}/right'.format(prefix))(prev)]))

            with K.name_scope('comb_iter_2'):
                output.append(Add()([AveragePooling2D(pool_size=3, strides=1, padding='same')(cur),
                                     prev]))

            with K.name_scope('comb_iter_3'):
                output.append(Add()([AveragePooling2D(pool_size=3, strides=1, padding='same')(prev),
                                     AveragePooling2D(pool_size=3, strides=1, padding='same')(prev)]))

            with K.name_scope('comb_iter_4'):
                prefix = '{}/comb_iter_4'.format(self.prefix)
                output.append(Add()([Separable(self.filters, 3, prefix='{}/left'.format(prefix))(cur),
                                     cur]))

            return Concatenate()(output)


class ReductionCell:
    def __init__(self, filters, prefix):
        self.filters = filters
        self.prefix = prefix

    def __call__(self, prev, cur):
        with K.name_scope('reduce'):
            prev = Fit(self.filters, cur, self.prefix)(prev)
            cur = SqueezeChannels(self.filters, self.prefix)(cur)

            # Full in
            with K.name_scope('comb_iter_0'):
                prefix = '{}/comb_iter_0'.format(self.prefix)
                add_0 = Add()([Separable(self.filters, 5, strides=2, prefix='{}/left'.format(prefix))(cur),
                               Separable(self.filters, 7, strides=2, prefix='{}/right'.format(prefix))(prev)])

            with K.name_scope('comb_iter_1'):
                prefix = '{}/comb_iter_1'.format(self.prefix)
                add_1 = Add()([MaxPooling2D(3, strides=2, padding='same')(cur),
                               Separable(self.filters, 7, strides=2, prefix='{}/right'.format(prefix))(prev)])

            with K.name_scope('comb_iter_2'):
                prefix = '{}/comb_iter_2'.format(self.prefix)
                add_2 = Add()([AveragePooling2D(3, strides=2, padding='same')(cur),
                               Separable(self.filters, 5, strides=2, prefix='{}/right'.format(prefix))(prev)])

            # Reduced after stride
            with K.name_scope('comb_iter_3'):
                add_3 = Add()([AveragePooling2D(3, strides=1, padding='same')(add_0), add_1])

            with K.name_scope('comb_iter_4'):
                prefix = '{}/comb_iter_4'.format(self.prefix)
                add_4 = Add()([Separable(self.filters, 3, strides=1, prefix='{}/left'.format(prefix))(add_0),
                               MaxPooling2D(3, strides=2, padding='same')(cur)])

            return Concatenate()([add_1, add_2, add_3, add_4])


class AuxiliaryTop:
    def __init__(self, classes, prefix):
        self.classes = classes
        self.prefix = '{}/aux_logits'.format(prefix)

    def __call__(self, x):
        with K.name_scope('auxiliary_output'):
            x = Activation('relu')(x)
            x = AveragePooling2D(5, strides=3, padding='valid')(x)
            x = Convolution2D(128, kernel_size=1, padding='same',
                              kernel_initializer='he_normal', use_bias=False,
                              name='{}/proj'.format(self.prefix))(x)
            x = BatchNormalization(name='{}/aux_bn0'.format(self.prefix))(x)

            x = Activation('relu')(x)
            x = Convolution2D(768, kernel_size=int(x.shape[2]), padding='valid',
                              kernel_initializer='he_normal', use_bias=False,
                              name='{}/Conv'.format(self.prefix))(x)
            x = BatchNormalization(name='{}/aux_bn1'.format(self.prefix))(x)

            x = Activation('relu')(x)
            x = GlobalAveragePooling2D()(x)

            x = Dense(self.classes, activation='softmax', name='{}/FC'.format(self.prefix))(x)

        return x


def NASNetA(include_top=True,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            add_aux_output=False,
            stem=None,
            stem_filters=96,
            num_cell_repeats=18,
            penultimate_filters=768,
            num_classes=10,
            num_reduction_cells=2,
            dropout_rate=0.5):
    aux_outputs = None

    if input_tensor is None:
        input_tensor = Input(input_shape)

    if pooling is None:
        pooling = 'avg'

    if stem is None:
        stem = ImagenetStem

    filters = int(penultimate_filters / ((2 ** num_reduction_cells) * 6))
    prev, cur = stem(filters=filters, stem_filters=stem_filters)(input_tensor)

    for repeat in range(num_reduction_cells + 1):
        if repeat == num_reduction_cells and add_aux_output:
            prefix = 'aux_{}'.format(repeat * num_cell_repeats - 1)
            aux_outputs = AuxiliaryTop(num_classes, prefix=prefix)(cur)

        if repeat > 0:
            filters *= 2
            prev, cur = cur, prev
            cur = ReductionCell(filters, prefix='reduction_cell_{}'.format(repeat - 1))(cur, prev)

        for cell_index in range(num_cell_repeats):
            prev, cur = cur, prev
            cur = NormalCell(filters, prefix='cell_{}'.format(cell_index + repeat * num_cell_repeats))(cur, prev)

    with K.name_scope('final_layer'):
        x = Activation('relu', name='last_relu')(cur)

        if include_top:
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dropout(rate=dropout_rate)(x)
            outputs = Dense(num_classes, activation='softmax', name='final_layer/FC')(x)
        else:
            if pooling == 'avg':
                outputs = GlobalAveragePooling2D(name='avg_pool')(x)
            elif pooling == 'max':
                outputs = GlobalMaxPooling2D(name='max_pool')(x)
            else:
                outputs = None
                raise Exception('Supported options for pooling: `avg` or `max` given pooling: {}'.format(pooling))

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = input_tensor

    model_name = 'NASNet-A_{}@{}'.format(num_cell_repeats, penultimate_filters)
    if add_aux_output:
        return Model(inputs, [outputs, aux_outputs], name='{}_with_auxiliary_output'.format(model_name))
    else:
        return Model(inputs, outputs, name=model_name)


def load_weights_from_tf_checkpoint(model, checkpoint_file):
    print('Load weights from tensorflow checkpoint')
    progbar = Progbar(target=len(model.layers))

    reader = tf.train.NewCheckpointReader(checkpoint_file)
    for index, layer in enumerate(model.layers):
        progbar.update(current=index)

        if isinstance(layer, layers.convolutional.SeparableConv2D):
            depthwise = reader.get_tensor('{}/depthwise_weights'.format(layer.name))
            pointwise = reader.get_tensor('{}/pointwise_weights'.format(layer.name))
            layer.set_weights([depthwise, pointwise])
        elif isinstance(layer, layers.convolutional.Convolution2D):
            weights = reader.get_tensor('{}/weights'.format(layer.name))

            layer.set_weights([weights])
        elif isinstance(layer, layers.BatchNormalization):
            beta = reader.get_tensor('{}/beta'.format(layer.name))
            gamma = reader.get_tensor('{}/gamma'.format(layer.name))
            moving_mean = reader.get_tensor('{}/moving_mean'.format(layer.name))
            moving_variance = reader.get_tensor('{}/moving_variance'.format(layer.name))

            layer.set_weights([gamma, beta, moving_mean, moving_variance])
        elif isinstance(layer, layers.Dense):
            weights = reader.get_tensor('{}/weights'.format(layer.name))
            biases = reader.get_tensor('{}/biases'.format(layer.name))

            layer.set_weights([weights[:, 1:], biases[1:]])


def cifar10(include_top=True, input_tensor=None, aux_output=False):
    """Table 1: CIFAR-10: 6 @ 768, 3.3M parameters"""

    return NASNetA(include_top=include_top,
                   input_tensor=input_tensor,
                   input_shape=(32, 32, 3),
                   num_cell_repeats=6,
                   add_aux_output=aux_output,
                   stem=CifarStem,
                   stem_filters=96,
                   penultimate_filters=768,
                   num_classes=10)


def large(include_top=True, input_tensor=None, aux_output=False, load_weights=False):
    """Table 2: NASNet-A (6 @ 4032), 88.9M parameters"""

    model = NASNetA(include_top=include_top,
                    input_tensor=input_tensor,
                    input_shape=(331, 331, 3),
                    num_cell_repeats=6,
                    add_aux_output=aux_output,
                    stem=ImagenetStem,
                    stem_filters=96,
                    penultimate_filters=4032,
                    num_classes=1000)

    if load_weights:
        origin = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
        path = get_file('nasnet_large', origin=origin, extract=True, md5_hash='5286bdbb29bab27c4d3431c70f8becf9')
        checkpoint_file = os.path.join(path, '..', 'model.ckpt')

        load_weights_from_tf_checkpoint(model, checkpoint_file)

    return model


def mobile(include_top=True, input_tensor=None, aux_output=False, load_weights=False):
    """Table 3: NASNet-A (4 @ 1056), 5.3M parameters"""

    model = NASNetA(include_top=include_top,
                    input_tensor=input_tensor,
                    input_shape=(224, 224, 3),
                    num_cell_repeats=4,
                    add_aux_output=aux_output,
                    stem=ImagenetStem,
                    stem_filters=32,
                    penultimate_filters=1056,
                    num_classes=1000)

    if load_weights:
        origin = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz'
        path = get_file('nasnet_mobile', origin=origin, extract=True, md5_hash='7777886f3de3d733d3a6bf8b80e63555')
        checkpoint_file = os.path.join(path, '..', 'model.ckpt')

        load_weights_from_tf_checkpoint(model, checkpoint_file)

    return model


if __name__ == '__main__':
    model = mobile()
    model.summary()
