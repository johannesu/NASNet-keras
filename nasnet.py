"""Implementation of NASNet-A"""
from __future__ import division

from keras import Input, Model
from keras import backend as K
from keras.engine import get_source_inputs
from keras.layers import Activation, SeparableConv2D, BatchNormalization, Dropout
from keras.layers import ZeroPadding2D, GlobalMaxPooling2D, Cropping2D
from keras.layers import AveragePooling2D, MaxPooling2D, Add, Concatenate
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense


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
                              kernel_initializer='he_normal', padding='valid', use_bias=False)(x)
            x = BatchNormalization()(x)

            prev = ReductionCell(self.filters // 4)(None, x)
            cur = ReductionCell(self.filters // 2)(x, prev)

        return prev, cur


class Separable:
    def __init__(self, filters, kernel_size, strides=1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def __call__(self, x):
        with K.name_scope('Sep_{0}x{0}_stride_{1}'.format(self.kernel_size,
                                                          self.strides)):
            for repeat in range(2):
                strides = self.strides if repeat == 0 else 1

                x = Activation('relu')(x)
                x = SeparableConv2D(self.filters,
                                    kernel_size=self.kernel_size,
                                    kernel_initializer='he_normal',
                                    strides=strides,
                                    padding='same',
                                    use_bias=False)(x)
                x = BatchNormalization()(x)

        return x


class SqueezeChannels:
    """Use 1x1 convolutions to squeeze the input channels to match the cells filter count"""

    def __init__(self, filters):
        self.filters = filters

    def __call__(self, x):
        x = Activation('relu')(x)
        x = Convolution2D(self.filters, kernel_size=1, kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization()(x)

        return x


class Fit:
    """Make the cell outputs compatible"""

    def __init__(self, filters, target_layer):
        self.filters = filters
        self.target_layer = target_layer

    def __call__(self, x):
        if x is None:
            return self.target_layer

        elif int(x.shape[2]) != int(self.target_layer.shape[2]):
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

            with K.name_scope('reduce_shape'):
                p1 = AveragePooling2D(pool_size=1, strides=(2, 2), padding='valid')(x)
                p1 = Convolution2D(self.filters // 2, kernel_size=1, kernel_initializer='he_normal',
                                   padding='same', use_bias=False)(p1)

                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
                p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = AveragePooling2D(pool_size=1, strides=(2, 2), padding='valid')(p2)
                p2 = Convolution2D(self.filters // 2, kernel_size=1,
                                   kernel_initializer='he_normal', padding='same', use_bias=False)(p2)

                x = Concatenate(axis=concat_axis)([p1, p2])
                x = BatchNormalization()(x)

                return x
        else:
            return SqueezeChannels(self.filters)(x)


class NormalCell:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, prev, cur):
        with K.name_scope('normal'):
            output = list()

            cur = SqueezeChannels(self.filters)(cur)
            prev = Fit(self.filters, cur)(prev)

            with K.name_scope('block_1'):
                output.append(Add()([Separable(self.filters, 3)(cur),
                                     cur]))

            with K.name_scope('block_2'):
                output.append(Add()([Separable(self.filters, 3)(prev),
                                     Separable(self.filters, 5)(cur)]))

            with K.name_scope('block_3'):
                output.append(Add()([AveragePooling2D(pool_size=3, strides=1, padding='same')(cur),
                                     prev]))

            with K.name_scope('block_4'):
                output.append(Add()([AveragePooling2D(pool_size=3, strides=1, padding='same')(prev),
                                     AveragePooling2D(pool_size=3, strides=1, padding='same')(prev)]))

            with K.name_scope('block_5'):
                output.append(Add()([Separable(self.filters, 5)(prev),
                                     Separable(self.filters, 3)(prev)]))

            output.append(prev)

            return Concatenate()(output)


class ReductionCell:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, prev, cur):
        with K.name_scope('reduce'):
            cur = SqueezeChannels(self.filters)(cur)
            prev = Fit(self.filters, cur)(prev)

            # Full in
            with K.name_scope('block_1'):
                add_1 = Add()([Separable(self.filters, 7, strides=2)(prev),
                               Separable(self.filters, 5, strides=2)(cur)])

            with K.name_scope('block_2'):
                add_2 = Add()([MaxPooling2D(3, strides=2, padding='same')(cur),
                               Separable(self.filters, 7, strides=2)(prev)])

            with K.name_scope('block_3'):
                add_3 = Add()([AveragePooling2D(3, strides=2, padding='same')(cur),
                               Separable(self.filters, 5, strides=2)(prev)])

            # Reduced after stride
            with K.name_scope('block_4'):
                add_4 = Add()([MaxPooling2D(3, strides=2, padding='same')(cur),
                               Separable(self.filters, 3, strides=1)(add_1)])

            with K.name_scope('block_5'):
                add_5 = Add()([AveragePooling2D(3, strides=1, padding='same')(add_1),
                               add_2])

            return Concatenate()([add_2, add_3, add_4, add_5])


class AuxiliaryTop:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, x):
        with K.name_scope('auxiliary_output'):
            x = Activation('relu')(x)
            x = AveragePooling2D(5, strides=3, padding='valid')(x)
            x = Convolution2D(128, kernel_size=1, padding='same',
                              kernel_initializer='he_normal', use_bias=False, name='aux_conv1')(x)
            x = BatchNormalization(name='aux_bn1')(x)

            x = Activation('relu')(x)
            x = Convolution2D(768, kernel_size=int(x.shape[2]), padding='valid',
                              kernel_initializer='he_normal', use_bias=False, name='aux_conv2')(x)
            x = BatchNormalization(name='aux_bn2')(x)

            x = Activation('relu')(x)
            x = GlobalAveragePooling2D()(x)

            x = Dense(self.classes, activation='softmax', name='aux_output')(x)

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
    aux_output = None

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
            aux_output = AuxiliaryTop(num_classes)(cur)

        if repeat > 0:
            filters *= 2
            prev, cur = cur, prev
            cur = ReductionCell(filters)(cur, prev)

        for cell_index in range(num_cell_repeats):
            prev, cur = cur, prev
            cur = NormalCell(filters)(cur, prev)

    with K.name_scope('final_layer'):
        x = Activation('relu', name='last_relu')(cur)
        x = Dropout(rate=dropout_rate)(x)

        if include_top:
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
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
        return Model(inputs, [outputs, aux_output], name='{}_with_auxiliary_output'.format(model_name))
    else:
        return Model(inputs, outputs, name=model_name)


def cifar10(include_top=True, input_tensor=None, add_aux_output=False):
    """Table 1: CIFAR-10: 6 @ 768, 3.3M parameters"""

    return NASNetA(include_top=include_top,
                   input_tensor=input_tensor,
                   input_shape=(32, 32, 3),
                   num_cell_repeats=6,
                   add_aux_output=add_aux_output,
                   stem=CifarStem,
                   stem_filters=96,
                   penultimate_filters=768,
                   num_classes=10)


def large(include_top=True, input_tensor=None, add_aux_output=False):
    """Table 2: NASNet-A (6 @ 4032), 88.9M parameters"""

    return NASNetA(include_top=include_top,
                   input_tensor=input_tensor,
                   input_shape=(331, 331, 3),
                   num_cell_repeats=6,
                   add_aux_output=add_aux_output,
                   stem=ImagenetStem,
                   stem_filters=96,
                   penultimate_filters=4032,
                   num_classes=1000)


def mobile(include_top=True, input_tensor=None, add_aux_output=False):
    """Table 3: NASNet-A (4 @ 1056), 5.3M parameters"""

    return NASNetA(include_top=include_top,
                   input_tensor=input_tensor,
                   input_shape=(224, 224, 3),
                   num_cell_repeats=4,
                   add_aux_output=add_aux_output,
                   stem=ImagenetStem,
                   stem_filters=32,
                   penultimate_filters=1056,
                   num_classes=1000)


if __name__ == '__main__':
    model = large()
    model.summary()
