"""Implementation of NASNet-A"""
from keras import Input, Model
from keras import backend as K
from keras.engine import get_source_inputs
from keras.layers import Activation, SeparableConv2D, BatchNormalization, Dropout
from keras.layers import ZeroPadding2D, GlobalMaxPooling2D, Cropping2D
from keras.layers import AveragePooling2D, MaxPooling2D, Add, Concatenate
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense


class Separable:
    def __init__(self, filters, kernel_size, strides=1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def __call__(self, x):
        with K.name_scope('Sep_{0}x{0}_stride_{1}'.format(self.kernel_size,
                                                          self.strides)):
            for repeat in range(2):
                strides = self.strides if repeat == 1 else 1

                x = Activation('relu')(x)
                x = SeparableConv2D(self.filters,
                                    kernel_size=self.kernel_size,
                                    kernel_initializer='he_normal',
                                    strides=strides,
                                    padding='same')(x)
                x = BatchNormalization()(x)

        return x


class Fit:
    """Make the cell outputs compatible

    1. Use 1x1 convolutions to squeeze the input channels to match the cells filter count
    2. Use 2x2 average pooling to reduce layer shape if they do not match
    """

    def __init__(self, filters):
        self.filters = filters

    def squeeze(self, x, name):
        with K.name_scope('squeeze_{}_to_{}'.format(name, self.filters)):
            x = Activation('relu')(x)
            x = Convolution2D(self.filters, kernel_size=1, kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)

        return x

    def half_shape(self, x):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        with K.name_scope('half_shape'):
            p1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
            p1 = Convolution2D(self.filters // 2, kernel_size=1, kernel_initializer='he_normal',
                               padding='same', use_bias=False)(p1)

            p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
            p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
            p2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(p2)
            p2 = Convolution2D(self.filters // 2, kernel_size=1, kernel_initializer='he_normal', padding='same')(p2)

            x = Concatenate(axis=concat_axis)([p1, p2])
            x = BatchNormalization()(x)

            return x

    def __call__(self, prev, cur):
        with K.name_scope('fit'):
            diff = int(prev.shape[2]) - int(cur.shape[2])

            if diff > 0:
                prev = self.half_shape(prev)
            elif diff < 0:
                cur = self.half_shape(cur)

            prev = self.squeeze(prev, 'prev')
            cur = self.squeeze(cur, 'cur')

            return prev, cur


class NormalCell:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, prev, cur):
        with K.name_scope('normal'):
            output = list()
            prev, cur = Fit(self.filters)(prev, cur)

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
            prev, cur = Fit(self.filters)(prev, cur)

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


def NASNetA(include_top=True,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            num_stem_filters=96,
            num_cell_repeats=18,
            penultimate_filters=768,
            num_classes=10,
            num_reduction_cells=2,
            dropout_rate=0.5) -> Model:
    if input_tensor is None:
        input_tensor = Input(input_shape)

    if pooling is None:
        pooling = 'avg'

    with K.name_scope('stem'):
        cur = Convolution2D(num_stem_filters, 3, kernel_initializer='he_normal', padding='same')(input_tensor)
        cur = BatchNormalization()(cur)

        prev = cur

    num_filters = int(penultimate_filters / ((2 ** num_reduction_cells) * 6))

    for repeat in range(num_reduction_cells + 1):
        if repeat > 0:
            num_filters *= 2
            prev, cur = cur, prev
            cur = ReductionCell(num_filters)(cur, prev)

        for cell_index in range(num_cell_repeats):
            prev, cur = cur, prev
            cur = NormalCell(num_filters)(cur, prev)

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

    return Model(inputs, outputs, name='NASNet-A_{}@{}'.format(num_cell_repeats, penultimate_filters))


if __name__ == '__main__':
    # Table 1: CIFAR-10: 6 @ 768
    model = NASNetA(input_shape=(32, 32, 3),
                    num_stem_filters=96,
                    num_cell_repeats=6,
                    penultimate_filters=768,
                    num_classes=10)

    print(model.summary())
