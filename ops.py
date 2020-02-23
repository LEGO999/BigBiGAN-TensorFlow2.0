import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from utils import orthogonal_regularizer_fully, orthogonal_regularizer
from tensorflow.keras.layers import Wrapper

##################################################################################
# Initialization
##################################################################################

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(1.0) / relu = sqrt(2), the others = 1.0

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) / orthogonal_regularizer_fully(0.0001)

weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
# weight_init = tf.initializers.orthogonal()
# Original version of BigGAN is tf.initializers.orthogonal()
weight_regularizer = None #orthogonal_regularizer(0.0001)
weight_regularizer_fully = None #orthogonal_regularizer_fully(0.0001)
# if FLAGS.orth_reg:
#     weight_regularizer = orthogonal_regularizer(0.0001)
#     weight_regularizer_fully = orthogonal_regularizer_fully(0.0001)
# else:
#     weight_regularizer = None
#     weight_regularizer_fully = None


class SpectralNormalization(Wrapper):   # Wrapper takes another layer and argument it
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, config, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)  # inherits the properties from wrapper,
        # including layers and kwargs.
        self.device = config.device

    def build(self, input_shape):
        """Build `Layer`"""
        with tf.device('{}:*'.format(self.device)):
            if not self.layer.built:  # Question: why we must detect 'built' here?
                # Because i check the source in github, it's self.built=True everywhere and no False.
                self.layer.build(input_shape)

                if not hasattr(self.layer, 'kernel'):  # E.g. We could check them under the source of Github.
                    # Normally, it will be like self.add_weights or something.
                    raise ValueError(
                        '`SpectralNormalization` must wrap a layer that'
                        ' contains a `kernel` for weights')

                self.w = self.layer.kernel
                self.w_shape = self.w.shape.as_list()  # If something is a tensorshape, you could use as_list()
                # method to return a list of shape.
                self.u = self.add_weight(
                    shape=tuple([1, self.w_shape[-1]]),
                    initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                    # truncatedNormal is a initializer which is similar to the initializer normal,
                    # but the samples outside 2 stddev will be redrawn. A quite good initializer from beginning.
                    name='sn_u',
                    trainable=False, # These weights are by default not trainable.
                    dtype=tf.float32)

        super(SpectralNormalization, self).build()  # inherits the build method from wrappers.

    #@tf.function
    def call(self, inputs):
        """Call `Layer`"""
        # Recompute weights for each forward pass
        self._compute_weights()   # every time generates the new normalized weights when is called.
        output = self.layer(inputs)
        return output  # Output of the layer.

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])  # reshape the weights of the layer into 2 dimensions.
        # ==> -1*the last dimension of the weights
        eps = 1e-12
        _u = tf.identity(self.u)  # Return a tensor which has the same shape and contents as input.
        _v = tf.matmul(_u, tf.transpose(w_reshaped))  # (1*self.w_shape[-1])*(self.w_shape[-1]*(-1))
        _v = tf.nn.l2_normalize(_v, epsilon=eps)  # It's now call tf.math.l2_normalize
        # normalizes along dimension axis using an L2 norm. For a 1D signal, output = x / sqrt(max(sum(x**2), epsilon))
        # axis=0 along columns, axis=1 along rows. Epsilon is the lower bound of the normalization.
        # Return a tensor with same shape.
        _u = tf.matmul(_v, w_reshaped)  # Finally, we get dimensionality of [1,self.w_shape[-1]
        _u = tf.nn.l2_normalize(_u,epsilon=eps)

        # Stop gradient
        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
        self.u.assign(_u)
        sigma = tf.matmul(tf.matmul(_v, w_reshaped), tf.transpose(_u))

        self.layer.kernel = self.w / sigma

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())  # Return a output shape.


class Attention(tf.keras.Model):
    'https://stackoverflow.com/questions/50819931/self-attention-gan-in-keras'
    def __init__(self, ch, config):
        super(Attention, self).__init__()
        self.filters_f_g_h = ch // 8
        self.filters_v = ch
        self.f_l = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.filters_f_g_h, kernel_size=1, strides=1, use_bias=True), config=config)
        self.g_l = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.filters_f_g_h, kernel_size=1, strides=1, use_bias=True), config=config)
        self.h_l = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.filters_f_g_h, kernel_size=1, strides=1, use_bias=True), config=config)
        self.v_l = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.filters_v, kernel_size=1, strides=1, use_bias=True), config=config)
        self.gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)

    #@tf.function
    def __call__(self, inputs, training=False):
        def hw_flatten(x):
            return tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2], tf.shape(x)[-1]])

        f = self.f_l(inputs)

        g = self.g_l(inputs)

        h = self.h_l(inputs)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
        # tf.matmul could afford to have 2 up to 3d Tensor multiplication, i.e. [Batchsize, rows, columns]
        # But the output of the Conv2d layer is obviously 4D[Batchsize, rows, columns, numbers of fiters]
        # So we have to use hw_flatten to reshape the matrix at first.
        # And please be careful, matrices´s dimensions should be passed to each other.
        beta = tf.nn.softmax(s, axis=-1)  # attention map

        v = tf.matmul(beta, hw_flatten(h))
        v = tf.reshape(v, shape=[inputs.get_shape().as_list()[0],inputs.get_shape().as_list()[1],inputs.get_shape().as_list()[2], -1]) # [bs, h, w, C]

        o = self.v_l(v)

        output = self.gamma * o + inputs

        return output


##################################################################################
# Residual-block
##################################################################################


class resblock(tf.keras.Model):
    def __init__(self, channels, config, use_bias=True):
        super().__init__()
        with tf.name_scope('resblock'):
            self.conv0 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer=weight_init), config=config)
            self.conv1 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer=weight_init), config=config)

    #@tf.function
    def __call__(self, inputs, training=False):
        # res1
        x = self.conv0(inputs)
        x = tf.nn.relu(x)
        # res2
        x = self.conv1(x)

        return x+inputs

class resblock_up_condition_top(tf.keras.Model):

    def __init__(self, channels, config, use_bias=True):
        super().__init__()
        with tf.name_scope('resblock_up_condition_top'):
            self.cond_batchnorm0 = condition_batch_norm(channels=channels)
            self.upsampling = tf.keras.layers.UpSampling2D()
            self.conv1 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer= weight_init,
                                                                kernel_regularizer= weight_regularizer), config=config)
            self.cond_batchnorm1 = condition_batch_norm(channels=channels)
            self.conv2 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer= weight_init,
                                                                kernel_regularizer= weight_regularizer), config=config)
            self.skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                                                   kernel_size=1,
                                                                   strides=1,
                                                                   padding='SAME',
                                                                   use_bias=use_bias,
                                                                   kernel_initializer= weight_init,
                                                                   kernel_regularizer= weight_regularizer), config=config)

    ##@tf.function
    def __call__(self, inputs, z, training=False):
        # res1
        x = self.cond_batchnorm0(inputs, z, training=training)
        x = tf.nn.relu(x)
        x = self.upsampling(x)
        x = self.conv1(x)

        # res2
        x = self.cond_batchnorm1(x, z, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # skip
        x_init = self.upsampling(inputs)
        x_init = self.skip_conv(x_init)

        return x + x_init

class resblock_up_condition(tf.keras.Model):

    def __init__(self, channels, config, use_bias=True):
        super().__init__()
        with tf.name_scope('resblock_up_condition'):
            self.cond_batchnorm0 = condition_batch_norm(channels=channels * 2)
            self.upsampling = tf.keras.layers.UpSampling2D()
            self.conv1 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer= weight_init,
                                                                kernel_regularizer= weight_regularizer), config)
            self.cond_batchnorm1 = condition_batch_norm(channels=channels)
            self.conv2 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer= weight_init,
                                                                kernel_regularizer= weight_regularizer), config)
            self.skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                                                   kernel_size=1,
                                                                   strides=1,
                                                                   padding='SAME',
                                                                   use_bias=use_bias,
                                                                   kernel_initializer= weight_init,
                                                                   kernel_regularizer= weight_regularizer), config)

    ##@tf.function
    def __call__(self, inputs, z, training=False):
        # res1
        x = self.cond_batchnorm0(inputs, z, training=training)
        x = tf.nn.relu(x)
        x = self.upsampling(x)
        x = self.conv1(x)

        # res2
        x = self.cond_batchnorm1(x, z, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # skip
        x_init = self.upsampling(inputs)
        x_init = self.skip_conv(x_init)

        return x + x_init

class resblock_down(tf.keras.Model):
    def __init__(self, channels, config, use_bias=True):
        super().__init__()
        with tf.name_scope('resblock_down'):

            self.conv0 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer=weight_init), config=config)
            self.conv1 = SpectralNormalization(Conv2D(filters=channels,
                                                      kernel_size=3,
                                                      strides=1,
                                                      padding='SAME',
                                                      use_bias=use_bias,
                                                      kernel_initializer=weight_init), config=config)
            self.skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                                      kernel_size=1,
                                                      strides=1,
                                                      padding='SAME',
                                                      use_bias=use_bias,
                                                      kernel_initializer=weight_init), config=config)
            self.avg_pooling = tf.keras.layers.AveragePooling2D(padding='SAME')

    #@tf.function
    def __call__(self, inputs, training=False):
        # res1
        x = tf.nn.relu(inputs)
        x = self.conv0(x)
        # res2
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.avg_pooling(x)
        # skip
        x_init = self.skip_conv(inputs)
        x_init = self.avg_pooling(x_init)

        return x + x_init


class resblock_dense(tf.keras.Model):
    def __init__(self, units, config):
        super().__init__()
        with tf.name_scope('resblock_dense'):
            self.dense0 = SpectralNormalization(Dense(units=units, kernel_initializer= weight_init), config=config)
            self.dropout = tf.keras.layers.Dropout(0.2)
            self.dense1 = SpectralNormalization(Dense(units=units, kernel_initializer= weight_init), config=config)
            self.dense_skip = SpectralNormalization(Dense(units=units, kernel_initializer=weight_init), config=config)

    #@tf.function
    def __call__(self, inputs, training=False):

        l1 = self.dense0(inputs)
        l1 = self.dropout(l1, training=training)
        l1 = tf.nn.leaky_relu(l1)

        l2 = self.dense1(l1)
        l2 = self.dropout(l2, training=training)
        l2 = tf.nn.leaky_relu(l2)

        skip = self.dense_skip(inputs)
        skip = tf.nn.leaky_relu(skip)


        output = l2+skip

        return output


class resblock_dense_no_sn(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        with tf.name_scope('resblock_dense_no_sn'):
            self.dense0 = Dense(units=units, kernel_initializer=weight_init)
            self.dropout = tf.keras.layers.Dropout(0.2)
            self.dense1 = Dense(units=units, kernel_initializer=weight_init)
            self.dense_skip = Dense(units=units, kernel_initializer=weight_init)

    #@tf.function
    def __call__(self, inputs, training=False):

        l1 = self.dense0(inputs)
        l1 = self.dropout(l1, training=training)
        l1 = tf.nn.leaky_relu(l1)

        l2 = self.dense1(l1)
        l2 = self.dropout(l2, training=training)
        l2 = tf.nn.leaky_relu(l2)

        skip = self.dense_skip(inputs)
        skip = tf.nn.leaky_relu(skip)

        output = l2+skip

        return output



class bottleneck_s(tf.keras.Model):
    def __init__(self, filters, strides=1):
        super().__init__()
        self.filters = filters
        self.strides = strides
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False, kernel_initializer=weight_init)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', strides=strides, use_bias=False,kernel_initializer=weight_init)
        self.skip_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='SAME', kernel_initializer=weight_init)

    #@tf.function
    def __call__(self, inputs, training=False):
        l1 = self.conv0(inputs)
        l1 = self.bn0(l1, training=training)
        l1 = tf.nn.relu(l1)

        l2 = self.conv1(l1)
        l2 = self.bn1(l2, training=training)
        l2 = tf.nn.relu(l2)

        # Project input if necessary
        if (self.strides > 1) or (self.filters != inputs.get_shape().as_list()[-1]):
            x_shortcut = self.skip_conv(inputs)
        else:
            x_shortcut = inputs

        return l2 + x_shortcut

class bottleneck_rev_s(tf.keras.Model):
    def __init__(self, ch, strides=1):
        super().__init__()
        self.unit = bottleneck_s(filters=ch//2, strides=strides)

    def __call__(self, inputs, training=False):
        # split with 2 parts and along axis=3
        x1, x2 = tf.split(inputs, 2, 3)
        y1 = x1 + self.unit(x2, training=training)
        y2 = x2

        #concatenate y2 and y1 along axis=3
        return tf.concat([y2, y1], axis=3)

def pool_and_double_channels(x, pool_stride):

    if pool_stride > 1:
        x = tf.nn.avg_pool2d(x, ksize=pool_stride,
                                        strides=pool_stride,
                                        padding='SAME')
    return tf.pad(x, [[0, 0], [0, 0], [0, 0],
                      [x.get_shape().as_list()[3] // 2, x.get_shape().as_list()[3] // 2]])



##################################################################################
# Normalization function
##################################################################################


class condition_batch_norm(tf.keras.Model):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.decay = 0.9
        self.epsilon = 1e-05
        self.test_mean = tf.Variable(tf.zeros([channels]), dtype=tf.float32, trainable=False)
        self.test_var = tf.Variable(tf.ones([channels]), dtype=tf.float32, trainable=False)
        self.beta0 = tf.keras.layers.Dense(units=channels, use_bias=True, kernel_initializer=weight_init, kernel_regularizer= weight_regularizer_fully)
        self.gamma0 = tf.keras.layers.Dense(units=channels, use_bias=True, kernel_initializer=weight_init, kernel_regularizer= weight_regularizer_fully)

    #@tf.function
    def __call__(self, x, z, training=False):

        beta0 = self.beta0(z)

        gamma0 = self.gamma0(z)

        beta = tf.reshape(beta0, shape=[-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma0, shape=[-1, 1, 1, self.channels])

        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            self.test_mean.assign(self.test_mean * self.decay + batch_mean * (1 - self.decay))
            self.test_var.assign(self.test_var * self.decay + batch_var * (1 - self.decay))

            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, self.epsilon)
        else:
            return tf.nn.batch_normalization(x, self.test_mean, self.test_var, beta, gamma, self.epsilon)



