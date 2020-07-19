import tensorflow as tf
import ops
from tensorflow.python.keras.layers import Dense, Conv2D, BatchNormalization, Conv2DTranspose, Flatten


class BIGBIGAN_G(tf.keras.Model):
    def __init__(self, config, weight_init):
        super().__init__()
        with tf.device('{}:*'.format(config.device)):
            with tf.name_scope('generator'):
                if config.dataset in ['mnist', 'fashion_mnist']:  # color channel of the dataset
                    self.c_dim = 1
                else:
                    self.c_dim = 3
                self.weight_init = weight_init
                self.num_cont_noise = config.num_cont_noise
                self.ch = config.gen_disc_ch * 4
                if config.conditional:
                    self.emb = tf.keras.layers.Embedding(config.num_classes, config.num_emb)
                self.dense0 = ops.SpectralNormalization(Dense(units=4 * 4 * self.ch,
                                                              use_bias=True,
                                                              kernel_initializer=self.weight_init,
                                                              kernel_regularizer=ops.weight_regularizer_fully),
                                                        config=config)
                self.res_up0 = ops.resblock_up_condition_top(channels=self.ch, config=config, weight_init=self.weight_init, use_bias=False)
                self.res_up1 = ops.resblock_up_condition(channels=self.ch//2, config=config, weight_init=self.weight_init, use_bias=False)
                self.bn0 = BatchNormalization()
                self.att0 = ops.Attention(ch=self.ch // 2, config=config)
                self.res_up2 = ops.resblock_up_condition(channels=self.ch//4, config=config, weight_init=self.weight_init, use_bias=False)
                self.bn1 = BatchNormalization()
                self.conv0 = ops.SpectralNormalization(Conv2DTranspose(filters=self.c_dim,
                                                                       kernel_size=3,
                                                                       strides=1,
                                                                       padding='SAME',
                                                                       use_bias=False,
                                                                       kernel_initializer=self.weight_init,
                                                                       kernel_regularizer=ops.weight_regularizer), config=config)

    def __call__(self, cont_noise, label=None, training=False):

        z_split = tf.split(cont_noise, num_or_size_splits=[self.num_cont_noise//4] * 4, axis=-1)

        n_split_0 = z_split[0]
        n_split_1 = z_split[1]
        n_split_2 = z_split[2]
        n_split_3 = z_split[3]

        if label is not None:
            label = self.emb(label)
            n_split_1 = tf.concat([z_split[1], label], axis=-1)
            n_split_2 = tf.concat([z_split[2], label], axis=-1)
            n_split_3 = tf.concat([z_split[3], label], axis=-1)

        # Fully connected
        l1 = self.dense0(n_split_0)  # 4*4*4*ch
        l1 = tf.reshape(l1, shape=[-1, 4, 4, self.ch])  #[-1, 4, 4, 4*ch]

        l2 = self.res_up0(l1, n_split_1, training=training)  # 8*8 4*ch

        l3 = self.res_up1(l2, n_split_2, training=training)  # 16*16 2*ch

        # Non-local layer
        l4 = self.att0(l3, training=training)

        l5 = self.res_up2(l4, n_split_3, training=training)
        l5 = self.bn0(l5, training=training)
        l5 = tf.nn.relu(l5)

        l6 = self.conv0(l5)

        # Output layer
        output = tf.nn.sigmoid(l6)

        return output


class BIGBIGAN_D_F(tf.keras.Model):
    def __init__(self, config, weight_init):
        super().__init__()
        with tf.device('{}:*'.format(config.device)):
            with tf.name_scope('discriminator_f'):
                self.weight_init = weight_init
                self.ch = config.gen_disc_ch
                if config.conditional:
                    self.emb = tf.keras.layers.Embedding(config.num_classes, self.ch*4)
                self.res_down0 = ops.resblock_down(channels=self.ch, config=config, weight_init=self.weight_init, use_bias=False)
                self.bn0 = BatchNormalization()
                self.att0 = ops.Attention(ch=self.ch, config=config)
                self.res_down1 = ops.resblock_down(channels=self.ch*2, config=config, weight_init=self.weight_init, use_bias=False)
                self.res_down2 = ops.resblock_down(channels=self.ch*4, config=config, weight_init=self.weight_init, use_bias=False)
                self.res0 = ops.resblock(channels=self.ch*4, config=config, weight_init=self.weight_init, use_bias=False)
                self.dense0 = Dense(units=1, use_bias=True, kernel_initializer=self.weight_init)

    def __call__(self, inputs, label=None, training=False):

        x = inputs

        l1 = self.res_down0(x)

        # non-local layer
        l2 = self.att0(l1, training=training)

        l3 = self.res_down1(l2)

        l4 = self.res_down2(l3)

        l5 = self.res0(l4)

        l6 = tf.nn.relu(l5)

        output1 = tf.math.reduce_sum(l6, axis=[1, 2])

        output2 = self.dense0(output1)

        # Projection discriminator
        if label is not None:
            label = self.emb(label)
            output2 += tf.reduce_sum(label*output1, axis=-1, keepdims=True)

        return output1, output2


class BIGBIGAN_D_H(tf.keras.Model):
    def __init__(self, config,weight_init):
        super().__init__()
        with tf.device('{}:*'.format(config.device)):
            with tf.name_scope('discriminator_h'):
                self.weight_init = weight_init
                self.flatten = Flatten()
                self.res_dense0 = ops.resblock_dense(units=64, weight_init=self.weight_init, config=config)
                self.res_dense1 = ops.resblock_dense(units=64, weight_init=self.weight_init, config=config)
                self.res_dense2 = ops.resblock_dense(units=64, weight_init=self.weight_init, config=config)
                self.dense0 = Dense(units=1, use_bias=True, kernel_initializer=self.weight_init)

    def __call__(self, inputs, training=False):
        l1 = self.flatten(inputs)
        l1 = self.res_dense0(l1, training=training)
        l2 = self.res_dense1(l1)

        output1 = self.res_dense2(l2)

        output2 = self.dense0(output1)

        return output1, output2


class BIGBIGAN_D_J(tf.keras.Model):
    def __init__(self, config,weight_init):
        super().__init__()
        with tf.device('{}:*'.format(config.device)):
            with tf.name_scope('discriminator_j'):
                self.weight_init = weight_init
                self.flatten = Flatten()
                self.res_dense0 = ops.resblock_dense(units=64, weight_init=self.weight_init, config=config)
                self.res_dense1 = ops.resblock_dense(units=64, weight_init=self.weight_init, config=config)
                self.res_dense2 = ops.resblock_dense(units=64, weight_init=self.weight_init, config=config)
                self.dense0 = Dense(units=1, kernel_initializer=self.weight_init)

    def __call__(self, inputs_from_f, inputs_from_h, training=False):
        inputs_from_f = tf.reshape(inputs_from_f, [tf.shape(inputs_from_f)[0], -1])
        inputs_from_h = tf.reshape(inputs_from_h, [tf.shape(inputs_from_h)[0], -1])

        l1 = tf.concat([inputs_from_f, inputs_from_h], axis=-1)
        l1 = self.flatten(l1)
        l1 = self.res_dense0(l1, training=training)

        l2 = self.res_dense1(l1, training=training)

        l3 = self.res_dense2(l2, training=training)

        output = self.dense0(l3)

        return output


class BIGBIGAN_E(tf.keras.Model):
    # Encoder should take higher resolution than Generator and Discriminator

    def __init__(self, config, weight_init):
        super().__init__()
        with tf.device('{}:*'.format(config.device)):
            with tf.name_scope('encoder'):
                if config.dataset in ['mnist', 'fashion_mnist']:  # color channel of the dataset
                    self.c_dim = 1
                else:
                    self.c_dim = 3
                self.weight_init = weight_init
                self.ch = config.en_ch
                self.cont_dim = config.num_cont_noise
                self.upsample = tf.keras.layers.UpSampling2D(2)
                self.conv0 = Conv2D(self.ch, (7, 7), strides=2, padding='SAME',
                                    kernel_initializer=self.weight_init)
                self.maxpooling0 = tf.keras.layers.MaxPooling2D((3, 3), strides=2)

                self.bot_rev0 = ops.bottleneck_rev_s(ch=self.ch, weight_init=self.weight_init)
                self.bot_rev1 = ops.bottleneck_rev_s(ch=self.ch, weight_init=self.weight_init)

                self.bot_rev2 = ops.bottleneck_rev_s(ch=self.ch * 2, weight_init=self.weight_init)
                self.bot_rev3 = ops.bottleneck_rev_s(ch=self.ch * 2, weight_init=self.weight_init)

                self.bot_rev4 = ops.bottleneck_rev_s(ch=self.ch * 4, weight_init=self.weight_init)
                self.bot_rev5 = ops.bottleneck_rev_s(ch=self.ch * 4, weight_init=self.weight_init)

                self.res_dense0 = ops.resblock_dense_no_sn(units=256, weight_init=self.weight_init)
                self.res_dense1 = ops.resblock_dense_no_sn(units=256, weight_init=self.weight_init)
                self.dense0 = Dense(units=config.num_cont_noise, kernel_initializer=self.weight_init)
                self.flatten = Flatten()

    def __call__(self, inputs, training=False):

        l1 = self.upsample(inputs)
        l1 = tf.reshape(l1, (-1, 64, 64, self.c_dim))
        l1 = self.conv0(l1)

        l2 = self.maxpooling0(l1)
        l2 = self.bot_rev0(l2, training=training)
        l2 = self.bot_rev1(l2, training=training)
        l2 = ops.pool_and_double_channels(l2, 2)

        l3 = self.bot_rev2(l2, training=training)
        l3 = self.bot_rev3(l3, training=training)
        l3 = ops.pool_and_double_channels(l3, 2)

        l4 = self.bot_rev4(l3, training=training)
        l4 = self.bot_rev5(l4, training=training)

        l5 = tf.math.reduce_mean(l4, axis=[1, 2], keepdims=True)

        l6 = self.flatten(l5)

        l7 = self.res_dense0(l6, training=training)

        l8 = self.res_dense1(l7, training=training)

        l9 = self.dense0(l8)

        return l9






