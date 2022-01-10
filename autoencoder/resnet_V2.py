import tensorflow as tf
from tensorflow import keras


class ResiduaBlock(keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResiduaBlock, self).__init__()

        self.residual_block = keras.Sequential([
            keras.layers.Conv2D(out_channels, kernel_size, stride, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(out_channels, kernel_size, stride, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU()
        ], name='ResidualBlock')

    def call(self, x):
        return x + self.residual_block(x)

    def build_graph(self):
        x = keras.layers.Input(shape=(256, 256, 3))
        return keras.models.Model(inputs=[x], outputs=self.call(x))


class ResNetEncoder(keras.Model):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 input_ch=3,
                 z_dim=10,
                 bUseMultiResSkips=True,
                 drop_out=None):
        super(ResNetEncoder, self).__init__()

        self.max_filters = 2 ** (n_levels + 3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        if drop_out:
            self.dropout = keras.layers.Dropout(drop_out)
        else:
            self.dropout = None

        self.input_conv = keras.Sequential([
            keras.layers.Conv2D(8, 3, 1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2)
        ], name='en_input_conv')

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                keras.Sequential([ResiduaBlock(n_filters_1, n_filters_1) for _ in range(n_ResidualBlock)]
                                 , name='res_block_' + str(i))
            )

            self.conv_list.append(
                keras.Sequential([
                    keras.layers.Conv2D(n_filters_2, 2, strides=2, padding='valid'),
                    keras.layers.BatchNormalization(),
                    keras.layers.LeakyReLU(0.2)
                ], name='conv_block_' + str(i))
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    keras.Sequential([
                        keras.layers.Conv2D(self.max_filters,
                                            ks, ks, padding='valid'),
                        keras.layers.BatchNormalization(),
                        keras.layers.LeakyReLU(0.2)
                    ], name='multi_skip_block_' + str(i))
                )
        self.output_conv = keras.layers.Conv2D(z_dim, 3, 1, padding='same')

    def call(self, x):
        if self.dropout:
            x = self.dropout(x)

        x = self.input_conv(x)

        skips = []

        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        return x

    def build_graph(self):
        x = keras.layers.Input(shape=(256, 256, 3))
        return keras.models.Model(inputs=[x], outputs=self.call(x))


class ResNetDecoder(keras.Model):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 output_channels=3,
                 bUseMultiResSkips=True):
        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels + 3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        self.input_conv = keras.Sequential([
            keras.layers.Conv2D(self.max_filters, 3, 1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2)
        ], name='de_input_conv')

        for i in range(n_levels):
            n_filters_0 = 2 ** (self.n_levels - i + 3)
            n_filters_1 = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                keras.Sequential([ResiduaBlock(n_filters_1, n_filters_1)
                                  for _ in range(n_ResidualBlock)], name='de_res_block_' + str(i))
            )

            self.conv_list.append(
                keras.Sequential([
                    keras.layers.Conv2DTranspose(n_filters_1, 2, 2, padding='valid'),
                    keras.layers.BatchNormalization(),
                    keras.layers.LeakyReLU(0.2)
                ], name='de_conv_block_' + str(i))
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    keras.Sequential([
                        keras.layers.Conv2DTranspose(n_filters_1,
                                                     ks, ks, padding='valid'),
                        keras.layers.BatchNormalization(),
                        keras.layers.LeakyReLU(0.2)
                    ], name='de_multi_skip_block_' + str(i))
                )
        self.output_conv = keras.layers.Conv2D(output_channels, 3, 1, padding='same')

    def call(self, z):
        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)
        return z

    def build_graph(self):
        x = keras.layers.Input(shape=(16, 16, 256))
        return keras.models.Model(inputs=[x], outputs=self.call(x))


class ResNetAE(keras.Model):
    def __init__(self,
                input_shape=(256, 256, 3),
                n_ResidualBlock=8,
                n_levels=4,
                z_dim=128,
                bottleneck_dim=128,
                bUseMultiResSkips=True,
                drop_out=None):

        super(ResNetAE, self).__init__()

        image_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock,
                                     n_levels=n_levels, input_ch=image_channels,
                                     z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips, drop_out=drop_out)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock,
                                     n_levels=n_levels, output_channels=image_channels,
                                     z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)

        self.fc1 = keras.layers.Dense(bottleneck_dim)
        self.fc2 = keras.layers.Dense(self.z_dim * self.img_latent_dim * self.img_latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        h = keras.layers.Flatten()(h)
        # h = tf.reshape(h, shape=(-1, self.z_dim * self.img_latent_dim, self.img_latent_dim))
        return self.fc1(h)

    def decode(self, z):
        h = self.fc2(z)
        h = keras.layers.Reshape((self.img_latent_dim, self.img_latent_dim, self.z_dim))(h)
        # h = tf.reshape(h, shape=(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim))
        h = self.decoder(h)
        return keras.layers.Activation('sigmoid', name='sigmoid')(h)

    def call(self, x):
        return self.decode(self.encode(x))

    def build_graph(self):
        x = keras.layers.Input(shape=(256, 256, 3))
        return keras.models.Model(inputs=[x], outputs=self.call(x))
