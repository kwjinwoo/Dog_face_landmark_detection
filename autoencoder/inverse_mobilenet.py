import tensorflow as tf
from tensorflow import keras


# Decoder block
# conv --> convtranspose --> conv
class DecoderBlock(keras.Model):
    def __init__(self, in_channels, out_channels, stride):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.filters = in_channels // 4
        self.out_channels = out_channels
        self.relu = keras.layers.LeakyReLU(0.2)
        self.stride = stride
        self.conv1 = keras.layers.Conv2D(self.filters, kernel_size=1, padding='same', use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.deconv = keras.layers.Conv2DTranspose(self.filters, kernel_size=4,
                                                   padding='same', strides=self.stride,
                                                   use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(out_channels, kernel_size=1, padding='same', use_bias=False)
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


# Decoder
class InverseMobileNetV2(keras.Model):
    def __init__(self, input_size, z_dim):
        super(InverseMobileNetV2, self).__init__()
        self.z_dim = z_dim
        self.latent_dim = input_size // 2 ** 5
        self.relu = keras.layers.LeakyReLU(0.2)
        self.deconv1 = keras.layers.Conv2D(320, 1, use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.inv_res_block = self._make_blocks()
        self.deconv2 = keras.layers.Conv2DTranspose(3, 3, padding='same', strides=2, use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, x):
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.inv_res_block(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        out = keras.layers.Activation('sigmoid')(out)
        return out

    def _make_blocks(self):
        config = [
            [320, 160, 1],
            [160, 96, 2],
            [96, 64, 1],
            [64, 32, 2],
            [32, 24, 2],
            [24, 16, 2],
            [16, 32, 1]
        ]
        blocks = keras.models.Sequential(
            [DecoderBlock(i[0], i[1], i[2]) for i in config]
        )
        return blocks


# Autoencoder
class MobileNetAE(keras.Model):
    def __init__(self, input_size, z_dim):
        super(MobileNetAE, self).__init__()
        self.input_size = input_size
        self.z_dim = z_dim
        self.latent_size = input_size[0] // 2 ** 5
        self.AE = self._make_AE()

        # self.encoder = self._make_encoder()
        # self.decoder = InverseMobileNetV2(self.input_size[0], self.z_dim)

    def call(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def _make_AE(self):
        encoder = keras.applications.MobileNetV2(input_shape=self.input_size, alpha=1.0, include_top=False, weights=None)

        c5 = [layer for layer in encoder.layers if layer.name == 'block_16_project_BN'][0].output
        c4 = [layer for layer in encoder.layers if layer.name == 'block_12_add'][0].output
        c3 = [layer for layer in encoder.layers if layer.name == 'block_5_add'][0].output
        c2 = [layer for layer in encoder.layers if layer.name == 'block_2_add'][0].output
        c1 = [layer for layer in encoder.layers if layer.name == 'expanded_conv_project_BN'][0].output

        print(c1.shape)
        return encoder

    def _make_decoder_blocks(self):
        config = [
            [320, 160, 1],
            [160, 96, 2],
            [96, 64, 1],
            [64, 32, 2],
            [32, 24, 2],
            [24, 16, 2],
            [16, 32, 1]
        ]
        blocks = keras.models.Sequential(
            [DecoderBlock(i[0], i[1], i[2]) for i in config]
        )
        return blocks

ae = MobileNetAE((256, 256, 3), 99)
# temp = tf.random.normal(shape=(1, 256, 256, 3))
# out = ae(temp)
# print(out.shape)
# print(ae.summary())


