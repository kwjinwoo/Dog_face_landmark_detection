from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Conv2DTranspose, BatchNormalization, Flatten, concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


class MobileNetV2_UNet:
    def __init__(self, input_shape=(256, 256, 3), encoder_output_size=None):
        super().__init__()
        self.model = None
        self.input_shape = input_shape
        self.encoder_output_size = encoder_output_size

        if encoder_output_size is not None:
            self.encoder_last_layer_name = "latent-space-rep"
        else:
            self.encoder_last_layer_name = "block_16_project_BN"
        self.encoder = None

    def build_model(self, nodes):
        mobileNet = MobileNetV2(input_shape=self.input_shape, include_top=False)
        inputs = mobileNet.inputs

        c5 = [layer for layer in mobileNet.layers if layer.name == 'block_16_project_BN'][0].output
        c4 = [layer for layer in mobileNet.layers if layer.name == 'block_12_add'][0].output
        c3 = [layer for layer in mobileNet.layers if layer.name == 'block_5_add'][0].output
        c2 = [layer for layer in mobileNet.layers if layer.name == 'block_2_add'][0].output
        c1 = [layer for layer in mobileNet.layers if layer.name == 'expanded_conv_project_BN'][0].output

        # c5 = [layer for layer in mobileNet.layers if layer.name == 'block_16_project'][0].output
        # c4 = [layer for layer in mobileNet.layers if layer.name == 'block_12_project'][0].output
        # c3 = [layer for layer in mobileNet.layers if layer.name == 'block_5_project'][0].output
        # c2 = [layer for layer in mobileNet.layers if layer.name == 'block_2_project'][0].output
        # c1 = [layer for layer in mobileNet.layers if layer.name == 'Conv1'][0].output

        bottleneck = Conv2D(1280, 1, use_bias=False, padding='same')(c5)
        c5 = BatchNormalization(name='encoder_last_layer')(bottleneck)
        if self.encoder_output_size is not None:
            volumeSize = K.int_shape(c5)
            x = Flatten()(c5)
            x = Dense(self.encoder_output_size, name="latent-space-rep")(x)
            x = Dense(np.prod(volumeSize[1:]))(x)
            c5 = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        c5 = Conv2DTranspose(320, 1, use_bias=False, padding='same')(c5)
        c5 = BatchNormalization()(c5)

        # DECODER Unet
        u6 = Conv2DTranspose(nodes * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(nodes * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(nodes * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(nodes * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(nodes * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(nodes * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(nodes * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(nodes * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(nodes * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)

        u10 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same', name='last_block_transponse')(c9)
        #         u10 = concatenate([u9, c1], axis=3)
        c10 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u10)
        c10 = BatchNormalization()(c10)
        c10 = Dropout(0.1)(c10)
        c10 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c10)
        c10 = BatchNormalization()(c10)

        outputs = Conv2D(3, (1, 1), activation='sigmoid')(c10)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        return self.model

