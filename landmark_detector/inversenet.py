import tensorflow as tf
from tensorflow import keras
from resnet_encoder import conv3x3


def deconv4x4(filters, stride=1):
    return keras.layers.Conv2DTranspose(filters, kernel_size=4, strides=stride, padding=1, use_bias=False)


class InvBasicBloc(keras.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, layer_normalization='batch',
                 with_spectral_norm=False):
        super(InvBasicBloc, self).__init__()
        self.layer_normalization = layer_normalization
        if upsample is not None:
            self.conv1 = deconv4x4(planes, stride)
        else:
            self.conv1 = conv3x3(planes, stride)
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.conv2 = conv3x3(planes)
        self.upsample = upsample
        self.stride = stride

        # study 필요
        # if with_spectral_norm:

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        return self.relu(out)

