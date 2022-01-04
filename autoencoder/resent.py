import tensorflow as tf
from tensorflow import keras


zsize = 48

def conv3x3(in_planes, out_planes, stride=1):
    return keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride,
                               padding='same', use_bias=False)


class BasicBlock(keras.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = keras.layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(keras.Model):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = keras.layers.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(planes, kernel_size=3, strides=stride,
                                         padding='same', use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(planes * 4, kernel_size=1, use_bias=False)
        self.bn3 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out =self.relu(out)

        return out


class Encoder(keras.Model):
    def __init__(self, block, layers, num_classes=23):
        super(Encoder, self).__init__()
        self.inplanes = 64
        self.conv1 = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same',
                                         use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.maxpool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = keras.layers.AvgPool2D(pool_size=7, strides=1)
        self.fc = keras.layers.Dense(1000)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                keras.layers.Conv2D(planes * block.expansion,
                                    kernel_size=1, strides=stride, use_bias=False),
                keras.layers.BatchNormalization()
            ]
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return keras.Sequential(layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = keras.layers.Flatten()(x)
        x = self.fc(x)

        return x


# class Binary(Function)

class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dfc3 = keras.layers.Dense(4096)
        self.bn3 = keras.layers.BatchNormalization()
        self.dfc2 = keras.layers.Dense(4096)
        self.bn2 = keras.layers.BatchNormalization()
        self.dfc1 = keras.layers.Dense(256 * 6 * 6)
        self.bn1 = keras.layers.BatchNormalization()
        self.upsample1 = keras.layers.UpSampling2D()
        self.dconv5 = keras.layers.Conv2DTranspose(256, 3, padding='valid')
        self.dconv4 = keras.layers.Conv2DTranspose(384, 3, padding='same')
        self.dconv3 = keras.layers.Conv2DTranspose(192, 3, padding='same')
        self.dconv2 = keras.layers.Conv2DTranspose(64, 5, padding='same')
        self.dconv1 = keras.layers.Conv2DTranspose(3, 12, strides=4, padding='same')
        self.relu = keras.layers.Activation('relu')
        self.sigmoid = keras.layers.Activation('sigmoid')

    def call(self, x):
        x = self.dfc3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.dfc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.dfc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = tf.reshape(x, shape=(-1, 6, 6, 256))
        x = self.upsample1(x)

        x = self.dconv5(x)
        x = self.relu(x)

        x = self.dconv4(x)
        x = self.relu(x)

        x = self.dconv3(x)
        x = self.relu(x)

        x = self.upsample1(x)

        x = self.dconv2(x)
        x = self.relu(x)

        x = self.upsample1(x)

        x = self.dconv1(x)
        x = self.sigmoid(x)

        return x


class Autoencoder(keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(Bottleneck, [3, 4, 6, 3])
        self.decoder = Decoder()

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    