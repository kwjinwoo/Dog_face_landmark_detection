import tensorflow as tf
from tensorflow import keras
import math


def conv3x3(filters, stride=1):
    return keras.layers.Conv2d(filters, kernel_size=3, padding=1, strides=stride, use_bias=False)


def conv1x1(filters, stride=1):
    return keras.layers.Conv1D(filters, kernel_size=1, padding=0, strides=stride, use_bias=False)


class BasicBlock(keras.Model):
    expansion = 1

    def get_config(self):
        pass

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_normalization='batch'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(planes, stride)
        self.relu = keras.layers.ReLU()
        self.conv2 = conv3x3(planes)
        self.layer_norm = layer_normalization
        self.bn1 = keras.layers.BatchNormalization()
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


class LandmarkHead(keras.Model):
    def get_config(self):
        pass

    def __init__(self, block, layers, num_classes=1000, input_size=256, input_channels=3, layer_normalization='batch'):
        super(LandmarkHead, self).__init__()
        self.input_size = input_size
        self.layer_norm = layer_normalization
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding=1)
        self.x2 = None
        self.inplanes = 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = keras.layers.GlobalAvgPool2D()   # study 필요
        self.fc = keras.layers.Dense(512 * block.expansion, num_classes)

        for m in self.layers:
            if isinstance(m, keras.layers.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.output_shape[-1]    # study 필요
                m.set_weights()   # study 필요
            elif isinstance(m, keras.layers.BatchNormalization):
                m.set_weights()   # study 필요
                m.bias.data.zero_()   # study 필요

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential(
                keras.layers.Conv2D(planes * block.expansion,
                                    kernel_size=1, strides=stride, use_bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, layer_normalization=self.layer_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_normalization=self.layer_norm))

        return keras.Sequential(*layers)

    def call(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x


class Bottleneck(keras.Model):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_normalization='batch'):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(planes)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(planes)
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = conv1x1(planes * 4)
        self.bn3 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
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
        out = self.relu(out)

        return out


class ResNet(keras.Model):
    def __init__(self, block, layers, num_classes=1000, input_size=256, input_channels=3, layer_normalization='batch'):
        super(ResNet, self).__init__()
        self.input_size = input_size
        self.inplanes = 64
        self.conv1 = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False)
        self.layer_norm = layer_normalization
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding=1)
        self.x2 = None

        self.with_additional_layers = True

        if input_size == 256 and self.with_additional_layers:
            self.layer0 = self._make_layer(block, 64, layers[0])
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        elif input_size == 512 and self.with_additional_layers:
            self.layer0 = self._make_layer(block, 64, layers[0])
            self.layer01 = self._make_layer(block, 64, layers[0], stride=2)
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = keras.layers.GlobalAvgPool2D()   # study 필요
        self.fc = keras.layers.Dense(512 * block.expansion, num_classes)

        # study 필요
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                keras.layers.Conv2D(planes * block.expansion,
                                    kernel_size=1, strides=stride, use_bias=False),
                keras.layers.BatchNormalization()
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, layer_normalization=self.layer_norm))

        return keras.Sequential(*layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.input_size > 128 and self.with_additional_layers:
            x = self.layer0(x)
            if self.input_size > 256:
                x = self.layer01(x)

        self.x1 = self.layer1(x)
        self.x2 = self.layer2(self.x1)
        x = self.layer3(self.x2)
        x = self.layer4(x)
        self.ft = x
        x = self.avgpool(x)
        x = self.fc(x)

        return x


class DiscriminatorHead(keras.Model):
    def __init__(self, **params):
        super(DiscriminatorHead, self).__init__()
        conv = conv3x3
        self.t1 = conv(64)
        self.t2 = conv(128)
        self.t3 = conv(256)
        self.t4 = conv(512)
        self.lin = keras.layers.Dense(512 * 1, 1)
        self.avgpool = keras.layers.GlobalAvgPool2D()
        self.sigmoid = keras.layers.Activation('sigmoid')

    # study 필요
    def call(self, Q, qx1):
        x = qx1

        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.avgpool(x)

        return self.sigmoid(self.lin(x))


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
