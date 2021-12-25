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


class LandmarkHead(keras.Model):
    def __init__(self, block, layers, output_size=128, output_channels=68, layer_normalization='none', start_layer=2):
        super(LandmarkHead, self).__init__()
        self.layer_normalization =layer_normalization
        self.lin_landmarks = None
        self.output_size= output_size
        self.output_channels = output_channels
        self.start_layer = start_layer
        self.lin = keras.layers.Conv2DTranspose(output_channels, kernel_size=4, strides=2, padding=1, use_bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = keras.Sequential([
                keras.layers.Conv2DTranspose(planes * block.expansion,
                                             kernel_size=4, strides=stride, padding=1, use_bias=False),
                keras.layers.BatchNormalization()
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, layer_normalization=self.layer_normalization))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return keras.Sequential(*layers)


class LandmarkHeadV2(LandmarkHead):
    def __init__(self, block, layers, **params):
        super(LandmarkHeadV2, self).__init__(block, layers, **params)
        conv = conv3x3
        self.t1 = conv(256)
        self.t2 = conv(128)
        self.t3 = conv(64)
        self.t4 = conv(64)

    def call(self, P):
        x = P.x1
        x = self.t1(x)

        x = P.layer2(x)
        x = self.t2(x)

        x = P.layer3(x)
        x = self.t3(x)

        x = P.layer4(x)
        x = self.t4(x)

        return self.lin(x)


class SemanticFeatureHead(LandmarkHead):
    def __init__(self, block, layers, **params):
        super(SemanticFeatureHead, self).__init__(block, layers, **params)
        conv = conv3x3
        self.t1 = conv(256)
        self.t2 = conv(128)
        self.t3 = conv(64)
        self.t4 = conv(64)
        self.t5 = conv(64)

    def call(self, P):
        inputs = {
            'x1': P.x1,
            'x2': P.x2,
            'x3': P.x3
        }

        gen_layers = {
            'l1': P.layer1,
            'l2': P.layer2,
            'l3': P.layer3,
            'l4': P.layer4
        }

        x = inputs['x1']
        x = self.t1(x)

        x = gen_layers['l2'](x)
        x = self.t2(x)

        x = gen_layers['l3'](x)
        x = self.t3(x)

        x = gen_layers['l4'](x)
        x = self.t4

        x = self.lin(x)

        # study 필요
        # return torch.nn.functional.normalize(x, dim=1)
        return x


class InvResNet(keras.Model):
    def __init__(self, block, layers, output_size=256, output_channels=3, input_dims=99,
                 layer_normalization='none', spectral_norm=False):
        super(InvResNet, self).__init__()
        self.layer_normalization = layer_normalization
        self.with_spectral_norm = spectral_norm

        self.sn = lambda x: x   # study 필요

        self.lin_landmark = None
        self.inplanes = 512
        self.output_size = output_size
        self.output_channels = output_channels
        self.fc = keras.layers.Dense(input_dims, 512)
        self.conv1 = self.sn(keras.layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding=0, use_bias=False))
        self.add_in_tensor = None

        if layer_normalization:
            self.b1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        inplanes_after_layer1 = self.inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        inplanes_after_layer2 = self.inplanes
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.tanh = keras.layers.Activation('tanh')
        self.x2 = None
        if self.output_size == 256:
            self.layer5 = self._make_layer(block, 64, layers[3], stride=2)
        elif self.output_size == 512:
            self.layer5 = self._make_layer(block, 64, layers[3], stride=2)
            self.layer6 = self._make_layer(block, 64, layers[3], stride=2)

        self.lin = keras.layers.Conv2DTranspose(output_channels, kernel_size=4, strides=2, padding=1, use_bias=False)

        # study 필요
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def init_finetuning(self, batchsize):
        if self.add_in_tensor is None or self.add_in_tensor[0] != batchsize:
            self._create_finetune_layers(batchsize)
        else:
            self._reset_finetune_layers()

    def _make_layer_down(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                keras.layers.Conv2DTranspose(planes * block.expansion,
                                             kernel_size=1, strides=stride, use_bias=False)
            ])
            if self.layer_normalization:
                downsample.add(keras.layers.BatchNormalization())

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return keras.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = keras.Sequential([
                self.sn(keras.layers.Conv2DTranspose(
                    planes * block.expansion, kernel_size=4, strides=stride,
                    padding=1, use_bias=False
                ))
            ])
            if self.layer_normalization:
                upsample.add(keras.layers.BatchNormalization())

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample,
                            layer_normalization=self.layer_normalization,
                            with_spectral_norm=self.with_spectral_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return keras.Sequential(*layers)

    def call(self, x):
        x = self.fc(x)
        x = tf.reshape(x, (x.shape[0], -1, 1, 1))

        x = self.conv1(x)
        if self.layer_normalization:
            x = self.b1(x)
        x = self.relu(x)
        self.x0 = x

        x1 = self.layer1(x)
        self.x1 = x1
        self.x2 = self.layer2(x1)
        self.x3 = self.layer3(self.x2)
        self.x4 = self.layer4(self.x3)

        if self.output_size == 128:
            x = self.x4
        elif self.output_size == 256:
            x = self.layer5(self.x4)
        elif self.output_size == 512:
            x = self.layer5(self.x4)
            x = self.layer6(x)
        x = self.lin(x)
        x = self.tanh(x)

        return x
    