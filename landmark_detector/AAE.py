from archs import Discriminator, D_net_gauss
import resnet_encoder
import inversenet
import archs
import tensorflow as tf
from tensorflow import keras


class AAE(keras.Model):
    def __init__(self, input_size, output_size=None, z_dim=99):
        super(AAE, self).__init__()

        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.z_dim = z_dim
        input_channels = 3

        self.Q = resnet_encoder.resnet18(num_classes=self.z_dim,
                                         input_size=input_size,
                                         input_channels=input_channels,
                                         layer_normalization='batch')

        decoder_class = inversenet.InvResNet
        num_blocks = [1] * 4
        self.P = decoder_class(inversenet.InvBasicBlock,
                               num_blocks,
                               input_dims=self.z_dim,
                               output_size=output_size,
                               output_channels=input_channels,
                               layer_normalization='batch',
                               spectral_norm=False)
        self.D_z = D_net_gauss(self.z_dim)
        self.D = Discriminator()

        self.total_iter = 0
        self.iter = 0
        self.z =None
        self.images = None
        self.current_dataset = None

    def call(self, X):
        self.z = self.Q(X)
        outputs = self.P(self.z)
        self.landmark_heatmaps = None
        if outputs.shape[1] > 3:
            self.landmark_heatmaps = outputs[:, 3:]
        return outputs[:, :3]


aae = AAE(256)