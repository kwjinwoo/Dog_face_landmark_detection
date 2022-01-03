import loss
import AAE
import tensorflow as tf
from tensorflow import keras
import model_utils


aae = AAE.AAE(256)
_ = model_utils.builder(aae, 256)
print(aae.Q.trainable_variables)
