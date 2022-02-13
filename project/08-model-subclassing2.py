from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, regularizers
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


os.system('clear')

print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
# print(style.GREEN)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# need: CNN -> BatchNorm -> ReLU
# need 10 times this much, this is a lot of code!
# Model subclassing to the rescue


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same")
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

# model = keras.Sequential(
#     [CNNBlock(32), CNNBlock(64), CNNBlock(128), layers.Flatten(), layers.Dense(10),]
# )

class ResBlock(layers.Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.cnn1 = CNNBlock(channels[0], 3)
        self.cnn2 = CNNBlock(channels[1], 3)
        self.cnn3 = CNNBlock(channels[2], 3)
        self.pooling = layers.MaxPooling2D()
        self.identity_mapping = layers.Conv2D(channels[1], 3, padding="same")

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x + self.identity_mapping(input_tensor), training=training,)
        x = self.pooling(x)
        return x


class ResNet_Like(keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet_Like, self).__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x, training=training)
        x = self.classifier(x)
        return x

# print(style.GREEN)

model = ResNet_Like(num_classes=10)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=1, )
print(model.summary())
model.evaluate(x_test, y_test, batch_size=64, )
