import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

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

x_train = x_train.reshape(-1,28,28,1).astype("float32") /255.0
x_test = x_test.reshape(-1,28,28,1).astype("float32") /255.0

# need: CNN -> BatchNorm -> ReLU
# need 10 times this much, this is a lot of code!
# Model subclassing to the rescue

class CNNBlock(layers.Layer): 
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False): 
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

model = keras.Sequential(
    [
        CNNBlock(32), 
        CNNBlock(64), 
        CNNBlock(128), 
        layers.Flatten(), 
        layers.Dense(10),
    ]
)
# model.build(input_shape=(1000,28,28,1))
# print(style.BLUE + model.summary())
print(style.GREEN)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer = keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=1, )
model.evaluate(x_test, y_test, batch_size=64, )

