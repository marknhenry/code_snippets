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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    # layers.SimpleRNN(256, return_sequences=True, activation='tanh')
    # layers.GRU(256, return_sequences=True, activation='tanh')
    # layers.LSTM(256, return_sequences=True, activation='tanh')
    layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, activation='tanh')   
    )
)
model.add(
    layers.Bidirectional(
        layers.LSTM(32, activation='tanh')   
    )
)
model.add(layers.Dense(10))

print(style.BLUE)
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=keras.optimizers.Adam(lr=0.001), 
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=1,)
print(style.MAGENTA, style.UNDERLINE)
model.evaluate(x_test, y_test, batch_size=64)
print(style.RESET)
