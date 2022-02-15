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
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# need: CNN -> BatchNorm -> ReLU
# need 10 times this much, this is a lot of code!
# Model subclassing to the rescue


class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
        )

        self.b = self.add_weight(
            name='b', shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MyReLU(layers.Layer):
    def __init__(self):
        super(MyReLU, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)


class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64)  # Keras implementation
        self.dense2 = Dense(num_classes)  # Keras implementation
        # self.dense1 = layers.Dense(64) # Keras implementation
        # self.dense2 = layers.Dense(num_classes) # Keras implementation
        self.relu = MyReLU()

    def call(self, input_tensor):
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)


model = MyModel()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)
print(style.GREEN, end='')
model.fit(x_train, y_train, batch_size=64, epochs=1, )

model.evaluate(x_test, y_test, batch_size=64, )

# 1. Save and load weights only
model.save_weights('save_model/',
                #    save_format='h5'
                   )

# To load:
# model.load_weights('saved_model/')

# 2. Save and load entire model:
# 2.1. Save weights
# 2.2. Model architecture
# 2.3. Training Configuration (model.compile())
# 2.4. Optimizer and states

model.save('complete_saved_model/')
model = keras.models.load_model('complete_saved_model/')
print('loaded model, training for 1 more epoch')
print(style.BLUE, end='')
model.fit(x_train, y_train, batch_size=64, epochs=1, )
model.evaluate(x_test, y_test, batch_size=64, )
