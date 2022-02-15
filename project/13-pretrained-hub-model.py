from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, regularizers
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2

from utils import style

os.system('clear')

print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
# print(style.GREEN)

x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'
base_model = hub.KerasLayer(url, input_shape=(299,299,3))
base_model.trainable = False
model = keras.Sequential([
    base_model, 
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(5),
])


print(style.GREEN, end='')
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer = keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

model.fit(x, y, epochs = 1)