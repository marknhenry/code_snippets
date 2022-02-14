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

model = keras.applications.InceptionV3(include_top=True)
print(model.summary())


base_input = model.layers[0].input
base_output = model.layers[-2].output
final_output = layers.Dense(5) (base_output)

new_model = keras.Model(inputs = base_input, outputs = final_output)
print(style.GREEN, end='')
new_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer = keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

new_model.fit(x, y, epochs = 1)