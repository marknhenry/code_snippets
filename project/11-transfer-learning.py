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

# Pretrained Model from '04-basic-cnn-w-reg.py'
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.models.load_model('models/pretrained-cnn/')
# print(model.summary())

#add a custom layer of 100 nodes at the end!
base_inputs = model.layers[0].input
base_outputs = model.layers[-3].output
x = layers.Dense(128, name='dense_128')(base_outputs)
x = layers.Dropout(0.5)(x)
final_outputs = layers.Dense(100, name='dense_10')(x)

new_model = keras.Model(inputs=base_inputs, outputs = final_outputs)
# print(new_model.summary())
print(style.GREEN, end='')
new_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer = keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

# # freeze all layers except last one: 
# for layer in model.layers[:-1]: 
#     layer.trainable=False

new_model.fit(x_train, y_train, batch_size=64, epochs=1, )
model.save('models/pretrained-cnn/')