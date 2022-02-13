
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


# HYPERPARAMETERS
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001

# os.system('conda install -y pandas')
import pandas as pd
os.system('clear')

print(style.RESET + f'Tensorflow  version: {tf.__version__}\n')
# print(f'Tensorflow  version: {tf.__version__}\n')
print(os.getcwd())
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
train_images = os.getcwd() + '/data/train_images/' + train_df.iloc[:, 0].values
test_images = os.getcwd() + '/data/test_images/' + test_df.iloc[:, 0].values

train_labels = train_df.iloc[:, 1:].values
test_labels = test_df.iloc[:, 1:].values


def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    image.set_shape((64, 64, 1))
    label[0].set_shape([])
    label[1].set_shape([])

    labels = {'first_number': label[0], 'second_number': label[1]}
    return image, labels


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)
)
train_dataset = (
    train_dataset.shuffle(buffer_size=len(train_labels))
    .map(read_image)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)
)
test_dataset = (
    test_dataset.shuffle(buffer_size=len(test_labels))
    .map(read_image)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

inputs = keras.Input(shape=(64, 64, 1))
x = layers.Conv2D(32, 3,
                  padding='same',
                  kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                  )(inputs)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(64, 3, 
                  kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
)(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3,
                  activation='relu',
                  kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                  )(x)
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
output1 = layers.Dense(10, name='first_number')(x)
output2 = layers.Dense(10, name='second_number')(x)

model = keras.Model(inputs=inputs, outputs = [output1, output2])

model.compile(
    optimizer = keras.optimizers.Adam(LEARNING_RATE),
    loss = [
        keras.losses.SparseCategoricalCrossentropy(),
        keras.losses.SparseCategoricalCrossentropy(),
    ], 
    metrics = ['accuracy']
)

model.fit(train_dataset, epochs=5)
model.evaluate(test_dataset)
