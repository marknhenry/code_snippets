import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU

from utils import style
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2

os.system('clear')
print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
print(style.GREEN, end='')

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# Setting up training dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setting up test dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)

# Building the Model
model = keras.Sequential(
    [
        Input((28, 28, 1)),
        Conv2D(32, 3, activation='relu'),
        Flatten(),
        Dense(10, activation='softmax'),
    ]
)

print(style.GREEN, end='')

num_epochs = 5

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=3e-4)
acc_metric = keras.metrics.SparseCategoricalAccuracy()

for epoch in range(num_epochs):
    print(f'\nStart of Training Epoch {epoch}')
    for batch_idx, (x_batch, y_batch) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y_batch, y_pred)

    train_acc = acc_metric.result()
    print(f'Accuracy over epoch {train_acc}')
    acc_metric.reset_states()

for batch_idx, (x_batch, y_batch) in enumerate(ds_test):
    y_pred = model(x_batch, training=False)
    acc_metric.update_state(y_batch, y_pred)

train_acc = acc_metric.result()
print(f'Accuracy over test set: {train_acc}')
acc_metric.reset_states()
