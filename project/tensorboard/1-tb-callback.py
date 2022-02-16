import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2
import io
import tensorflow as tf
# from project.utils import style
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow_datasets as tfds

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
print(style.GREEN, end='')

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label

def augment(image, label):
    
    # convert 10% of images to greyscale
    percent_to_convert = 0.1 #10%
    if tf.random.uniform((), minval=0, maxval=1) < percent_to_convert: 
        image = tf.image.rgb_to_grayscale(image)
        
        # since grayscale has 1 channel, and input expects 3, we can duplicate
        image = tf.tile(image, [1,1,3])

    # Add random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    #  flip left to right
    image = tf.image.random_flip_left_right(image)

    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# Setting up training dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setting up test dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def get_model():
    # Building the Model
    model = keras.Sequential(
        [
            Input((32, 32, 3)),
            # Conv2D(8, 3, padding='same', activation='relu'),
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            # Dense(64, activation='relu'),
            # Dropout(0.1),
            Dense(10),
        ]
    )

    return model

print(style.GREEN, end='')

model = get_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='project/tensorboard/tb_callback_dir', histogram_freq=1,
)

save_callback = keras.callbacks.ModelCheckpoint(
    'models/cifar10-basic/checkpoint/', 
    save_weights_only=False, 
    monitor='accuracy', 
    save_best_only = False, # save all of them
)

class CustomCallback(keras.callbacks.Callback):
    def on_batch_end(self, epoch, logs=None): 
        # print(logs.keys(), logs.values())
        if logs.get('accuracy') > 0.97:
            print('\nAccuracy over 97%. Quitting Training!')
            self.model.stop_training = True

model = keras.models.load_model('models/cifar10-basic/checkpoint/')

model.fit(
    ds_train, 
    epochs=500,
    validation_data=ds_test,
    callbacks=[tensorboard_callback, save_callback, CustomCallback()], 
)

# num_epochs = 1

# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = keras.optimizers.Adam(learning_rate=1e-4)
# acc_metric = keras.metrics.SparseCategoricalAccuracy()
# train_writter = tf.summary.create_file_writer('logs/train/')
# test_writter = tf.summary.create_file_writer('logs/test/')
# train_step = test_step = 0

# for epoch in range(num_epochs):
#     print(f'\nStart of Training Epoch {epoch}')
#     for batch_idx, (x, y) in enumerate(ds_train):
#         with tf.GradientTape() as tape:
#             y_pred = model(x, training=True)
#             loss = loss_fn(y, y_pred)

#         gradients = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#         acc_metric.update_state(y, y_pred)

#     # train_acc = acc_metric.result()
#     # print(f'Accuracy over epoch {train_acc}')
#     acc_metric.reset_states()

# for batch_idx, (x, y) in enumerate(ds_test):
#     y_pred = model(x, training=False)
#     loss = loss_fn(y, y_pred)
#     acc_metric.update_state(y, y_pred)

# # train_acc = acc_metric.result()
# # print(f'Accuracy over test set: {train_acc}')
# acc_metric.reset_states()
