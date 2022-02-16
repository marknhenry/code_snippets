import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2
import io
import tensorflow as tf
from utils import style, image_grid, plot_to_image
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

    # Matplotlib wants [0,1] values
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
    
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
            Conv2D(8, 3, padding='same', activation='relu'),
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(10),
        ]
    )

    return model

print(style.GREEN, end='')

model = get_model()

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

num_epochs = 1

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
writer = tf.summary.create_file_writer('logs/img-logs/')
step = 0

for epoch in range(num_epochs):
    # print(f'\nStart of Training Epoch {epoch}')
    for batch_idx, (x, y) in enumerate(ds_train):
        figure = image_grid(x, y, class_names)
        with writer.as_default():
            tf.summary.image(
                'Vizualize Images', plot_to_image(figure), step=step
            )
            step += 1
    #     with tf.GradientTape() as tape:
    #         y_pred = model(x, training=True)
    #         loss = loss_fn(y, y_pred)

    #     gradients = tape.gradient(loss, model.trainable_weights)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    #     acc_metric.update_state(y, y_pred)

    #     with train_writer.as_default(): 
    #         tf.summary.scalar('Loss', loss, step=train_step)
    #         tf.summary.scalar('Accuracy', acc_metric.result(), step=train_step)
    #         train_step += 1

    # acc_metric.reset_states()

    # for batch_idx, (x, y) in enumerate(ds_test):
    #     y_pred = model(x, training=False)
    #     loss = loss_fn(y, y_pred)
    #     acc_metric.update_state(y, y_pred)

    #     with test_writer.as_default(): 
    #         tf.summary.scalar('Loss', loss, step=test_step)
    #         tf.summary.scalar('Accuracy', acc_metric.result(), step=test_step)
    #         test_step += 1
    
    # acc_metric.reset_states()