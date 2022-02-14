from utils import style
import tensorflow_datasets as tfds
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import Resizing, RandomFlip, RandomContrast
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2


os.system('clear')

print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
print(style.GREEN)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True, 
    as_supervised=True, 
    with_info=True, 
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label

# HYPERPARAMETERS
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE=128

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

model = keras.Sequential(
    [
        Input((28,28,1)),
        Conv2D(32, 3, activation = 'relu'), 
        Flatten(), 
        Dense(10), 
    ]
)

save_callback = ModelCheckpoint(
    'models/checkpoint/', 
    save_weights_only=False, 
    monitor='accuracy', 
    save_best_only = False, # save all of them
)

# Scheduler Decay

def scheduler(epoch, lr):
    if epoch < 2: 
        return lr
    else: 
        return lr * 0.99

lr_scheduler = LearningRateScheduler(scheduler, verbose = 1)

class CustomCallback(keras.callbacks.Callback):
    def on_batch_end(self, epoch, logs=None): 
        # print(logs.keys(), logs.values())
        if logs.get('accuracy') > 0.97:
            print('\nAccuracy over 97%. Quitting Training!')
            self.model.stop_training = True

print(style.GREEN, end='')
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)
  
model = keras.models.load_model('models/checkpoint/')
model.fit(ds_train, epochs = 1, 
    callbacks=[save_callback, lr_scheduler, CustomCallback()])

model.evaluate(ds_test)