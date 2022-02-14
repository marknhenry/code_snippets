from utils import style
import tensorflow_datasets as tfds
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from layers.experimental.preprocessing import Resizing, RandomFlip, RandomContrast
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2


os.system('clear')

print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
print(style.GREEN)

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True, 
    as_supervised=True, 
    with_info=True, 
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label

# HYPERPARAMETERS
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE=32

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

data_augmentation = keras.Sequential(
    [
        Resizing(height=32, width=32), 
        RandomFlip(mode='horizontal'), 
        RandomContrast(factor=0.1), 
    ]
)

# Building the Model
model=keras.Sequential(
    [
        Input((32,32,3)),
        data_augmentation, 
        Conv2D(4,3,padding='same', activation='relu'), 
        Conv2D(8,3,padding='same', activation='relu'), 
        MaxPooling2D(), 
        Conv2D(16,3,padding='same', activation='relu'), 
        Flatten(), 
        Dense(64, activation='relu'), 
        Dense(10), 
    ]
)

print(style.GREEN, end='')
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer = keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)
  
model = keras.models.load_model('models/cifar10-data-aug-2')
model.fit(ds_train, epochs = 1)
model.save('models/cifar10-data-aug-2/')
model.evaluate(ds_test)