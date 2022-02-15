from utils import style
from tensorflow.keras.datasets import cifar10
import tensorflow_datasets as tfds
from tensorflow.keras import layers, regularizers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2


os.system('clear')

print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
print(style.GREEN)

# print(tfds.list_builders())

# ds = tfds.load('celeb_a_hq', split='train', shuffle_files=True)
# assert isinstance(ds, tf.data.Dataset)
# print(ds)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    # shuffle_files=True,
    # as_supervised=False,
    with_info=True,
)

# fig = tfds.show_examples(ds_train, ds_info, rows=4, cols=4)
print(ds_info)


def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

ds_train = ds_train.map(normalize_img,
                        num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(1000)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds