from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Dropout
from tensorflow.keras import Input
from utils import style
os.system('clear')
print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
print(style.GREEN, end='')

(ds_train, ds_test), ds_info = tfds.load(
    'mnist', 
    split=['train', 'test'], 
    shuffle_files = True, 
    as_supervised = True, 
    with_info = True,
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# Setting up training dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setting up test dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)

model = Sequential([
    Flatten(input_shape=(28,28,1)), 
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer = Adam(0.001), 
    metrics=['accuracy']
)

dt = datetime.now().strftime('%Y%m%d-%H%M%S')
logs = f'logs/{dt}'

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq=1, profile_batch='500,520')

model.fit(ds_train, epochs=2, validation_data = ds_test, callbacks=[tboard_callback])
