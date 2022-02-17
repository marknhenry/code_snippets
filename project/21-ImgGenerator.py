from utils import style
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, regularizers
from tensorflow import keras
import tensorflow as tf
import os
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2


os.system('clear')
print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
print(style.GREEN, end='')

img_height = 28
img_width = 28
batch_size = 2
DS_LEN = 3005

model = keras.Sequential([
    Input((28, 28, 1)),
    Conv2D(16, 3, padding='same'),
    # Conv2D(32, 3, padding='same'),
    # MaxPooling2D(),
    Flatten(),
    Dense(10),
])


# # Method 1: Dataset from Directory
# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     'data/trainingSet/trainingSet/',
#     labels='inferred',
#     label_mode='int',
#     color_mode='grayscale',
#     batch_size=batch_size,
#     image_size=(img_height, img_width),  # reshape if not this size
#     shuffle=True,
#     seed=1984,
#     validation_split=0.1,
#     subset='training'
# )

# ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
#     'data/trainingSet/trainingSet/',
#     labels='inferred',
#     label_mode='int',
#     color_mode='grayscale',
#     batch_size=batch_size,
#     image_size=(img_height, img_width),  # reshape if not this size
#     shuffle=True,
#     seed=1984,
#     validation_split=0.1,
#     subset='validation'
# )


# def augment(x, y):  # done on each image, in sequence
#     # Add random brightness
#     image = tf.image.random_brightness(x, max_delta=0.1)

#     return image, y


# ds_train = ds_train.map(augment)

# # custom loos
# # for epochs in range(10):
# #     for x, y in ds_train:
# #         pass

# # for built in training
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(learning_rate=3e-4),
#     metrics=['accuracy']
# )

# model.fit(ds_train, epochs=1)

# Method 2 ImageDataGenerator and flow_from_directory

datagen = ImageDataGenerator(
    rescale=1./255,  # scaling + float
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=False,
    vertical_flip=False,
    data_format='channels_last',
    validation_split=0.1,
    dtype=tf.float32,
)

train_generator = datagen.flow_from_directory(
    'data/trainingSet/trainingSet/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    subset='training',
    seed=1984,
)

validation_generator = datagen.flow_from_directory(
    'data/trainingSet/trainingSet/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    subset='validation',
    seed=1984,
)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=3e-4)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

for epoch in range(10):
    num_batches = 0
    print(f'Epoch {epoch+1} in 10.  ', end='')
    for x, y in train_generator:
        num_batches += 1
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y, y_pred)

        train_acc = acc_metric.result()
        
        acc_metric.reset_states()

        if num_batches >= DS_LEN/batch_size:
            break  # otherwise the generator would go for ever.
    
    num_batches = 0
    for x, y in validation_generator: 
        num_batches+=1
        with tf.GradientTape() as tape: 
            y_pred = model(x, training=False)
        val_acc_metric.update_state(y, y_pred)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        if num_batches >= DS_LEN/batch_size/10:
            break  # otherwise the generator would go for ever.

    print(f'train-acc: {train_acc}, valid-acc: {val_acc}')