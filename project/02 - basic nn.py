import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2
os.system('clear')

print(style.YELLOW + f'Tensorflow version: {tf.__version__}\n')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(style.BLUE +
      f'Old Train Data Shapes: train: {x_train.shape} and {y_train.shape}')

print(f'Old Test Data Shapes: train: {x_test.shape} and {x_test.shape}')

flattened_dim = x_train.shape[1]*x_train.shape[2]

# To flatten:
# -1 means keep as is, or you calculate it
x_train = x_train.reshape(-1, flattened_dim)
x_train = x_train.astype("float32")  # to minimize the computation
x_train = x_train / 255.0  # to normalize

# -1 means keep as is, or you calculate it
x_test = x_test.reshape(-1, flattened_dim)
x_test = x_test.astype("float32")  # to minimize the computation
x_test = x_test / 255.0  # to normalize

print(f'New Train Data Shapes: train: {x_train.shape} and {y_train.shape}')
print(f'New Test Data Shapes: train: {x_test.shape} and {x_test.shape}')

# print(style.GREEN + '\nCreate model using Sequential API')
# model = keras.Sequential(
#     [
#         keras.Input(shape=(flattened_dim)),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(10),
#     ]
# )

# # print(model.summary())

# # # To debug layer by layer
# # model = keras.Sequential()
# # mode.add(keras.Input(shape=(flattened_dim)))
# # mode.add(layers.Dense(512, activation='relu'))
# # print(model.summar())
# # mode.add(layers.Dense(256, activation='relu'))
# # mode.add(layers.Dense(10))


# # Compile, i.e. how to configure the training part of the n/w, or n/w config
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     # Sparse means that the y is a value, like 3
#     # if you remove sparse, then you need 1 hot encoding 0010000000 for 3
#     optimizer=keras.optimizers.Adam(lr=0.001),
#     metrics=['accuracy'],
# )

# model.fit(x_train, y_train, batch_size=32, epochs=1,)

# # print(model.summary())

# print('Evaluation on Test set')
# model.evaluate(x_test, y_test, batch_size=32, verbose=2)

print(style.GREEN + '\nCreating model using functional API')

# Functional API
inputs = keras.Input(shape=(flattened_dim))
x = layers.Dense(512, activation='relu', name='optionally_layer_1')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs) # to build the whole n/w


# # If you want to debug and get outputs of specific models, then you could: 
# model = keras.Model(inputs=model.inputs, 
#                     # outputs=[model.layers[-1].output] # Option 1: Output of last layer
#                     # outputs=[model.layers[-2].output] # Option 2: Output of 2 layers back from end
#                     # outputs=[model.get_layer('optionally_layer_1').output] # Option 3: Get layer by name
#                     outputs=[layer.output for layer in model.layers] # Option 4: Output of 2 layers back from end
#                     ) 

# # for options 1-3 above
# feature = model.predict(x_train)
# print(f'Shape of intermediate layer: {feature.shape}')

# # for option 4
# features = model.predict(x_train)
# for feature in features: 
#     print(f'Shape of intermediate layer: {feature.shape}')  


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=False),  # since we are using softmax for last layer
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=1, )

print('Evaluation on Test set')
model.evaluate(x_test, y_test, batch_size=32, )

print(style.YELLOW + '\nEND OF FILE\n')
print(style.RESET)
