from utils import style
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # change to 2


os.system('clear')
print(style.YELLOW + f'Tensorflow  version: {tf.__version__}\n')
print(style.GREEN, end='')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.0

model = keras.Sequential(
    [
        Input((28,28,1)),
        # Conv2D(64, 3, padding = 'same'), 
        # ReLU(), 
        Conv2D(128, 3, padding = 'same'), 
        ReLU(), 
        Flatten(), 
        Dense(10), 
    ], 
    name = 'model'
)

class CustomFit(keras.Model): 
    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model
    
    def compile(self, optimizer, loss):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data): 
        x, y = data
        
        with tf.GradientTape() as tape: 
            # forward prop
            # tape records steps, so that it could be used in back prop
            y_pred = self.model(x, training=True)
            loss = self.loss(y, y_pred) # from model.compile

        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(gradients, training_vars))
        acc_metric.update_state(y, y_pred)
        # self.compiled_metrics.update_state(y, y_pred) # for the metrics (accuracy)

        return {'loss': loss, 'accuracy': acc_metric.result()}
    
    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        loss = self.loss(y, y_pred)
        acc_metric.update_state(y, y_pred)

        return {'loss': loss, 'accuracy': acc_metric.result()}


print(style.GREEN, end='')

acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')  

training = CustomFit(model)
training.compile(
    optimizer = keras.optimizers.Adam(learning_rate=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    # metrics=['accuracy'],
)
training.fit(x_train, y_train, batch_size=32, epochs = 1)

training.evaluate(x_test, y_test, batch_size=32)