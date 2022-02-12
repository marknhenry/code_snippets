import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # change to 2
os.system('clear')

print(f'Tensorflow version: {tf.__version__}\n')

## Initialization of Tensors

x = tf.constant(4)
x = tf.constant(4.)
x = tf.constant(4, shape=(1,1))
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x = tf.constant([[1,2,3],[4,5,6]])
x = tf.ones((3,3))
x = tf.zeros((2,3))
x = tf.eye(3)
x = tf.random.normal((3,3), mean=0, stddev=1)
x = tf.random.uniform((3,3), minval=0, maxval=1)
x = tf.range(9)
x = tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float64)

## Mathematical Operations

x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

# Element-wise
z = tf.add(x,y) # same as z = x+y 
z = tf.subtract(x,y) # same as z = x-y
z = tf.divide(x,y) # same as z = x/y
z = tf.multiply(x,y) # same as z = x*y
z = x**5

# Dot Product
z = tf.tensordot(x, y, axes = 1) # same as z = tf.reduce_sum(x*y, axis=0)

x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = tf.matmul(x, y) # same as z = x @ y

## Indexing

x = tf.constant([0,1,1,2,3,1,2,3])
indices = tf.constant([0,3])
x_ind = tf.gather(x, indices)
x = tf.constant([[1,2],[3,4],[5,6]])

## Reshaping
x = tf.range(9)
x = tf.reshape(x, (3,3))
y = tf.transpose(x)
y = tf.transpose(x, perm=[1,0])
print('\n')

