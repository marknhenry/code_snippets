import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f'Tensorflow version: {tf.__version__}')

# Initialization of Tensors

x = tf.constant(4)
print(x)
x = tf.constant(4.)
print(x)
x = tf.constant(4, shape(1,1))
print(x)
x = tf.constant(4, shape(1,1), dtype=tf.float32)
print(x)

x = tf.constant([[1,2,3],[4,5,6]])
print(x)

x = tf.ones((3,3))
print(x)

x = tf.zeros((2,3))
print(x)

x = tf.eye(3)
print(x)

x = tf.random.normal((3,3), mean=0, stddev=1)
print(x)

x = tf.random.uniform((3,3), minval=0, maxval=1)
print(x)

x = tf.range(9)
print(x)

x = tf.range(start=1, limit=10, delta=2)
print(x)

x = tf.cast(x, dtype=tf.float64)
print(x)


# Mathematical Operations
# Indexing
# Reshaping