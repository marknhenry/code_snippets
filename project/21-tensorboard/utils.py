import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
import sklearn.metrics
from tensorboard.plugins import projector
import os
import shutil



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


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(data, labels, class_names):
    # Data should be in (BATCH_SIZE, H, W, C)
    assert data.ndim == 4
    
    figure = plt.figure(figsize=(10,10))
    num_images = data.shape[0]
    size = int(np.ceil(np.sqrt(num_images)))

    for i in range(data.shape[0]):
        plt.subplot(size, size, i + 1, title=class_names[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # if grayscale
        if data.shape[3] == 1: 
            plt.imshow(data[i], cmap = plt.cm.binary)
        
        else: 
            plt.imshow(data[i])
        
    return figure