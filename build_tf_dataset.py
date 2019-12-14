"""Based on this tutorial: https://www.tensorflow.org/tutorials/load_data/images

Returns:
    [type] -- [description]
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from loguru import logger

from config import DATA_DIR, TEST_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, AUTOTUNE


logger.info("Starting up...")

image_count = len(list(DATA_DIR.glob("train/*/*.jpg")))
logger.debug(image_count)

STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)


CLASS_NAMES = np.array(
    [
        item.name
        for item in DATA_DIR.glob("*")
        if item.name != "LICENSE.txt" and item.name != ".DS_STORE"
    ]
)
logger.debug(CLASS_NAMES)


list_ds = tf.data.Dataset.list_files(str(DATA_DIR / "*/*"))

# for f in list_ds.take(5):
#     print(f.numpy())


def get_label(file_path):
    """A short pure-tensorflow function that converts a file paths to an (image_data, label) pair:"""
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)

    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in labeled_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())


def prepare_for_training(ds, cache=False, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))

# show_batch(image_batch.numpy(), label_batch.numpy())

