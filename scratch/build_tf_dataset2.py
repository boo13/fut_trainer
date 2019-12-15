import tensorflow as tf
print(tf.__version)

import pandas as pd
import tf.data.Dataset

train_csv = pd.read_csv('data/train.csv')# Prepend image filenames in train/ with relative path
filenames = ['train/' + fname for fname in train_csv['id'].tolist()]
labels = train_csv['has_cactus'].tolist()

train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames, labels, train_size=0.9, random_state=42)


train_data = tf.data.Dataset.from_tensor_slices(
  (tf.constant(train_filenames), tf.constant(train_labels))
)val_data = tf.data.Dataset.from_tensor_slices(
  (tf.constant(val_filenames), tf.constant(val_labels))
)

IMAGE_SIZE = 96 # Minimum image size for use with MobileNetV2BATCH_SIZE = 32# Function to load and preprocess each image
def _parse_fn(filename, label):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img)
    img = (tf.cast(img, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, label
# Run _parse_fn over each example in train and val datasets
# Also shuffle and create batchestrain_data = (train_data.map(_parse_fn)
             .shuffle(buffer_size=10000)
             .batch(BATCH_SIZE)
             )val_data = (val_data.map(_parse_fn)
           .shuffle(buffer_size=10000)
           .batch(BATCH_SIZE)
           )

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)# Pre-trained model with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)# Freeze the pre-trained model weights
base_model.trainable = False# Trainable classification head
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')# Layer classification head with feature detector
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])learning_rate = 0.0001# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy']
)