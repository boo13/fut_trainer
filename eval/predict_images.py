# import the necessary packages
from tensorflow.keras.models import load_model

# from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

from fifa_screenshot_classifier import config

# load the trained model from disk
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# grab the paths to the fire and non-fire images, respectively
print("[INFO] predicting...")

unsorted_images = list(paths.list_images(config.UNSORTED_DIR))

if config.SAMPLE_SIZE > 0:
    unsorted_images = unsorted_images[: config.SAMPLE_SIZE]

# loop over the sampled image paths
for (i, imagePath) in enumerate(unsorted_images):
    # load the image and clone it
    image = cv2.imread(imagePath)
    output = image.copy()

    # resize the input image to be a fixed 128x128 pixels, ignoring
    # aspect ratio
    image = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))
    image = image.astype("float32") / 255.0

    # make predictions on the image
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    j = np.argmax(preds)
    label = config.CLASSES[j]

    # draw the activity on the output frame
    # text = label if label == "Non-Fire" else "WARNING! Fire!"
    # output = imutils.resize(output, width=500)
    cv2.putText(output, label, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)

    # write the output image to disk
    filename = "{}.png".format(i)
    p = os.path.sep.join([str(config.OUTPUT_IMAGE_PATH), filename])
    cv2.imwrite(p, output)

