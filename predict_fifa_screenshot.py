# import the necessary packages
from tensorflow.keras.models import load_model
from fifa_screenshot_classifier import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

from data_collect import screen_capture

# load the trained model from disk
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# # grab the paths to the fire and non-fire images, respectively
# print("[INFO] predicting...")
# firePaths = list(paths.list_images(config.FIRE_PATH))
# nonFirePaths = list(paths.list_images(config.NON_FIRE_PATH))

# # combine the two image path lists, randomly shuffle them, and sample them
# imagePaths = firePaths + nonFirePaths
# random.shuffle(imagePaths)
# imagePaths = imagePaths[: config.SAMPLE_SIZE]

# # # loop over the sampled image paths
# for (i, imagePath) in enumerate(imagePaths):
#     # load the image and clone it
#     image = cv2.imread(imagePath)
#     output = image.copy()

#     # resize the input image, ignoring aspect ratio
#     image = cv2.resize(image, (128, 128))
#     image = image.astype("float32") / 255.0

#     # make predictions on the image
#     preds = model.predict(np.expand_dims(image, axis=0))[0]
#     j = np.argmax(preds)
#     label = config.CLASS_NAMES[j]

#     # draw the activity on the output frame
#     text = label if label == "Non-Fire" else "WARNING! Fire!"
#     output = imutils.resize(output, width=500)
#     cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

#     # write the output image to disk
#     filename = "{}.png".format(i)
#     p = os.path.sep.join([config.OUTPUT_IMAGE_PATH, filename])
#     cv2.imwrite(p, output)


# # New version

cap = cv2.VideoCapture(0)

while True:

    ok, frame = cap.read()
    output = frame.copy()

    # resize the input image, ignoring aspect ratio
    frame = cv2.resize(frame, (config.IMG_WIDTH, config.IMG_HEIGHT))
    frame = frame.astype("float32") / 255.0

    # make predictions on the image
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    j = np.argmax(preds)
    label = config.CLASS_NAMES[j]

    cv2.putText(output, label, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    cv2.imshow("Predictio output", output)

    # esc to quit
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()

