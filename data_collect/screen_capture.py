"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""
# Built-ins
import os
import random
import string
from pathlib import Path
from datetime import datetime

# Pip Installs
import tensorflow as tf
import cv2
import pandas as pd
from loguru import logger
import imutils

# logger.debug(f"Tensorflow Version: {tf.__version__}")

DATA_DIR = Path("data")
assert DATA_DIR.is_dir()

# OUTPUT_DIR = Path("../../data/raw")
# assert OUTPUT_DIR.is_dir()

# from fifa_screenshot_classifier import config

def random_string(stringLength=4):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))


def screen_capture(vid_source=0, show=True, save=True, save_timer=30):
    screenshot_count = 0
    session_id = random_string()

    cap = cv2.VideoCapture(vid_source)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = round(1000 / fps)

    # Print video properties
    logger.info(f"Video Captured")
    logger.debug(f"Session ID: {session_id}")
    logger.debug(f"{width}px X {height}px")
    logger.debug(f"FPS: {fps}    Frame Time: {frame_time}")

    t = datetime.now()

    while True:
        ok, frame = cap.read()

        # Exit loop if no frame found
        if not ok:
            break

        # If we have a frame from the Mira capture card we have a letter-boxed frame...
        # Then crop the frame to lose the black bars on top and bottom of the frame
        if width == 640 and height == 480:
            frame = frame[60:420, 0:width]

        # Display the frame being captured (Default: True)
        if show:
            cv2.imshow(f"Video Source: {vid_source}", frame)

        # Check the timer, used for screenshooting
        delta = datetime.now() - t

        # Take a screenshot if we're saving (Default: True)
        if save and delta.seconds >= save_timer:
            screenshot_count += 1
            _fn = str(OUTPUT_DIR / f"{session_id}_{screenshot_count:05}.jpg")
            cv2.imwrite(_fn, frame)
            logger.debug(f"[SCREENSHOT] {_fn}")
            t = datetime.now()

        # esc to quit
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Records screenshots from hardware-connected video stream"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without saving screenshots"
    )

    args = parser.parse_args()

    if not args.dry_run:
        logger.info("Preparing to save screenshots...")
        screen_capture()

    else:
        logger.debug("Not Saving screenshots")
