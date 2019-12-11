"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import tensorflow as tf
import cv2
import os
from pathlib import Path
import pandas as pd
from loguru import logger
import imutils

logger.debug(f"Tensorflow Version: {tf.__version__}")

DATA_DIR = Path("../../data")
assert DATA_DIR.is_dir()

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ok, img = cap.read()

        if ok:
            cv2.imshow('webcam', img)
        else:
            break
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Records screenshots from hardware-connected video stream")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving screenshots")

    args = parser.parse_args()

    if not args.dry_run:
        logger.debug("Preparing to save screenshots...")
        main()

    else:
        logger.debug("Not Saving screenshots")