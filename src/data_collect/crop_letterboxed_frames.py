from pathlib import Path

# Pip Installs
import tensorflow as tf
import cv2
import pandas as pd
from loguru import logger

logger.debug(f"Tensorflow Version: {tf.__version__}")

DATA_DIR = Path("../../data")
assert DATA_DIR.is_dir()

OUTPUT_DIR = Path("../../output")
assert OUTPUT_DIR.is_dir()

# If we have a frame from the Mira capture card we have a letter-boxed frame...
# Then crop the frame to lose the black bars on top and bottom of the frame
if width == 640 and height == 480:
    frame = frame[60:420, 0:width]