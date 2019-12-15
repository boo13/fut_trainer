# ###########################################################
#
# ###########################################################

# Built-ins
from pathlib import Path
from datetime import datetime

# Pips
from loguru import logger
import numpy as np

# ###########################################################
#                           Paths
# ###########################################################

# Data
DATA_DIR = Path("data/fifa_screenshot_classifier")
assert DATA_DIR.is_dir()

# Train
TRAIN_DIR = DATA_DIR / "train"
assert TRAIN_DIR.is_dir()

# Test
TEST_DIR = DATA_DIR / "test"
assert TEST_DIR.is_dir()

# Output
OUTPUT_DIR = Path("output")
assert OUTPUT_DIR.is_dir()

# Model output dir
MODEL_DIR = Path("models")

# Model output
MODEL_PATH = MODEL_DIR / "fifa_screenshot_classifier.model"

# define the path to the output learning rate finder plot
LRFIND_PLOT_PATH = OUTPUT_DIR / "lrfind_plot.png"

# define output path for training history plot
TRAINING_PLOT_PATH = OUTPUT_DIR / "training_plot.png"

CLASS_NAMES = np.array(
    [
        item.name
        for item in TRAIN_DIR.glob("*")
        if item.name != "LICENSE.txt" and item.name != ".DS_Store"
    ]
)

# Calculate number of images
TOTAL_NUM_TRAIN_IMAGES = len(list(TRAIN_DIR.glob("*/*.jpg")))
TOTAL_NUM_TEST_IMAGES = len(list(TEST_DIR.glob("*/*.jpg")))


# ###########################################################
#                           Config
# ###########################################################

IMG_WIDTH = 224
IMG_HEIGHT = 224

INIT_LR = 1e-2
EPOCHS = 25
BATCH_SIZE = 32
STEPS_PER_EPOCH = np.ceil(TOTAL_NUM_TRAIN_IMAGES / BATCH_SIZE)

# ###########################################################
#                           Fimctopms
# ###########################################################


def print_config_report():
    logger.info(f"Input image Dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    logger.info(f"Model will be saved to {MODEL_PATH}")
    logger.info(f"{len(CLASS_NAMES)} classes found")
    logger.info(f"CLASS_NAMES: {CLASS_NAMES}")
    logger.info(f"{TOTAL_NUM_TRAIN_IMAGES} train images")
    logger.info(f"{TOTAL_NUM_TEST_IMAGES} test images")
    logger.info(f"Learning Rate: {INIT_LR}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Steps per Epoch: {STEPS_PER_EPOCH}")
