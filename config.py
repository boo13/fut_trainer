from pathlib import Path
import tensorflow as tf


DATA_DIR = Path("data/classifier")
assert DATA_DIR.is_dir()

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
assert TRAIN_DIR.is_dir()
assert TEST_DIR.is_dir()

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
