import os
os.chdir("../")
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf

from keras import layers, models
from keras.utils import to_categorical

from src import pipelines
from src.data.extract_data import read_true_letters


%load_ext autoreload
%autoreload 2

RANDOM_STATE = 42  # for reproducibility
np.random.seed(RANDOM_STATE)

NUM_ELECTRODES = 64
FS = 240          # Hz
NUM_TRAIN_LETTERS = 85
NUM_TEST_LETTERS = 100
NUM_ROWCOLS = 12    # col:1-6 row:7-12
NUM_REPEAT = 15
SECONDS_TO_SLICE = 0.65    # after the simulation 0.65s data is treated as a sample

DATA_DIR = "./data/raw/BCI_Comp_III_Wads_2004/"