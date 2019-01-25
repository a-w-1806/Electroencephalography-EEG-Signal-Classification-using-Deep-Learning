import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from src import pipelines
from src.data.extract_data import read_true_letters
from src.models.PAMI import CNN_1_P300_PAMI_BCIIII
from src.pipelines import PARADIGM, testing_pipeline

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_dir",help = "the directory of data")
parser.add_argument("subject",choices = ['A','a','B','b'])
parser.add_argument("-Ns",type = int,help = "Number of filters in the first conv layer")
parser.add_argument("-o","--optimizer")
parser.add_argument("-s","--seconds_to_slice",type = float,default=0.65,
                    help = "how long time after the signal should be included in one single data point")
parser.add_argument("-r","--random_seed",type = int,default=42)
parser.add_argument("-t","--test_size",type = float,default = 0.05,
                    help = "proportion of validation data in training data")
parser.add_argument("-p","--patience",type = int,help = "patience while training")
parser.add_argument("-b","--batch_size",type = int)
args = parser.parse_args()

RANDOM_STATE = args.random_seed
np.random.seed(RANDOM_STATE)   # for reproducibility
DATA_DIR = args.data_dir
SUBJECT = args.subject
SECONDS_TO_SLICE = args.seconds_to_slice    # after the simulation 0.65s data is treated as a sample

NUM_ELECTRODES = 64
FS = 240          # Hz
NUM_TRAIN_LETTERS = 85
NUM_TEST_LETTERS = 100
NUM_ROWCOLS = 12    # col:1-6 row:7-12
NUM_REPEAT = 15

TRAINING_FILE = "Subject_" + SUBJECT.upper() + "_Train"
TEST_FILE = "Subject_" + SUBJECT.upper() + "_Test"
TEST_LABEL_FILE = SUBJECT.upper() + "_test_labels.txt"

##############################################################################################
train = pipelines.signal_mat_sub_band_norm(DATA_DIR + TRAINING_FILE)
test = pipelines.signal_mat_sub_band_norm(DATA_DIR + TEST_FILE)

train['signal'] = train['signal'].reshape([-1, NUM_ELECTRODES, train['signal'].shape[-1], 1])
test['signal'] = test['signal'].reshape([-1, NUM_ELECTRODES, test['signal'].shape[-1], 1])

train['label'] = to_categorical(train['label'])

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train['signal'], train['label'], test_size=args.test_size,
                                                  random_state=RANDOM_STATE, stratify=train['label'])

A_test_true_letters = read_true_letters(DATA_DIR+TEST_LABEL_FILE)

##########################################################################################################
model_PAMI = CNN_1_P300_PAMI_BCIIII(Ns = args.Ns,seconds_to_slice=SECONDS_TO_SLICE)
model_PAMI.compile(optimizer = args.optimizer,loss = 'binary_crossentropy',metrics = ['acc', 'mse'])
earlystopping = EarlyStopping(monitor = "val_mean_squared_error",patience = args.patience)
log = model_PAMI.fit(x = X_train, y = y_train, batch_size=args.batch_size, epochs = 1000, callbacks = [earlystopping],
              validation_data = [X_val,y_val])
model_PAMI_scores = testing_pipeline(test['signal'],test['code'],model_PAMI,"all",PARADIGM,
                                    A_test_true_letters)
