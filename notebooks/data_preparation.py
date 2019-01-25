A_train = pipelines.signal_mat_sub_band_norm(DATA_DIR+"Subject_A_Train")
A_test = pipelines.signal_mat_sub_band_norm(DATA_DIR + "Subject_A_Test")

A_train['signal'] = A_train['signal'].reshape([-1,NUM_ELECTRODES,A_train['signal'].shape[-1],1])
A_test['signal'] = A_test['signal'].reshape([-1,NUM_ELECTRODES,A_test['signal'].shape[-1],1]) 

A_train['label'] = to_categorical(A_train['label']) 

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(A_train['signal'], A_train['label'], test_size=0.05,
                                                  random_state=RANDOM_STATE, stratify=A_train['label'])

A_test_true_letters = read_true_letters(DATA_DIR+"A_test_labels.txt")
