BCI_III_P300
=============

- ~~set_random_seed~~ `np.random.seed()`

## Data

Subject_A
Subject_B
- Train
    - Signal (85 * 7794 * 64)
    - Flashing
    - StimulusCode
    - StimulusType
    - TargetChar (85)
- Test
  - Signal (100 * 7794 * 64)
  - StimulusCode (100 * 7794)
  - Flashing


## Code to write

### Data Preparation

- ~~.mat to nparray~~ `extract_data()`
    - Train
        - signal (85 * 15 * 12 * 64 * 156)
            - PAMI slice **0.65s**
        - label (85 * 15 * 12)
            - 0 0 0 ... 1 0 0 ..
        - code (85 * 15 * 12)
            - [12,11,3,10 ...]
        - targetchar
            - 'EAEVQTDOJG8RBRGONCEDHCTUIDB
              '
    - Test
        - signal (100, 15, 12, 64, 156)
        - code (100, 15, 12)
            - [12,11,3,10 ...]

- ~~subsample~~ `subsample()`
- ~~bandpass filter~~ `butter_band_pass_filter()`
- ~~normalize along channel~~ `standardize_along()`

### Build a model


- ~~`np.expand_dims()` then use conv2d~~

- `reshape()` data:4D [all,num_electrodes,timesteps,1] label: 1D

- `to_categorical()` label: 2D

- ~~Stratified~~ `train_test_split()`

- ~~PAMI model~~ `CNN_1_P300_PAMI_BCIIII()`

### Training

- EarlyStopping 'val_mean_squared_error'
- ~~print and return all stats~~ `print_all_stats()`
- write log to a file (maybe?)
- save model

### Testing

- ~~`extract_data()`~~
- ~~`subsample()`~~
- ~~`bandpass()`~~
- ~~normalize along channel~~
- `signal_mat_sub_band_norm()` (85, 15, 12, 64, 78)
- `reshape()` data:4D

- `predict()` 2D --[num_samples,num_classes]


- stats
  - `model.predict()` [num,2]
  - `argmax()` [num,]


- letters `test_pipeline()` 
  - ~~drop the 0's column~~ [num_samples,]
  - ~~`reshape()` back to~~ --[num_letters,15,12]
  - ~~sort [12] using 'code'~~ -- [num_letters,15,12]
  - ~~aggregate or not~~ --[num_letters,12]
  - ~~[12] to letter~~  --[num_samples,]
      - ~~first [6] last [6]~~ `prob_to_rowcols`  --(num_letters,2) row/column  0-5->1-6  0-5->7-12
      - ~~look up~~ `letter_lookup()`
  - calculate accuracy

DESKTOP-T9MLK18







# Thought

-[ ] customize loss function ROC of 1~15 (accuracy or ITR ?)
-[ ] New model  

