from src.data import preprocessing
import numpy as np

def test_standardize_along():
    data = np.random.random([35,15,12,64,128])
    standardized = preprocessing.standardize_along(data, axis=-1)
    assert np.allclose(np.mean(standardized,axis=-1),np.full(data.shape[0:-1],0))
    assert np.allclose(np.std(standardized,axis=-1),np.full(data.shape[0:-1],1))