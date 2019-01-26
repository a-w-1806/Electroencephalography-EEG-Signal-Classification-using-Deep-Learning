from scipy.io import loadmat

def load_data_from_mat(file_path):
    """ Read .mat file into a dict """
    data = loadmat(file_path)
    del data['__globals__'], data['__header__'], data['__version__']
    return data