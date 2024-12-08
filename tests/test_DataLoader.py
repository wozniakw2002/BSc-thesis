import os
import pytest
import numpy as np
import pickle
from tempfile import NamedTemporaryFile
from Modules.DataLoaderModule import DataLoader


def test_loaded_photos_number():
    loader = DataLoader()
    data, Y = loader.load_data_from_folder_as_png('data')
    assert len(data) == 2991
    assert len(Y) == 2991

def test_load_single_photo():
    photo = DataLoader.load_single_photo(r'data\Valid\Valid\Normal\9003126R.png')
    assert type(photo) == list

def test_save_data_as_array():
    X_test = np.array([[1, 2, 3], [4, 5, 6]])
    Y_test = np.array([0, 1])

    temp_path = 'tem_path.pkl'
    
    try:
        DataLoader.save_data_as_array(temp_path, X_test, Y_test)
        assert os.path.exists(temp_path)
        with open(temp_path, 'rb') as infile:
                X, Y = pickle.load(infile)

        assert np.array_equal(X_test, X)
        assert np.array_equal(Y_test, Y)
    finally:
         if os.path.exists(temp_path):
            os.remove(temp_path)
    

def test_load_data_as_array():
    X_test = np.array([[1, 2, 3], [4, 5, 6]])
    Y_test = np.array([0, 1])

    temp_path = 'tem_path.pkl'
    DataLoader.save_data_as_array(temp_path, X_test, Y_test)

    try:
        X,Y = DataLoader.load_data_as_array(temp_path)
        assert np.array_equal(X_test, X)
        assert np.array_equal(Y_test, Y)
    finally:
            os.remove(temp_path)

def test_load_pickle_file():
    file = [1,2,3,4,5,3,4]

    temp_path = 'tem_path.pkl'
    with open(temp_path, 'wb') as outfile:
            pickle.dump(file, outfile)

    try:
        file_loaded = DataLoader.load_pickle_file(temp_path)
        assert file_loaded == file
    finally:
            os.remove(temp_path)