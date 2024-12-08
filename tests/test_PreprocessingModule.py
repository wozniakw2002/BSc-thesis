import os
import pytest
import numpy as np
import pickle
from Modules.PreprocessingModule import Preprocessor

def test_normalization():
    data = np.array([[1,2,3], [4,5,6], [7,8,9]])
    normalized = Preprocessor.normalize_data(data)
    true_normalized = np.array([[1/255,2/255,3/255], [4/255,5/255,6/255], [7/255,8/255,9/255]])
    assert np.array_equal(true_normalized, normalized)

def test_resize_photo():
    data = [np.array([[1,2,3], [4,5,6], [7,8,9]])]
    resized = Preprocessor.resize_photos(data)[0]
    assert np.shape(resized) == (224,224)

def test_flip_photos():
    X = np.empty(1, dtype=object)
    X[0] = np.array([[1,2,3], [4,5,6], [7,8,9]])
    Y = np.array([1])
    X_fliped, Y_fliped = Preprocessor.flip_photos(X,Y)
    X_fliped_true = np.empty(2, dtype=object)
    X_fliped_true[0] = np.array([[3,2,1], [6,5,4], [9,8,7]])
    X_fliped_true[1] = X[0]
    Y_fliped_true = np.array([1,1])
    for i in range(2):
        assert np.array_equal(X_fliped[i], X_fliped_true[i])
    assert np.array_equal(Y_fliped, Y_fliped_true)


def test_augmentation():
    X = np.empty(1, dtype=object)
    X[0] = np.array([[1,2,3], [4,5,6], [7,8,9]])
    Y = np.array([1])
    X_fliped, Y_fliped = Preprocessor.augmentation(X,Y)
    X_fliped_true = np.empty(8, dtype=object)
    X_fliped_true[0] = X[0]
    X_fliped_true[1] = np.array([[9,8,7], [6,5,4], [3,2,1]])
    X_fliped_true[2] = np.array([[7,8,9], [4,5,6], [1,2,3]])
    X_fliped_true[3] = np.array([[3,2,1], [6,5,4], [9,8,7]])
    X_fliped_true[4] = np.array([[3,6,9], [2,5,8], [1,4,7]])
    X_fliped_true[5] = np.array([[7,4,1], [8,5,2], [9,6,3]])
    X_fliped_true[6] = np.array([[1,4,7], [2,5,8], [3,6,9]])
    X_fliped_true[7] = np.array([[9,6,3], [8,5,2], [7,4,1]])
    Y_fliped_true = np.repeat(np.array([1]), 8)
    for i in range(8):
        assert np.array_equal(X_fliped[i], X_fliped_true[i])
    assert np.array_equal(Y_fliped, Y_fliped_true)