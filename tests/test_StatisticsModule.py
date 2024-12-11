import os
import pytest
import numpy as np
import pickle
from tempfile import NamedTemporaryFile
from Modules.StatisticsModule import Statistics

def test_accuracy():
    Y = np.array([1,0,1,0,1,1])
    Y_test = np.array([1,0,1,0,0,0])
    acc = Statistics.accuracy(Y,Y_test)
    assert acc == '0,67'

def test_f1_score():
    Y = np.array([1,0,1,0,1,1])
    Y_test = np.array([1,0,1,0,0,0])
    f1 = Statistics.f1_score(Y, Y_test)
    assert f1 == '0,67'

def test_preccision():
    Y = np.array([1,0,1,0,1,1])
    Y_test = np.array([1,0,1,0,0,0])
    prec = Statistics.preccision(Y, Y_test)
    assert prec == '1,00'

def test_preccision_denominator_0():
    Y = np.array([1,0,1,0,1,1])
    Y_test = np.array([0,0,0,0,0,0])
    prec = Statistics.preccision(Y, Y_test)
    assert prec == '0,00'

def test_recall():
    Y = np.array([1,0,1,0,1,1])
    Y_test = np.array([1,0,1,0,0,0])
    prec = Statistics.recall(Y, Y_test)
    assert prec == '0,50'

def test_recall_denominator_0():
    Y = np.array([1,0,1,0,1,1])
    Y_test = np.array([0,1,0,1,0,0])
    prec = Statistics.recall(Y,Y_test)
    assert prec == '0,00'

def test_plot_roc_curve():
    Y = np.array([1,0,1,0,1,1])
    Y_prob = np.random.uniform(0,1, 6)

    temp_path = 'tem_path.png'
    
    try:
        Statistics.plot_roc_curve(Y, Y_prob, True, temp_path)
        assert os.path.exists(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_plot_probability_histogram():
    Y_prob = np.random.uniform(0,1, 100)

    temp_path = 'tem_path.png'
    
    try:
        Statistics.plot_probability_histogram(Y_prob, True, temp_path)
        assert os.path.exists(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_plot_confusion_matrix():
    Y = np.array([1,0,1,0,1,1])
    Y_test = np.array([1,0,1,0,0,0])

    temp_path = 'tem_path.png'
    
    try:
        Statistics.plot_confusion_matrix(Y, Y_test, True, temp_path)
        assert os.path.exists(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_plot_learning_curve():
    Y = np.random.normal(5, 1, 40)
    Y_test = np.random.normal(5, 1, 40)

    temp_path = 'tem_path.png'
    
    try:
        Statistics.plot_learning_curve(Y, Y_test, True, temp_path)
        assert os.path.exists(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)