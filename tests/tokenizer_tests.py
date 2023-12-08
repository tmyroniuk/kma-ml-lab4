import pytest
import sys
import os

from scipy.sparse import csc_matrix
from sklearn.datasets import load_iris
import numpy as np

sys.path.append('./src/')
from preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer

@pytest.fixture
def sample_data():
    return [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

def test_fit_vectorize_with_new_vectorizer(sample_data):
    data_vectorized, vectorizer = fit_vectorize(sample_data)
    data_vectorized = data_vectorized.toarray()

    # Check if the result is a numpy array
    assert isinstance(data_vectorized, np.ndarray)

    # Check if the vectorizer is an instance of TfidfVectorizer
    assert isinstance(vectorizer, TfidfVectorizer)
    
    # Check the range of the number of documents in the vectorized data
    assert 0 <= data_vectorized.shape[0] <= len(sample_data)
    
    # Check the range of the number of features in the vectorized data
    assert 0 <= data_vectorized.shape[1] <= len(vectorizer.get_feature_names_out())

    # Check if all values in the resulting array are within the range [0, 1]
    assert np.all((data_vectorized >= 0) & (data_vectorized <= 1))

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def test_fit_scale():
    scaled_data, scaler_model = fit_scale(X)
    
    # Assert that the output is a tuple
    assert isinstance((scaled_data, scaler_model), tuple)
    
    # Assert that the scaled_data is a numpy array
    assert isinstance(scaled_data, np.ndarray)
    
    # Assert that the scaler_model is an instance of MinMaxScaler
    assert str(type(scaler_model)) == "<class 'sklearn.preprocessing._data.MinMaxScaler'>"
    
    # Assert that the scaled_data is within the range [0, 1]
    assert np.min(scaled_data) >= 0 and np.max(scaled_data) <= 1

def test_scale():
    _, scaler_model = fit_scale(X)
    
    scaled_data = scale(X, scaler_model)
    
    # Assert that the output is a numpy array
    assert isinstance(scaled_data, np.ndarray)
    
    # Assert that the scaled_data is within the range [0, 1]
    assert np.min(scaled_data) >= 0 and np.max(scaled_data) <= 1