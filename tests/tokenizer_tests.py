import pytest
import sys
import os

from scipy.sparse import csc_matrix
from sklearn.datasets import load_iris
import numpy as np

sys.path.append('./src/')
from preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer


def test_fit_vectorize():
    # Test with random text data
    text_data = ["This is a test.", "Another example.", "Random text."]
    text_data = list(map(clean_text, text_data))
    data_vectorized, vectorizer_model = fit_vectorize(text_data)
    check_vectorization_result(data_vectorized, vectorizer_model)

    # Test with a single text
    single_text = ["This is a single text."]
    single_text = list(map(clean_text, single_text))
    data_vectorized_single, vectorizer_single = fit_vectorize(single_text)
    check_vectorization_result(data_vectorized_single, vectorizer_single)

def test_vectorize():
    # Test with random text data
    text_data = ["This is a test.", "Another example.", "Random text."]
    text_data = list(map(clean_text, text_data))
    _, vectorizer_model = fit_vectorize(text_data)
    data_vectorized = vectorize(text_data, vectorizer_model)
    check_vectorization_result(data_vectorized, vectorizer_model)

    # Test with a single text
    single_text = ["This is a single text."]
    single_text = list(map(clean_text, single_text))
    _, vectorizer_model = fit_vectorize(single_text)
    data_vectorized_single = vectorize(single_text, vectorizer_model)
    check_vectorization_result(data_vectorized_single, vectorizer_model)

    # Test with same model
    sample_text = ["This is a text."]
    sample_text = list(map(clean_text, sample_text))
    data_vectorized_sample = vectorize(sample_text, vectorizer_model)
    check_vectorization_result(data_vectorized_sample, vectorizer_model)

    # Test with an empty text (edge case)
    empty_text = [""]
    data_vectorized_empty = vectorize(empty_text, vectorizer_model)
    check_vectorization_result(data_vectorized_empty, vectorizer_model)

def check_vectorization_result(data_vectorized, vectorizer_model):
    # Check that the output is a tuple
    assert isinstance((data_vectorized, vectorizer_model), tuple)

    # Check that the vectorizer_model is an instance of TfidfVectorizer
    assert isinstance(vectorizer_model, TfidfVectorizer)

    # Check that the data_vectorized has the correct shape
    assert data_vectorized.shape[0] > 0
    assert data_vectorized.shape[1] > 0

@pytest.fixture
def sample_data():
    return [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

def test_vectorizr_output_format(sample_data):
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

def test_fit_scale():
    # Test with random data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    scaled_data, scaler_model = fit_scale(X)
    check_scaling_result(scaled_data, scaler_model)

    # Test with a single data point
    single_data_point = np.array([[1, 2, 3]])
    scaled_single, scaler_single = fit_scale(single_data_point)
    check_scaling_result(scaled_single, scaler_single)

    # Test with all zeros (edge case)
    zeros_data = np.zeros((10, 3))
    scaled_zeros, scaler_zeros = fit_scale(zeros_data)
    check_scaling_result(scaled_zeros, scaler_zeros)

def test_scale():
    # Test with random data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    _, scaler_model = fit_scale(X)
    scaled_data = scale(X, scaler_model)
    check_scaling_result(scaled_data, scaler_model)

    # Test with a single data point
    single_data_point = np.array([[1, 2, 3]])
    _, scaler_model = fit_scale(single_data_point)
    scaled_single = scale(single_data_point, scaler_model)
    check_scaling_result(scaled_single, scaler_model)

    # Test with all zeros (edge case)
    zeros_data = np.zeros((10, 3))
    _, scaler_model = fit_scale(zeros_data)
    scaled_zeros = scale(zeros_data, scaler_model)
    check_scaling_result(scaled_zeros, scaler_model)

def check_scaling_result(scaled_data, scaler_model):
    # Check that the output is a tuple
    assert isinstance((scaled_data, scaler_model), tuple)

    # Check that the scaled_data is a numpy array
    assert isinstance(scaled_data, np.ndarray)

    # Check that the scaler_model is an instance of MinMaxScaler
    assert isinstance(scaler_model, MinMaxScaler)

    # Check that the scaled_data is within the range [0, 1]
    assert np.min(scaled_data) >= 0 and np.max(scaled_data) <= 1


def test_predict_kmeans():
    # Test with random data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    kmeans = fit_kmeans(X)
    result = predict_kmeans(X, kmeans)
    check_encoding(result, kmeans)

    # Test with a single data point
    single_data_point = np.array([[0.5, 0.5]])
    result_single = predict_kmeans(single_data_point, kmeans)
    check_encoding(result_single, kmeans)

    # Test with duplicate data points
    duplicate_data = np.array([[0.1, 0.2], [0.1, 0.2], [0.3, 0.4]])
    result_duplicate = predict_kmeans(duplicate_data, kmeans)
    check_encoding(result_duplicate, kmeans)

    # Test with fewer data points than clusters
    fewer_data_than_clusters = np.array([[0.1, 0.2], [0.3, 0.4]])
    result_fewer = predict_kmeans(fewer_data_than_clusters, kmeans)
    check_encoding(result_fewer, kmeans)

def check_encoding(result, kmeans):
    # Check if one-hot encoding is correct
    for row in result:
        assert np.sum(row[-num_clusters:]) == 1
        assert np.argmax(row[-num_clusters:]) == kmeans.predict([row[:-num_clusters]])[0]
