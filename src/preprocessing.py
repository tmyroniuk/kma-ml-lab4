import re
import pickle

import numpy as np

from config import random_state, num_clusters, random_state, num_clusters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def clean_text(text):
    text = text.lower()

    # Replace special characters with spaces
    text = re.sub(re.compile(r'[^a-zA-Z0-9\s\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\u2600-\u26FF\u2700-\u27BF\U0001F004]+'), ' ', text)

    # Remove subsequent spaces
    text = re.sub(re.compile(r'\s+'), ' ', text)
    return text.strip()


def fit_vectorize(data, vectorizer=None):
    """
    Fit Tfidf model on data and calculate Tfidf matrix.

    Parameters:
    - data (list[str], ndarray or Iterable): Text to be vectorized.
    - vectorizer (sstr or Any, optional): The vectorizer model or .pkl filepath with vectorizer model dump. If None new TfidfVectorizer is created.

    Returns:
    tuple:
        - ndarray or None: Vectorized text.
        - TfidfVectorizer: Vectorizer model.
    """
    # Initialize vectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
    elif isinstance(vectorizer, str):
        with open(vectorizer, 'rb') as file:
            vectorizer = pickle.load(file)

    data_vectorized = vectorizer.fit_transform(data)
    return data_vectorized, vectorizer


def vectorize(data, vectorizer):
    """
    Vectorize data using existing vectorizer model.

    Parameters:
    - data (list[str], ndarray or Iterable): Text to be vectorized.
    - vectorizer (str or Any): The vectorizer model or .pkl filepath with vectorizer model dump.

    Returns:
    ndarray or None: Vectorized text.
    """
    # Initialize vectorizer
    if isinstance(vectorizer, str):
        with open(vectorizer, 'rb') as file:
            vectorizer = pickle.load(file)

    data_vectorized = vectorizer.transform(data)
    return data_vectorized


def __handle_feature_names(selector, feature_names):
    # Get the chi-squared scores of the selected features
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_scores = selector.scores_[selected_feature_indices]

    # Get the names of the selected features
    selected_feature_names = feature_names[selected_feature_indices]

    # Sort the selected features by their scores in descending order
    sorted_indices = selected_feature_scores.argsort()[::-1]
    selected_feature_names = selected_feature_names[sorted_indices]
    selected_feature_scores = selected_feature_scores[sorted_indices]

    return selected_feature_names, selected_feature_scores


def fit_select_features(X, y, feature_names, selector=None, features_to_keep=1.0):
    """
    Fit chi2 model for feature selection.

    Parameters:
    - X (ndarray or DataFrame): Arguments matrix.
    - y (ndarray or DataFrame): Target labels matrix.
    - feature_names (ArrayLike): List of feature names.
    - selector (str or Any, optional): The selector model or .pkl filepath with selector model dump. If None new SelectKBest is created.
    - features_to_keep (float or int, optional): Number of features to be selected as ratio (if float) or absolute value (if int).
    
    Returns:
    tuple:
        - ndarray or crc_matrix: X matrix with only selected features.
        - SelectKBest: Selector model.
        - ArrayLike: Selected features names.    
        - ArrayLike: Selected features scores.
    """
    n_features = X.shape[1]

    # Initialize selector
    if selector is None:
        if isinstance(features_to_keep, float):
            selector = SelectKBest(chi2, k=int(features_to_keep*n_features))
        elif isinstance(features_to_keep, int):
            selector = SelectKBest(chi2, k=int(features_to_keep))
    elif isinstance(selector, str):
        with open(selector, 'rb') as file:
            selector = pickle.load(file)

    # Fit feature selection model
    X_selected = selector.fit_transform(X, y)
    
    selected_feature_names, selected_feature_scores = __handle_feature_names(selector, feature_names)

    return X_selected, selector, selected_feature_names, selected_feature_scores


def select_features(X, feature_names, selector):
    """
    Feature selection using existing chi2 model.

    Parameters:
    - X (ndarray or DataFrame): Arguments matrix.
    - feature_names (ArrayLike): List of feature names.
    - selector (str or Any): The selector model or .pkl filepath with selector model dump.

    Returns:
    tuple:
        - ndarray or crc_matrix: X matrix with only selected features.
        - ArrayLike: Selected features names.    
        - ArrayLike: Selected features scores. 
    """

    # Initialize selector
    if isinstance(selector, str):
        with open(selector, 'rb') as file:
            selector = pickle.load(file)

    # Fit feature selection model
    X_selected = selector.transform(X)
    
    selected_feature_names, selected_feature_scores = __handle_feature_names(selector, feature_names)

    return X_selected, selected_feature_names, selected_feature_scores


def fit_reduce_dimentions(X, pca=None, n_components=1.0):
    """
    Fit PCA model on provided data.

    Parameters:
    - X (crc_matrix): Argument sparce matrix.
    - pca (str or PCA, optional): PCA model or pkl filepath with PCA model dump. If None new PCA model is created with provided n_components.
    - n_components (float or int): Determines number of components in result as varience threshold (if float) or absolute value (if int).
    
    Returns:
    - PCA: Fitted PCA model.
    """
    # Initialize PCA
    if pca is None:
        pca = PCA(n_components=n_components, random_state=random_state)
    elif isinstance(pca, str):
        with open(pca, 'rb') as file:
            pca = pickle.load(file)

    pca.fit(X.toarray())
    return pca


def reduce_dimentions(X, pca, batch_size = 10000):
    """
    Dimentions reduction using existing PCA model in batches.

    Parameters:
    - X (crc_matrix): Argument sparce matrix.
    - pca (str or PCA): PCA model or pkl filepath with PCA model dump.
    - batch_size (int): Size of processed batch.

    Returns:
    ndarray: New X matrix with reduced dimentions.
    """
    # Initialize PCA
    if isinstance(pca, str):
        with open(pca, 'rb') as file:
            pca = pickle.load(file)

    # Transform the data in batches
    num_batches = (X.shape[0] + batch_size - 1) // batch_size
    pca_result = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X.shape[0])
        X_batch = X[start_idx:end_idx, :].toarray()
        transformed_batch = pca.transform(X_batch)
        pca_result.append(transformed_batch)

    # Concatenate the batches results
    pca_result = np.concatenate(pca_result, axis=0)
    return pca_result

def fit_scale(X, scaler=None):
    """
    Parameters:
    - X (ndarray): Argument matrix.
    - scaler(str or Any, optional): Scaler model or pkl filepath with scaler model dump. If None new MinMaxScaler will be created.

    Returns:
    tuple:
        - ndarray: Scaled argument matrix.
        - Any: Fitted scaler model.
    """
    # Initialize Scaler
    if scaler is None:
        scaler = MinMaxScaler()
    elif isinstance(scaler, str):
        with open(scaler, 'rb') as file:
            scaler = pickle.load(file)
    
    result = scaler.fit_transform(X)
    return result, scaler


def scale(X, scaler):
    """
    Parameters:
    - X (ndarray): Argument matrix.
    - scaler(str or Any, optional): Scaler model or pkl filepath with scaler model dump.

    Returns:
    ndarray: Scaled argument matrix.
    """
    # Initialize Scaler
    if isinstance(scaler, str):
        with open(scaler, 'rb') as file:
            scaler = pickle.load(file)
    
    result = scaler.transform(X)
    return result


def fit_kmeans(X):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=random_state)
    kmeans.fit(X)
    return kmeans

    
def predict_kmeans(X, kmeans):
    # Initialize KMeans
    if isinstance(kmeans, str):
        with open(kmeans, 'rb') as file:
            kmeans = pickle.load(file)

    cluster_labels = kmeans.predict(X)
    # Add cluster labels as new columns to X
    res = np.column_stack((X, np.eye(num_clusters)[cluster_labels]))
    return res