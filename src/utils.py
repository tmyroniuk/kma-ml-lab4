import pickle

import matplotlib.pyplot as plt

from config import *
from preprocessing import *
from sklearn.cluster import KMeans

def plot_kmeans(X, num_clusters_range):
    distortions = []

    # Calculate distortions for different values of k
    for k in num_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    # Plot the elbow graph
    plt.plot(num_clusters_range, distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.show()

class Preprocessor:
    def __init__(self):
        with open(f'../{tfidf_vectorizer_model}', 'rb') as file:
            self.tfidf_vectorizer = pickle.load(file)
        with open(f'../{selector_model}', 'rb') as file:
            self.selector = pickle.load(file)
        with open(f'../{pca_model}', 'rb') as file:
            self.pca = pickle.load(file)
        with open(f'../{scaler_model}', 'rb') as file:
            self.scaler = pickle.load(file)
        with open(f'../{kmeans_model}', 'rb') as file:
            self.kemans = pickle.load(file)
        self.clean_text = clean_text
        self.vectorize = vectorize
        self.select_features = select_features
        self.reduce_dimentions = reduce_dimentions
        self.scale = scale
        self.predict_kmeans = predict_kmeans


    def transform(self, text):
        # Clean text
        text = self.clean_text(text)
        # Vectorize
        tfidf = self.vectorize(np.array([text]), self.tfidf_vectorizer)
        # Feature selection
        if use_feature_selection:
            tfidf, _, _ = self.select_features(tfidf, self.tfidf_vectorizer.get_feature_names_out(), self.selector)
        # PCA
        tfidf = self.reduce_dimentions(tfidf, self.pca)
        # Scale [0, 1]
        tfidf = self.scale(tfidf, self.scaler)
        # K-Means clusterization
        tfidf = self.predict_kmeans(tfidf, self.kemans)


    def transform_array(self, text):
        # Clean text
        text = text.apply(self.clean_text)
        # Vectorize
        tfidf = self.vectorize(text, self.tfidf_vectorizer)
        # Feature selection
        if use_feature_selection:
            tfidf, _, _ = self.select_features(tfidf, self.tfidf_vectorizer.get_feature_names_out(), self.selector)
        # PCA
        tfidf = self.reduce_dimentions(tfidf, self.pca)
        # Scale [0, 1]
        tfidf = self.scale(tfidf, self.scaler)
        # K-Means clusterization
        tfidf = self.predict_kmeans(tfidf, self.kemans)

        return tfidf