"""
Simple implementation of k-Nearest Neighbors algorithm

@author Carrie Smith - carrie.elizabeth@gmail.Compute
Oct 2015
"""

import numpy as np
from scipy.stats import mode

class kNN(object):
    """
    Class for simple k-Nearest Neighbors
    """

    def __init__(self, k=5):
        """
        Initialize model.
        :param k: number of Neighbors (the 'k' in kNN)
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit model.
        :param X: pandas dataframe or numpy ndarray with features
        :param y: pandas series or numpy ndarray with classes
        """
        # kNN the model is entirely defined by k (number of neighbours)
        # and the training dataset
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X):
        """
        Predict classes of samples.
        :param X: pandas dataframe or numpy ndarray with features
        """
        
        # Convert input feature to numpy array
        X = np.array(X)

        # Pull number of items in training dataset and new dataset
        m_train = self.X.shape[0]
        m_pred = X.shape[0]

        # Distances = distance between 
        distances = np.zeros(shape = (m_train, 1))
        
        # Nearest = indices of k nearest neighbours in training X
        nearest = np.zeros(shape = (self.k,1))

        # Array of predicted classification
        y_pred = []

        # For each new observation
        for i in range(m_pred):
            # Compute euclidean distance between new observation (X[i,:]) and
            # each observation in the training sample (self.X)
            distances = ((X[i] - self.X)**2).sum(axis=1)**0.5

            # Find indices of k smallest distances
            nearest = np.argpartition(distances, self.k)[:self.k]

            # Choose the most common classification amongst k nearest neighbours
            most_common_neighbour = mode(self.y[nearest])[0].item()
            y_pred.append(most_common_neighbour)

        return y_pred


    def score(self, X, y):
        """
        Compute accuracy of predictions on X.
        :param X: pandas dataframe or numpy ndarray with features
        :param y: pandas series or numpy ndarray with true classes
        """
        
        y_pred = self.predict(X)
        m_pred = X.shape[0]

        accuracy = float((y_pred == y).sum()) / m_pred

        return accuracy
