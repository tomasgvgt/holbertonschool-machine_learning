#!/usr/bin/env python3
"""
perform K-means on a dataset
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing
            the centroid means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the
            index of the cluster in C that each data point belongs to
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    clss = kmeans.labels_
    C = kmeans.cluster_centers_
    return C, clss
