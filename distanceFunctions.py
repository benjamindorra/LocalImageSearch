"""
Common distance functions for similarity search
"""

import numpy as np

# https://www.tutorialexample.com/best-practice-to-calculate-cosine-distance-between-two-vectors-in-numpy-numpy-tutorial/


def cos_dist(vector1, vector2):
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    v1_norm = np.linalg.norm(vector1)
    v2_norm = np.linalg.norm(vector2)
    prod = np.dot(vector1, vector2)
    epsilon = 1e-8
    cos = prod / (v1_norm * v2_norm + epsilon)
    return 1 - cos


def euclidean_dist(vector1, vector2):
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    vector1 /= np.linalg.norm(vector1, axis=-1, keepdims=True)
    vector2 /= np.linalg.norm(vector2, axis=-1, keepdims=True)
    diff = np.subtract(vector1, vector2)
    diff_squared = np.square(diff)
    total = np.sum(diff_squared)
    root = np.sqrt(total)
    return root


def manhattan_dist(vector1, vector2):
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    vector1 /= np.linalg.norm(vector1, axis=-1, keepdims=True)
    vector2 /= np.linalg.norm(vector2, axis=-1, keepdims=True)
    diff = np.subtract(vector1, vector2)
    diff_abs = np.absolute(diff)
    total = np.sum(diff_abs)
    return total


def chebyshev_dist(vector1, vector2):
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    vector1 /= np.linalg.norm(vector1, axis=-1, keepdims=True)
    vector2 /= np.linalg.norm(vector2, axis=-1, keepdims=True)
    diff = np.subtract(vector1, vector2)
    diff_abs = np.absolute(diff)
    max_diff = np.max(diff_abs)
    return max_diff


def jaccard_dist(vector1, vector2):
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    vector1 /= np.linalg.norm(vector1, axis=-1, keepdims=True)
    vector2 /= np.linalg.norm(vector2, axis=-1, keepdims=True)
    intersection = np.sum(np.multiply(vector1, vector2))
    union = np.sum(vector1) + np.sum(vector2) - intersection
    epsilon = 1e-8
    jaccard_index = intersection / (union + epsilon)
    return 1 - jaccard_index
