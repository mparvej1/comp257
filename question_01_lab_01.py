# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:28:20 2024

@author: xxparvej
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import fetch_openml

#1. Retrieve and load the mnist_784 dataset of 70,000 instances. [5 points]
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target  # 70,000 instances and 784 features
print("Data shape:", X.shape)

#2. Display each digit. [5 points]
def plot_digits(data, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):  # Display the first 25 digits
        plt.subplot(5, 5, i + 1)
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.show()

plot_digits(X, y)

#3. Use PCA to retrieve the and principal component and output their explained variance ratio. [5 points]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio for first and second components:", explained_variance)

# 4. Plot the projections of the and principal component onto a 1D hyperplane. [5 points]
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5)
plt.title('Projections of MNIST onto first two principal components')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar()
plt.show()

# 5. Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. [10 points]
ipca = IncrementalPCA(n_components=154)
X_ipca = ipca.fit_transform(X)
print("Reduced shape using Incremental PCA:", X_ipca.shape)

# 6. Display the original and compressed digits from (5). [5 points]
def plot_compressed_digits(original, compressed, n=10):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        # Original digits
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # Reconstructed from compressed digits
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(ipca.inverse_transform(compressed[i]).reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

plot_compressed_digits(X, X_ipca)
