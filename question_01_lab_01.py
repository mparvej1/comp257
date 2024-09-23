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

# Step 1: Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target  # 70,000 instances and 784 features
print("Data shape:", X.shape)

# Step 2: Display some digits
def plot_digits(data, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):  # Display the first 25 digits
        plt.subplot(5, 5, i + 1)
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.show()

plot_digits(X, y)

# Step 3: PCA for the first and second principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio for first and second components:", explained_variance)

# Step 4: Plot projections onto a 1D hyperplane
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5)
plt.title('Projections of MNIST onto first two principal components')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar()
plt.show()

# Step 5: Incremental PCA to reduce to 154 dimensions
ipca = IncrementalPCA(n_components=154)
X_ipca = ipca.fit_transform(X)
print("Reduced shape using Incremental PCA:", X_ipca.shape)

# Step 6: Display original and compressed digits
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
