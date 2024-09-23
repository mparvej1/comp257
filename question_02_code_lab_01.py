# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:26:35 2024

@author: xxparvej
"""


# Import necessary libraries
import numpy as np
from sklearn.datasets import make_swiss_roll

# 1: Generate Swiss Roll dataset [5 points]
X, y = make_swiss_roll(n_samples=1000, noise=0.1)

# Output the shape of the generated dataset
print("Swiss roll data shape:", X.shape)


# Import libraries for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2: Plot the Swiss Roll dataset [2 points]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the Swiss roll dataset
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='Spectral')

# Setting labels and title
plt.title('Swiss Roll Dataset')
plt.xlabel('X axis')
plt.ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()



# Import necessary libraries for Kernel PCA
from sklearn.decomposition import KernelPCA

# 3: Apply Kernel PCA with different kernels [6 points]
kernels = ['linear', 'rbf', 'sigmoid']  # Define the kernels to use
kpca_results = {}

# Loop through each kernel and apply Kernel PCA
for kernel in kernels:
    kpca = KernelPCA(kernel=kernel, n_components=2)  # Using 2 components for visualization
    X_kpca = kpca.fit_transform(X)  # Fit and transform the Swiss roll dataset
    kpca_results[kernel] = X_kpca   # Store the results in dictionary


# Import libraries for plotting kPCA results
import matplotlib.pyplot as plt

# 4: Plot kPCA results for each kernel and compare [6 points]
plt.figure(figsize=(15, 5))  # Create a figure with a specified size

# Loop through each kernel and plot the corresponding kPCA results
for i, (kernel, X_kpca) in enumerate(kpca_results.items()):
    plt.subplot(1, 3, i + 1)  # Create a subplot for each kernel
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='Spectral')  # Scatter plot of the two principal components
    plt.title(f'Kernel PCA with {kernel} kernel')  # Set the title for each plot
    plt.xlabel('First Component')  # Label for x-axis
    plt.ylabel('Second Component')  # Label for y-axis

plt.tight_layout()  # Adjust subplots to fit the figure area
plt.show()  # Display the plot



# Import libraries for Logistic Regression and GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Binarizer

# 5: Logistic Regression with GridSearchCV [14 points]

# Binarize the target variable 'y'
y_binarized = Binarizer(threshold=y.mean()).fit_transform(y.reshape(-1, 1)).ravel()

# Create a pipeline with Kernel PCA and Logistic Regression
pipeline = Pipeline([
    ('kpca', KernelPCA(n_components=2)),  # Set n_components to 2 for Kernel PCA
    ('logistic', LogisticRegression(max_iter=1000))  # Logistic Regression for classification
])

# Define the parameter grid for GridSearchCV
param_grid = [
    {'kpca__kernel': ['linear']},  # Search with linear kernel
    {'kpca__kernel': ['rbf', 'sigmoid'], 'kpca__gamma': [0.1, 1, 10]}  # Search with RBF and Sigmoid kernels, varying gamma
]

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5)  # 5-fold cross-validation

# Fit the GridSearchCV object on the data
grid_search.fit(X, y_binarized)  # Ensure y_binarized is the correct target variable

# 6: Plot the results from using GridSearchCV in (5). [2 points]
print("Best parameters found by GridSearchCV:", grid_search.best_params_)
