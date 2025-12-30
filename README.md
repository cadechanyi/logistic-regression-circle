# Logistic Regression on Synthetic Circle Dataset

This project implements logistic regression from scratch to classify points as inside or outside a circle. The model demonstrates the limitations of linear decision boundaries and the benefits of nonlinear feature mapping in scientific machine learning.


Overview/Motivation:
  - Logistic regression is a foundational machine learning algorithm for binary classification. This project explores it in a controlled, scientific       context:
    - Dataset: synthetic 2D points labeled by their position relative to a unit circle.
    - Objective: train a logistic regression model from scratch and analyze its performance.
    - Extension: improve classification with nonlinear feature mapping.

Features:
  - From-scratch implementation using NumPy (no ML libraries)
  - Gradient descent optimization with learning rate analysis
  - Visualization of data points, true circle, and learned decision boundary
  - Optional: feature mapping to handle nonlinear decision boundaries

Dataset:
  - Synthetic 2D points sampled uniformly in [-2,2] x [-2, 2]
  - Labels: 1 if inside the unit circle, 0 otherwise
  - Optional: can extend to more complex geometric regions

Installation:
  1. Clone the repository
  2. Install dependencies: numpy, matplotlib
  3. Run main notebook / script

Usage:
  - Run `python generate_data.py` to create dataset
  - Run `python train_logistic_regression.py` to train the model
  - Run `python visualize.py` to see decision boundaries



