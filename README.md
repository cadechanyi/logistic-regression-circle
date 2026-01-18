# Logistic Regression on Synthetic Circle Dataset

This project implements **logistic regression from first principles** to classify points as inside or outside a circle. 
The model demonstrates the **limitations of linear decision boundaries** and how **nonlinear feature mapping** enables logistic regression to model nonlinear geometry - a core idea in **Scientific Machine Learning**  


Overview/Motivation:
  - Logistic regression is a foundational machine learning algorithm for binary classification. This project explores it in a **controlled, scientific** context:
    - Dataset: synthetic 2D points labeled by their position relative to a unit circle.
    - Objective: train a logistic regression model from scratch and analyze its performance.
    - Extension: improve classification with nonlinear feature mapping.

Dataset:
  - **Synthetic 2D data points** sampled uniformly in [-2,2] x [-2, 2]
  - Labels: 1 if x^2 + y^2 <= 1 (inside the unit circle), 0 otherwise
  - Dataset is generated programmatically for reproducibility

Methodology:
  - Baseline: Linear Logistic Regression
    - Impleted logistic regression **from scratch** using NumPy
    - Optimized using **batch gradient descent**
    - Model uses a **linear decision boundary** in (x, y)
    - This baseline Converges but cannot represent the circular boundary, illustraing a **capacity limitation** of linear models.
  - Extension: Nonlinear Feature Mapping
    - To overcome this limitation, the input space is extended via the feature map:
      ϕ(x,y)=(x,y,x2+y2)
    - This allows logistic regression to learn a **circular decision boundary** while keeping the model linear in feature space.
   
Results:
  - Linear Model (Baseline)
    - Training converges but plateaus due to model expressiveness limits
    - Accuracy ≈ **80%**
    - Learned boundary does not intersect the region of interest
     <img width="1200" height="1200" alt="linear_boundary" src="https://github.com/user-attachments/assets/bc321585-a0a9-4c81-8f9d-b50749e5ca96" />
    <img width="1280" height="960" alt="loss_curve" src="https://github.com/user-attachments/assets/57de8c91-4e7b-4e1b-bba7-c686e2bf9384" />
  - Nonlinear Feature Mapping
    - Same optimizer and loss function
    - Improved expressiveness via feature engineering
    - Accuracy ≈ **99%**
    - Learned decision boundary closely matches the true circle 
    <img width="1200" height="1200" alt="nonlinear_boundary" src="https://github.com/user-attachments/assets/8cc68152-ba30-487e-9c8d-e6e19f997372" />
    <img width="1280" height="960" alt="loss_curve_nonlinear" src="https://github.com/user-attachments/assets/720384c4-062f-46c4-b080-2d5aba760a3d" />

  

  
Implementation Details:
  - **No ML libraries used** (Numpy + Matplotlib only)
  - Numerically stable sigmoid
  - Binary cross-entropy loss
  - Explicit gradient computation
  - Fully vectorized operations

How to Run:
  - pip install numpy matplotlib
  - python src/train_circle.py





