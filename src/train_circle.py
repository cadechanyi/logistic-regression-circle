import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import fit, predict

def generate_circle_data(n_points=10000, radius=1.0, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, size=(n_points, 2))
    y = (X[:, 0]**2 + X[:, 1]**2 <= radius**2).astype(int)
    return X, y

if __name__ == "__main__":
    # 1) data
    X, y = generate_circle_data(n_points=5000)

    # 2) train
    w, b, history = fit(X, y, lr=0.1, n_iters=2000, verbose=True)

    # 3) predictions
    y_pred = predict(X, w, b)
    acc = np.mean(y_pred == y)
    print(f"Final accuracy: {acc:.4f}")
    print("w:", w, "b:", b)

    # 4) plot dataset + learned line
    inside = y == 1
    outside = y == 0

    plt.figure(figsize=(6, 6))
    plt.scatter(X[inside, 0], X[inside, 1], s=5, label="Inside circle")
    plt.scatter(X[outside, 0], X[outside, 1], s=5, label="Outside circle")

    # true circle
    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), linewidth=2, label="True boundary")

    # Plot learned linear decision boundary using contour
    grid_x = np.linspace(-2, 2, 300)
    grid_y = np.linspace(-2, 2, 300)
    xx, yy = np.meshgrid(grid_x, grid_y)

    Z = w[0] * xx + w[1] * yy + b
    plt.contour(xx, yy, Z, levels=[0], linewidths=2)
    plt.plot([], [], linewidth=2, label="Learned boundary (linear)")

    plt.gca().set_aspect("equal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f"Logistic Regression on Circle Data (acc={acc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../images/linear_boundary.png", dpi=200)
    plt.show()


    # 5) plot loss curve
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig("../images/loss_curve.png", dpi=200)
    plt.show()



