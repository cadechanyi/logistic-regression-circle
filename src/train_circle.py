import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from logistic_regression import fit, predict

def generate_circle_data(n_points=10000, radius=1.0, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, size=(n_points, 2))
    y = (X[:, 0]**2 + X[:, 1]**2 <= radius**2).astype(int)
    r2 = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
    X_mapped = np.hstack([X, r2])
    return X, X_mapped, y

if __name__ == "__main__":
    plt.close("all")

    # 1) data
    X, X_mapped, y = generate_circle_data(n_points=10000)

    # 2) train
    w, b, history = fit(X_mapped, y, lr=0.1, n_iters=2000, verbose=True)

    # 3) predictions
    y_pred = predict(X_mapped, w, b)
    acc = np.mean(y_pred == y)
    print(f"Final accuracy: {acc:.4f}")
    print("w:", w, "b:", b)

    # Diagnosis
    if w[2] != 0:
        print("Approx radius:", np.sqrt(-b / w[2]))

    # 4) plot dataset + learned line
    inside = y == 1
    outside = y == 0

    fig = plt.figure(figsize=(6, 6))

    plt.scatter(X[inside, 0], X[inside, 1], s=5)
    plt.scatter(X[outside, 0], X[outside, 1], s=5)

    # true circle
    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), linewidth=2, linestyle="--")

    # Plot learned linear decision boundary using contour
    grid_x = np.linspace(-2, 2, 400)
    grid_y = np.linspace(-2, 2, 400)
    xx, yy = np.meshgrid(grid_x, grid_y)

    Z = w[0]*xx + w[1]*yy + w[2]*(xx**2 + yy**2) + b
    plt.contour(xx, yy, Z, levels=[0], linewidths=2)

    plt.gca().set_aspect("equal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f"Logistic Regression on Circle Data (acc={acc:.3f})")

    legend_handles = [
        Line2D([], [], linestyle="None", marker="o", markersize=6, label="Inside circle"),
        Line2D([], [], linestyle="None", marker="o", markersize=6, label="Outside circle"),
        Line2D([], [], linestyle="--", linewidth=2, label="True boundary"),
        Line2D([], [], linestyle="-",  linewidth=2, label="Learned boundary (nonlinear)"),
    ]
    plt.legend(handles=legend_handles, loc="lower left")

    plt.tight_layout()
    plt.savefig("../images/nonlinear_boundary.png", dpi=200)
    plt.show()


    # 5) plot loss curve
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss (Nonlinear Feature Mapping")
    plt.tight_layout()
    plt.savefig("../images/loss_curve_nonlinear.png", dpi=200)
    plt.show()




