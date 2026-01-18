import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_points=10000, radius=1.0, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, size=(n_points, 2))
    y = (X[:, 0]**2 + X[:, 1]**2 <= radius**2).astype(int)
    return X, y

def plot_data(X, y, radius=1.0):
    inside = y == 1
    outside = y == 0

    plt.figure(figsize=(6,6))
    plt.scatter(X[inside, 0], X[inside, 1], s=8, alpha=0.6, label="Inside circle")
    plt.scatter(X[outside, 0], X[outside, 1], s=8, alpha=0.6, label="Outside circle")

    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(radius * np.cos(theta), radius * np.sin(theta),
             linewidth=2, label="True boundary")

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Circle Classification Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../images/dataset.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    X, y = generate_data()
    plot_data(X, y)
