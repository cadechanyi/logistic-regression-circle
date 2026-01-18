import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def predict_proba(X, w, b):
    z = X @ w + b
    return sigmoid(z)

def predict(X, w, b, threshold=0.5):
    probs = predict_proba(X, w, b)
    return (probs >= threshold).astype(int)

def compute_loss(X, y, w, b, eps=1e-15):
    probs = predict_proba(X, w, b)
    probs = np.clip(probs, eps, 1 -eps)

    loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
    return loss

def compute_gradients(X, y, w, b):
    N = X.shape[0]
    probs = predict_proba(X, w, b)
    error = probs - y

    dw = (X.T @ error) / N
    db = np.sum(error) / N

    return dw, db

def fit(X, y, lr=0.1, n_iters=2000, verbose=False):
    N, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []

    for i in range(n_iters):
        loss = compute_loss(X, y, w, b)
        history.append(loss)

        dw, db = compute_gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db

        if verbose and (i % 200 == 0):
            print(f"iter {i:4d} | loss {loss:.6f}")

    return w, b, history

if __name__ == "__main__":
    X_test = np.array([[0.0, 0.0],
                       [1.0, 1.0],
                       [-1.0, -1.0]])

    w = np.array([1.0, 1.0])
    b = 0.0

    probs = predict_proba(X_test, w, b)
    preds = predict(X_test, w, b)
    print("Probabilities:", probs)
    print("Predictions:", preds)

    y_test = np.array([0, 1, 0])
    print("Loss:", compute_loss(X_test, y_test, w, b))

    dw, db = compute_gradients(X_test, y_test, w, b)
    print("dw:", dw)
    print("db:", db)

