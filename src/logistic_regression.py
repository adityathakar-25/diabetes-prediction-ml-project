import numpy as np

def sigmoid(z):
    z = np.asarray(z)
    out = np.zeros_like(z, dtype=float)

    pos_mask = z >= 0
    neg_mask = ~pos_mask

    out[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    out[neg_mask] = exp_z / (1 + exp_z)

    return out




def binary_cross_entropy(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))




def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    losses = []

    for _ in range(epochs):
        z = np.dot(X, w) + b
        p = sigmoid(z)

        loss = binary_cross_entropy(y, p)
        losses.append(loss)

        dw = (1 / n_samples) * np.dot(X.T, (p - y))
        db = (1 / n_samples) * np.sum(p - y)

        w -= lr * dw
        b -= lr * db

    return w, b, losses




def train_logistic_regression_reg(X, y, lr=0.1, epochs=1000, lam=0.0):
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    losses = []

    for epoch in range(epochs):
        z = np.dot(X, w) + b
        p = sigmoid(z)

        loss = binary_cross_entropy(y, p)

        loss += (lam / (2 * n_samples)) * np.sum(w ** 2)
        losses.append(loss)

        dw = (1 / n_samples) * np.dot(X.T, (p - y)) + (lam / n_samples) * w
        db = (1 / n_samples) * np.sum(p - y)

        w -= lr * dw
        b -= lr * db

    return w, b, losses





def predict_proba(X, w, b):
    return sigmoid(np.dot(X, w) + b)


def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)


def confusion_matrix(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def recall(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) != 0 else 0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0