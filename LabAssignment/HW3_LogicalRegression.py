import numpy as np


class LogisticRegression():
    def __init__(self, n_features, n_classes, max_epoch, lr) -> None:
        self.w = np.zeros((n_features + 1, n_classes))
        self.max_epoch = max_epoch
        self.lr = lr

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        logits = np.dot(X_bias, self.w)
        prob = self.softmax(logits)
        return np.argmax(prob, axis=1)

    def fit(self, X, y):
        N = X.shape[0]
        X_bias = np.c_[np.ones(N), X]

        for epoch in range(self.max_epoch):
            logits = np.dot(X_bias, self.w)
            prob = self.softmax(logits)
            grad_w = np.dot(X_bias.T, (prob - y)) / N
            self.w -= self.lr * grad_w

        self.w = np.vstack([self.w[1:], self.w[0]])
        flattened_weights = self.w.reshape(-1)
        for value in flattened_weights:
            print(f"{value:.3f}")


if __name__ == "__main__":
    N, D, C, E, L = input().split()
    N, D, C, E = map(int, [N, D, C, E])
    L = float(L)

    X = np.array([list(map(float, input().split())) for _ in range(N)])
    y = np.array([list(map(int, input().split())) for _ in range(N)])

    model = LogisticRegression(D, C, E, L)
    model.fit(X, y)
