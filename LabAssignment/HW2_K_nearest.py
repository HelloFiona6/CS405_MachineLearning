import numpy as np


class KNN:
    def __init__(self, k=3, metric='L2'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = self._compute_distances(x)
        neighbors_indices = np.argsort(distances, kind='stable')[:self.k]
        neighbor_labels = self.y_train[neighbors_indices]

        count = np.bincount(neighbor_labels)
        most_common_label = np.argmax(count)

        most_common_labels = np.flatnonzero(count == count[most_common_label])
        return min(most_common_labels)

    def _compute_distances(self, x):
        if self.metric == 'L1':
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.metric == 'L2':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'L-inf':
            return np.max(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError("Unknown metric")


def load_data():
    N, M, D = map(int, input().strip().split())
    X_train, y_train = [], []
    for _ in range(N):
        data = list(map(float, input().strip().split()))
        X_train.append(data[:-1])
        y_train.append(int(data[-1]))

    X_val, y_val = [], []
    for _ in range(M):
        data = list(map(float, input().strip().split()))
        X_val.append(data[:-1])
        y_val.append(int(data[-1]))

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)


def evaluate_model(X_train, y_train, X_val, y_val, k, metric):
    knn = KNN(k=k, metric=metric)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_val)
    accuracy = np.mean(predictions == y_val)
    return accuracy


def main():
    X_train, y_train, X_val, y_val = load_data()
    best_accuracy = 0
    best_params = []

    for k in range(1, 6):
        for metric in ['L1', 'L2', 'L-inf']:
            accuracy = evaluate_model(X_train, y_train, X_val, y_val, k, metric)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = [(k, metric)]
            elif accuracy == best_accuracy:
                best_params.append((k, metric))

    metric_order = {'L1': 0, 'L2': 1, 'L-inf': 2}
    best_params = sorted(best_params, key=lambda x: (x[0], metric_order[x[1]]))

    for k, metric in best_params:
        print(k, metric)


if __name__ == "__main__":
    main()
