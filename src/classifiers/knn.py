import numpy as np


class StandardScalerScratch:

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        self.std[self.std < 1e-10] = 1.0

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNNFromScratch:

    def __init__(self, k=1, distance="euclidean"):
        self.k = k
        self.distance = distance

        if k < 1:
            raise ValueError("k must be positive")

        if distance not in ["euclidean", "cosine"]:
            raise ValueError("Unsupported distance")

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)

        if self.X_train.ndim != 2:
            raise ValueError("X must be a 2D matrix")

        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same length")

        if self.k > len(self.X_train):
            raise ValueError("k exceeds training size")

        return self

    def calculate_distances(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))

        if self.distance == "euclidean":

            X_norm = np.sum(X ** 2, axis=1, keepdims=True)

            train_norm = np.sum(
                self.X_train ** 2,
                axis=1
            )

            distances_squared = (
                X_norm
                + train_norm
                - 2 * X @ self.X_train.T
            )

            distances_squared = np.maximum(distances_squared, 0)

            return np.sqrt(distances_squared)

        X_norm = np.linalg.norm(X, axis=1, keepdims=True)

        train_norm = np.linalg.norm(
            self.X_train,
            axis=1,
            keepdims=True
        ).T

        denominator = np.maximum(
            X_norm * train_norm,
            1e-12
        )

        similarity = (
            X @ self.X_train.T
        ) / denominator

        return 1 - np.clip(similarity, -1, 1)

    def predict(self, X, batch_size=128):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))

        predictions = []

        for start in range(0, len(X), batch_size):

            batch = X[start:start + batch_size]

            distances = self.calculate_distances(batch)

            for row in distances:

                nearest_indices = np.argsort(
                    row,
                    kind="stable"
                )[:self.k]

                nearest_labels = self.y_train[nearest_indices]

                classes, counts = np.unique(
                    nearest_labels,
                    return_counts=True
                )

                max_votes = counts.max()

                tied_classes = set(classes[counts == max_votes])

                prediction = next(
                    label
                    for label in nearest_labels
                    if label in tied_classes
                )

                predictions.append(prediction)

        return np.asarray(predictions)
