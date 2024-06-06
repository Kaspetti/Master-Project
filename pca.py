import numpy as np


class PCAResult:
    D: np.ndarray
    V: np.ndarray
    B: np.ndarray
    C: np.ndarray

    def __init__(self, D, V, B, C, u):
        self.D = D
        self.V = V
        self.B = B
        self.C = C
        self.u = u

    def fit_transform(self, n_components: int):
        return self.B @ self.V[:, :n_components]

    def reconstruct(self, scores, n_components: int):
        return scores @ self.V[:, :n_components].T + self.u.T


def PCA_1(X: np.array) -> PCAResult:
    u = X.mean(axis=0).reshape(-1, 1)
    B = X - u.T

    C = (1 / (len(X) - 1)) * (B.conj().T @ B)

    D, V = np.linalg.eigh(C)

    sorted_indices = D.argsort()[::-1]

    D = D[sorted_indices]
    V = V[sorted_indices, :]

    return PCAResult(D, V, B, C, u)
