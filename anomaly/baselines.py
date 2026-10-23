"""
baselines.py - the detectors the autoencoder is compared against.

  rule_severity   fixed thresholds of the original project (temperature>75, vibration>20, sound>80)
  Mahalanobis     multivariate Gaussian of healthy data (linear, statistical)
  PCADetector     reconstruction error after projecting on k principal components
                  (= the best LINEAR autoencoder; the fair 'does the neural network add anything?' baseline)
  isolation_forest (optional, needs scikit-learn)
"""
import numpy as np

from anomaly import config


def rule_severity(df):
    """Number of fixed thresholds exceeded per row (0..3); also a crude anomaly score."""
    return ((df["temperature"] > config.TEMP_THRESHOLD).astype(int)
            + (df["vibration"] > config.VIBRATION_THRESHOLD).astype(int)
            + (df["sound_level"] > config.SOUND_THRESHOLD).astype(int)).to_numpy()


class Mahalanobis:
    def fit(self, X):
        self.mu = X.mean(0)
        self.inv = np.linalg.inv(np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1]))
        return self

    def errors(self, X):
        d = X - self.mu
        return np.einsum("ij,jk,ik->i", d, self.inv, d)


class PCADetector:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X):
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.P = vt[: self.k].T
        return self

    def errors(self, X):
        d = X - self.mu
        return ((d - d @ self.P @ self.P.T) ** 2).mean(axis=1)


def isolation_forest(Xtr):
    """Returns an object with .errors(X), or None if scikit-learn is not installed."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return None
    iso = IsolationForest(n_estimators=100, random_state=0).fit(Xtr)

    class _Wrap:
        errors = staticmethod(lambda X: -iso.score_samples(X))
    return _Wrap()
