"""Optional PyTorch LSTM sequence model with a scikit-learn-style interface.

This is an *optional* component (requires ``pip install '.[torch]'``). It is
provided as a clean, self-contained example of a sequence model that plugs into
the same train/predict interface as the tree/linear models.

The wrapper builds overlapping look-back windows of length ``seq_len`` from the
(time-ordered) feature rows and learns to predict the label aligned to the end
of each window. It is therefore best used on a single asset's ordered feature
series; for the default cross-sectional panel the tree models are recommended.
"""

from __future__ import annotations

import numpy as np

from quant_platform.config import ModelConfig
from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "The LSTM model requires PyTorch. Install with `pip install '.[torch]'`."
        ) from exc
    import torch

    return torch


class LSTMEstimator:
    """Minimal sklearn-style LSTM classifier/regressor.

    Parameters mirror typical sklearn estimators so it can be swapped into the
    walk-forward harness. Implements ``fit``, ``predict`` and (for
    classification) ``predict_proba``.
    """

    def __init__(
        self,
        *,
        task: str = "classification",
        seq_len: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.task = task
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._model = None
        self._n_features: int | None = None
        self.classes_ = np.array([0.0, 1.0])

    # -- internal helpers ----------------------------------------------------
    def _build_windows(self, X: np.ndarray):
        torch = _require_torch()
        n, f = X.shape
        if n < self.seq_len:
            # Not enough rows; pad at the front by repeating the first row.
            pad = np.repeat(X[:1], self.seq_len - n, axis=0)
            X = np.vstack([pad, X])
            n = X.shape[0]
        windows = np.stack([X[i - self.seq_len : i] for i in range(self.seq_len, n + 1)], axis=0)
        return torch.tensor(windows, dtype=torch.float32)

    def _make_model(self, n_features: int):
        torch = _require_torch()
        import torch.nn as nn

        torch.manual_seed(self.seed)

        class _Net(nn.Module):
            def __init__(self, n_in, hidden, layers, dropout, n_out):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_in,
                    hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden, n_out)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        n_out = 1
        return _Net(n_features, self.hidden_size, self.num_layers, self.dropout, n_out)

    # -- sklearn-style API ---------------------------------------------------
    def fit(self, X, y):
        torch = _require_torch()
        import torch.nn as nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._n_features = X.shape[1]
        self._model = self._make_model(self._n_features)

        windows = self._build_windows(X)  # (n_windows, seq_len, f)
        targets = torch.tensor(y[-windows.shape[0] :], dtype=torch.float32).view(-1, 1)

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss() if self.task == "classification" else nn.MSELoss()

        self._model.train()
        n = windows.shape[0]
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                pred = self._model(windows[idx])
                loss = loss_fn(pred, targets[idx])
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            if (epoch + 1) % max(1, self.epochs // 4) == 0:
                logger.debug("LSTM epoch %d/%d loss=%.5f", epoch + 1, self.epochs, total / n)
        return self

    def _raw_predict(self, X) -> np.ndarray:
        torch = _require_torch()
        X = np.asarray(X, dtype=np.float32)
        windows = self._build_windows(X)
        self._model.eval()
        with torch.no_grad():
            out = self._model(windows).view(-1).numpy()
        # Align: first (seq_len-1) rows share the first available prediction.
        pad = np.repeat(out[:1], len(X) - len(out)) if len(X) > len(out) else np.array([])
        return np.concatenate([pad, out])[: len(X)]

    def predict_proba(self, X) -> np.ndarray:
        logits = self._raw_predict(X)
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        if self.task == "classification":
            return (self.predict_proba(X)[:, 1] > 0.5).astype(float)
        return self._raw_predict(X)


def build_lstm_estimator(config: ModelConfig, *, seed: int = 42) -> LSTMEstimator:
    """Build an :class:`LSTMEstimator` from model config params."""
    _require_torch()
    params = dict(config.params)
    return LSTMEstimator(task=config.task, seed=seed, **params)
