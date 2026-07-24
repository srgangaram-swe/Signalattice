"""Leakage-safe panel sequences and an optional compact temporal network.

The sequence construction utilities in this module do not require PyTorch.  A
sequence always ends at its labelled observation and contains rows from exactly
one ticker, ordered by time.  This makes the causal contract inspectable through
the returned source-row metadata instead of relying on the input frame's row
order.

PyTorch is an optional dependency (``pip install '.[torch]'``).  The temporal
estimator deliberately accepts only pre-built, three-dimensional arrays with
shape ``[sample, time, feature]``.  Panel keys belong in
:func:`build_panel_sequences`; silently windowing a two-dimensional panel would
mix assets and leak information across samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quant_platform.config import ModelConfig
from quant_platform.logging_utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

logger = get_logger(__name__)


def _require_torch() -> Any:
    """Import PyTorch lazily and explain how to enable the optional model."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "The temporal model requires PyTorch. Install with `pip install '.[torch]'`."
        ) from exc
    return torch


@dataclass(frozen=True)
class PanelSequenceMetadata:
    """Keys and provenance aligned one-for-one with a sequence batch.

    ``history_row_indices`` contains positions in the original input arrays.
    Left-padding positions use ``-1`` and are ``False`` in ``valid_mask``.
    These fields make ticker isolation and point-in-time causality auditable.
    """

    dates: NDArray[np.datetime64]
    tickers: NDArray[np.str_]
    row_indices: NDArray[np.int64]
    history_row_indices: NDArray[np.int64]
    valid_mask: NDArray[np.bool_]
    lengths: NDArray[np.int64]


@dataclass(frozen=True)
class PanelSequences:
    """Causal model inputs, optional targets, and their aligned metadata."""

    X: NDArray[np.float32]
    y: NDArray[Any] | None
    metadata: PanelSequenceMetadata

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _normalise_panel_keys(
    dates: ArrayLike,
    tickers: ArrayLike,
    *,
    n_rows: int,
) -> tuple[NDArray[np.datetime64], NDArray[np.str_], NDArray[np.int64]]:
    raw_dates = np.asarray(dates)
    raw_tickers = np.asarray(tickers, dtype=object)
    if raw_dates.ndim != 1 or raw_tickers.ndim != 1:
        raise ValueError("dates and tickers must each be one-dimensional")
    if len(raw_dates) != n_rows or len(raw_tickers) != n_rows:
        raise ValueError("features, dates, and tickers must contain the same number of rows")

    parsed_dates = pd.to_datetime(pd.Series(raw_dates), errors="raise", utc=True)
    if parsed_dates.isna().any():
        raise ValueError("dates cannot contain missing values")
    # Normalise to UTC-naive datetime64[ns] so comparisons are unambiguous.
    date_values = parsed_dates.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")

    if pd.isna(raw_tickers).any():
        raise ValueError("tickers cannot contain missing values")
    if any(not isinstance(ticker, str) or not ticker.strip() for ticker in raw_tickers):
        raise ValueError("tickers must be non-empty strings")
    ticker_values = np.asarray([ticker.strip() for ticker in raw_tickers], dtype=str)

    key_frame = pd.DataFrame({"date": date_values, "ticker": ticker_values})
    duplicate = key_frame.duplicated(["date", "ticker"], keep=False)
    if duplicate.any():
        example = key_frame.loc[duplicate, ["date", "ticker"]].iloc[0]
        raise ValueError(
            "date/ticker keys must be unique; duplicate key "
            f"({example['date']}, {example['ticker']!r})"
        )

    # A canonical date-first output order makes chronological validation tails
    # well-defined even when the source panel is interleaved or unsorted.
    ordered_rows = (
        key_frame.assign(_row=np.arange(n_rows, dtype=np.int64))
        .sort_values(["date", "ticker"], kind="mergesort")["_row"]
        .to_numpy(dtype=np.int64)
    )
    return date_values, ticker_values, ordered_rows


def build_panel_sequences(
    features: ArrayLike,
    dates: ArrayLike,
    tickers: ArrayLike,
    *,
    targets: ArrayLike | None = None,
    sequence_length: int = 20,
    min_history: int = 1,
    pad_value: float = 0.0,
) -> PanelSequences:
    """Construct right-aligned, causal sequences from a keyed asset panel.

    Parameters
    ----------
    features:
        Numeric matrix with shape ``[row, feature]``.
    dates, tickers:
        Point-in-time keys aligned with ``features``.  A ``(date, ticker)`` key
        must be unique.  Input rows need not be sorted or ticker-contiguous.
    targets:
        Optional labels aligned with the source rows.  One- and multi-output
        targets are retained without coercing their dtype.
    sequence_length:
        Maximum history, including the sample's current row.
    min_history:
        Exclude samples with fewer ticker observations than this.  Use
        ``sequence_length`` to retain only complete windows.
    pad_value:
        Value used for left-padding incomplete histories.

    Returns
    -------
    PanelSequences
        Samples are in canonical ``(date, ticker)`` order.  ``row_indices`` map
        them back to the original arrays, and ``history_row_indices`` provide
        complete provenance for every non-padding timestep.
    """
    values = np.asarray(features)
    if values.ndim != 2:
        raise ValueError(
            f"features must have shape [row, feature]; received array with shape {values.shape}"
        )
    n_rows, n_features = values.shape
    if n_rows == 0 or n_features == 0:
        raise ValueError("features must contain at least one row and one feature")
    if not isinstance(sequence_length, (int, np.integer)) or isinstance(sequence_length, bool):
        raise ValueError("sequence_length must be a positive integer")
    if sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if not isinstance(min_history, (int, np.integer)) or isinstance(min_history, bool):
        raise ValueError("min_history must be between 1 and sequence_length")
    if not 1 <= min_history <= sequence_length:
        raise ValueError("min_history must be between 1 and sequence_length")
    if not np.isfinite(pad_value):
        raise ValueError("pad_value must be finite")
    try:
        numeric_values = values.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be numeric") from exc

    date_values, ticker_values, ordered_rows = _normalise_panel_keys(dates, tickers, n_rows=n_rows)

    target_values: NDArray[Any] | None = None
    if targets is not None:
        target_values = np.asarray(targets)
        if target_values.ndim == 0 or target_values.shape[0] != n_rows:
            raise ValueError("targets must have the same first dimension as features")

    history_by_row: dict[int, NDArray[np.int64]] = {}
    lengths_by_row = np.empty(n_rows, dtype=np.int64)
    for ticker in np.unique(ticker_values):
        ticker_rows = np.flatnonzero(ticker_values == ticker)
        ticker_rows = ticker_rows[np.argsort(date_values[ticker_rows], kind="stable")]
        for position, row_index in enumerate(ticker_rows):
            start = max(0, position + 1 - sequence_length)
            history = ticker_rows[start : position + 1].astype(np.int64, copy=False)
            history_by_row[int(row_index)] = history
            lengths_by_row[row_index] = len(history)

    output_rows = ordered_rows[lengths_by_row[ordered_rows] >= min_history]
    n_samples = len(output_rows)
    sequences = np.full((n_samples, sequence_length, n_features), pad_value, dtype=np.float32)
    history_indices = np.full((n_samples, sequence_length), -1, dtype=np.int64)
    valid_mask = np.zeros((n_samples, sequence_length), dtype=bool)
    lengths = lengths_by_row[output_rows]

    for sample_index, row_index in enumerate(output_rows):
        history = history_by_row[int(row_index)]
        start = sequence_length - len(history)
        sequences[sample_index, start:] = numeric_values[history]
        history_indices[sample_index, start:] = history
        valid_mask[sample_index, start:] = True

    metadata = PanelSequenceMetadata(
        dates=date_values[output_rows],
        tickers=ticker_values[output_rows],
        row_indices=output_rows,
        history_row_indices=history_indices,
        valid_mask=valid_mask,
        lengths=lengths,
    )
    output_targets = None if target_values is None else target_values[output_rows]
    return PanelSequences(X=sequences, y=output_targets, metadata=metadata)


# A descriptive alternative for callers that prefer construction terminology.
construct_panel_sequences = build_panel_sequences


class TemporalConvEstimator:
    """Compact causal dilated TCN for binary classification or regression.

    This is intentionally a temporal estimator, not a scikit-learn-compatible
    two-dimensional estimator.  ``fit`` and inference methods require arrays of
    shape ``[sample, time, feature]``.  Samples are assumed to be in
    chronological order unless ``sample_dates`` is supplied to ``fit``.
    """

    def __init__(
        self,
        *,
        task: str = "classification",
        sequence_length: int | None = None,
        channels: int = 32,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 256,
        validation_fraction: float = 0.2,
        patience: int = 5,
        min_delta: float = 1e-5,
        gradient_clip_norm: float = 1.0,
        inference_batch_size: int = 1024,
        seed: int = 42,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        integer_parameters = {
            "channels": channels,
            "num_blocks": num_blocks,
            "kernel_size": kernel_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "inference_batch_size": inference_batch_size,
        }
        if sequence_length is not None:
            integer_parameters["sequence_length"] = sequence_length
        if any(
            not isinstance(value, (int, np.integer)) or isinstance(value, bool)
            for value in integer_parameters.values()
        ):
            raise ValueError(
                "sequence lengths, architecture sizes, and batch sizes must be integers"
            )
        if sequence_length is not None and sequence_length < 1:
            raise ValueError("sequence_length must be positive when provided")
        if channels < 1 or num_blocks < 1:
            raise ValueError("channels and num_blocks must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if lr <= 0.0 or weight_decay < 0.0:
            raise ValueError("lr must be positive and weight_decay non-negative")
        if epochs < 1 or batch_size < 1 or inference_batch_size < 1:
            raise ValueError("epochs and batch sizes must be positive")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if patience < 1 or min_delta < 0.0 or gradient_clip_norm <= 0.0:
            raise ValueError("patience and gradient_clip_norm must be positive")

        self.task = task
        self.sequence_length = sequence_length
        self.channels = channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_clip_norm = gradient_clip_norm
        self.inference_batch_size = inference_batch_size
        self.seed = seed

        self._model: Any | None = None
        self._n_features: int | None = None
        self.classes_: NDArray[Any] | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
        self.best_epoch_: int | None = None
        self.n_epochs_: int = 0
        self.training_indices_: NDArray[np.int64] | None = None
        self.validation_indices_: NDArray[np.int64] | None = None

    def _validate_X(self, X: ArrayLike) -> NDArray[np.float32]:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                "TemporalConvEstimator requires X with shape [sample, time, feature]; "
                "build keyed panel inputs with build_panel_sequences(...)"
            )
        if not all(size > 0 for size in values.shape):
            raise ValueError("X dimensions must be non-empty")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values")
        if self.sequence_length is not None and values.shape[1] != self.sequence_length:
            raise ValueError(
                f"expected sequences of length {self.sequence_length}, got {values.shape[1]}"
            )
        if self._n_features is not None and values.shape[2] != self._n_features:
            raise ValueError(f"expected {self._n_features} features, got {values.shape[2]}")
        return values

    def _make_model(self, n_features: int) -> Any:
        torch = _require_torch()
        import torch.nn as nn
        import torch.nn.functional as functional

        torch.manual_seed(self.seed)

        class _CausalResidualBlock(nn.Module):
            def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
                super().__init__()
                self.left_padding = dilation * (kernel_size - 1)
                self.conv1 = nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation)
                self.conv2 = nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation)
                self.norm1 = nn.LayerNorm(width)
                self.norm2 = nn.LayerNorm(width)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)

            def _causal_conv(self, values: Any, convolution: Any) -> Any:
                return convolution(functional.pad(values, (self.left_padding, 0)))

            def forward(self, values: Any) -> Any:
                residual = values
                output = self._causal_conv(values, self.conv1)
                output = self.norm1(output.transpose(1, 2)).transpose(1, 2)
                output = self.dropout(self.activation(output))
                output = self._causal_conv(output, self.conv2)
                output = self.norm2(output.transpose(1, 2)).transpose(1, 2)
                return self.activation(residual + self.dropout(output))

        class _CausalTemporalNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Conv1d(n_features, self_width, kernel_size=1)
                self.blocks = nn.ModuleList(
                    _CausalResidualBlock(
                        self_width,
                        kernel_size=self_kernel,
                        dilation=2**level,
                        dropout=self_dropout,
                    )
                    for level in range(self_levels)
                )
                self.output_norm = nn.LayerNorm(self_width)
                self.head = nn.Linear(self_width, 1)

            def forward(self, values: Any) -> Any:
                output = self.input_projection(values.transpose(1, 2))
                for block in self.blocks:
                    output = block(output)
                final_state = self.output_norm(output[:, :, -1])
                return self.head(final_state).squeeze(-1)

        self_width = self.channels
        self_kernel = self.kernel_size
        self_dropout = self.dropout
        self_levels = self.num_blocks
        return _CausalTemporalNet()

    @staticmethod
    def _date_order(sample_dates: ArrayLike, n_samples: int) -> NDArray[np.int64]:
        values = np.asarray(sample_dates)
        if values.ndim != 1 or len(values) != n_samples:
            raise ValueError("sample_dates must be one-dimensional and aligned with X")
        parsed = pd.to_datetime(pd.Series(values), errors="raise", utc=True)
        if parsed.isna().any():
            raise ValueError("sample_dates cannot contain missing values")
        date_ns = parsed.astype("int64").to_numpy()
        return np.argsort(date_ns, kind="stable").astype(np.int64, copy=False)

    def _split_indices(
        self,
        n_samples: int,
        sample_dates: ArrayLike | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        order = (
            np.arange(n_samples, dtype=np.int64)
            if sample_dates is None
            else self._date_order(sample_dates, n_samples)
        )
        if self.validation_fraction == 0.0 or n_samples < 2:
            return order, np.empty(0, dtype=np.int64)

        if sample_dates is not None:
            ordered_dates = pd.to_datetime(
                pd.Series(np.asarray(sample_dates)[order]), errors="raise", utc=True
            ).astype("int64")
            unique_dates = np.unique(ordered_dates.to_numpy())
            if len(unique_dates) >= 2:
                n_validation_dates = min(
                    len(unique_dates) - 1,
                    max(1, int(np.ceil(len(unique_dates) * self.validation_fraction))),
                )
                boundary = unique_dates[-n_validation_dates]
                validation_mask = ordered_dates.to_numpy() >= boundary
                return order[~validation_mask], order[validation_mask]

        n_validation = min(
            n_samples - 1,
            max(1, int(np.ceil(n_samples * self.validation_fraction))),
        )
        return order[:-n_validation], order[-n_validation:]

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_dates: ArrayLike | None = None,
    ) -> TemporalConvEstimator:
        """Fit on causal sequences with an optional date-grouped validation tail."""
        values = self._validate_X(X)
        targets = np.asarray(y)
        if targets.ndim != 1 or len(targets) != len(values):
            raise ValueError("y must be one-dimensional and aligned with X")
        if self.task == "classification":
            if pd.isna(targets).any():
                raise ValueError("classification targets cannot contain missing values")
            classes = np.unique(targets)
            if len(classes) != 2:
                raise ValueError("classification requires exactly two target classes")
            self.classes_ = classes
            numeric_targets = (targets == classes[1]).astype(np.float32)
        else:
            self.classes_ = None
            try:
                numeric_targets = targets.astype(np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("regression targets must be numeric") from exc
            if not np.isfinite(numeric_targets).all():
                raise ValueError("regression targets must contain only finite values")

        train_indices, validation_indices = self._split_indices(len(values), sample_dates)
        self.training_indices_ = train_indices.copy()
        self.validation_indices_ = validation_indices.copy()
        self._n_features = values.shape[2]

        torch = _require_torch()
        import torch.nn as nn

        model = self._make_model(self._n_features).to("cpu")
        self._model = model

        # CPU-only execution and deterministic algorithms prioritise reproducible
        # research evidence over environment-dependent accelerator throughput.
        torch.manual_seed(self.seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            torch.use_deterministic_algorithms(True)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        feature_tensor = torch.from_numpy(values)
        target_tensor = torch.from_numpy(numeric_targets)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_function = nn.BCEWithLogitsLoss() if self.task == "classification" else nn.MSELoss()

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        self.history_ = {"train_loss": [], "validation_loss": []}

        for epoch in range(self.epochs):
            model.train()
            shuffled = train_indices[
                torch.randperm(len(train_indices), generator=generator).numpy()
            ]
            total_loss = 0.0
            for start in range(0, len(shuffled), self.batch_size):
                batch_indices = shuffled[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                predictions = model(feature_tensor[batch_indices])
                loss = loss_function(predictions, target_tensor[batch_indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.gradient_clip_norm)
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch_indices)

            train_loss = total_loss / len(train_indices)
            validation_loss = self._evaluate_loss(
                feature_tensor, target_tensor, validation_indices, loss_function
            )
            monitored_loss = validation_loss if len(validation_indices) else train_loss
            self.history_["train_loss"].append(train_loss)
            self.history_["validation_loss"].append(validation_loss)

            if monitored_loss < best_loss - self.min_delta:
                best_loss = monitored_loss
                best_state = {
                    key: tensor.detach().clone() for key, tensor in model.state_dict().items()
                }
                self.best_epoch_ = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            logger.debug(
                "TCN epoch %d/%d train_loss=%.6f validation_loss=%s",
                epoch + 1,
                self.epochs,
                train_loss,
                f"{validation_loss:.6f}" if np.isfinite(validation_loss) else "n/a",
            )
            if epochs_without_improvement >= self.patience:
                break

        self.n_epochs_ = len(self.history_["train_loss"])
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return self

    def _evaluate_loss(
        self,
        features: Any,
        targets: Any,
        indices: NDArray[np.int64],
        loss_function: Any,
    ) -> float:
        if not len(indices):
            return float("nan")
        torch = _require_torch()
        model = self._model
        if model is None:
            raise RuntimeError("TemporalConvEstimator is not fitted")
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(indices), self.inference_batch_size):
                batch_indices = indices[start : start + self.inference_batch_size]
                loss = loss_function(model(features[batch_indices]), targets[batch_indices])
                total_loss += float(loss) * len(batch_indices)
        return total_loss / len(indices)

    def _check_fitted(self) -> None:
        if self._model is None or self._n_features is None:
            raise RuntimeError("TemporalConvEstimator is not fitted")

    def __getstate__(self) -> dict[str, Any]:
        """Serialise weights without pickling method-local PyTorch classes."""
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        state["_model_state"] = (
            None
            if model is None
            else {
                name: tensor.detach().cpu().numpy().copy()
                for name, tensor in model.state_dict().items()
            }
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the network architecture and restore persisted CPU weights."""
        model_state = state.pop("_model_state", None)
        self.__dict__.update(state)
        self._model = None
        if model_state is not None:
            if self._n_features is None:
                raise ValueError("persisted temporal model is missing its feature dimension")
            torch = _require_torch()
            self._model = self._make_model(self._n_features).to("cpu")
            self._model.load_state_dict(
                {name: torch.from_numpy(values) for name, values in model_state.items()}
            )
            self._model.eval()

    def _raw_predict(self, X: ArrayLike) -> NDArray[np.float32]:
        self._check_fitted()
        values = self._validate_X(X)
        torch = _require_torch()
        tensor = torch.from_numpy(values)
        outputs: list[NDArray[np.float32]] = []
        model = self._model
        if model is None:
            raise RuntimeError("TemporalConvEstimator is not fitted")
        model.eval()
        with torch.no_grad():
            for start in range(0, len(tensor), self.inference_batch_size):
                output = model(tensor[start : start + self.inference_batch_size])
                outputs.append(output.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float32]:
        """Return negative/positive class probabilities for binary classification."""
        if self.task != "classification":
            raise AttributeError("predict_proba is available only for classification")
        logits = self._raw_predict(X).astype(np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        return np.column_stack((1.0 - probabilities, probabilities)).astype(np.float32)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        if self.task == "classification":
            self._check_fitted()
            positive = self.predict_proba(X)[:, 1] >= 0.5
            classes = self.classes_
            if classes is None:
                raise RuntimeError("fitted classification estimator is missing classes")
            return np.asarray(np.where(positive, classes[1], classes[0]))
        return self._raw_predict(X)

    def input_attributions(
        self,
        X: ArrayLike,
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.float32]:
        """Return gradient-times-input local attributions with the shape of ``X``.

        Classification attributes the positive-class probability; regression
        attributes the raw forecast.  This lightweight diagnostic is suitable
        for ranking time/feature sensitivity, not causal interpretation.
        """
        self._check_fitted()
        values = self._validate_X(X)
        torch = _require_torch()
        attribution_batches: list[NDArray[np.float32]] = []
        effective_batch_size = batch_size or self.inference_batch_size
        if effective_batch_size < 1:
            raise ValueError("batch_size must be positive")

        model = self._model
        if model is None:
            raise RuntimeError("TemporalConvEstimator is not fitted")
        model.eval()
        for start in range(0, len(values), effective_batch_size):
            batch = torch.from_numpy(values[start : start + effective_batch_size]).requires_grad_(
                True
            )
            output = model(batch)
            if self.task == "classification":
                output = torch.sigmoid(output)
            gradient = torch.autograd.grad(output.sum(), batch)[0]
            attribution_batches.append((gradient * batch).detach().cpu().numpy())
        return np.concatenate(attribution_batches).astype(np.float32, copy=False)


def _translate_legacy_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translate old LSTM names while keeping sequence construction explicit."""
    translated = dict(params)
    aliases = {
        "seq_len": "sequence_length",
        "hidden_size": "channels",
        "num_layers": "num_blocks",
    }
    for old_name, new_name in aliases.items():
        if old_name in translated:
            if new_name in translated:
                raise ValueError(f"provide only one of {old_name!r} and {new_name!r}")
            translated[new_name] = translated.pop(old_name)
    return translated


def build_temporal_estimator(
    config: ModelConfig,
    *,
    seed: int = 42,
) -> TemporalConvEstimator:
    """Build a causal temporal estimator from model configuration parameters."""
    _require_torch()
    params = _translate_legacy_params(dict(config.params))
    return TemporalConvEstimator(task=config.task, seed=seed, **params)


def build_lstm_estimator(config: ModelConfig, *, seed: int = 42) -> TemporalConvEstimator:
    """Compatibility builder for the existing ``model.type='lstm'`` factory key."""
    return build_temporal_estimator(config, seed=seed)


# Preserve the old import without preserving its unsafe row-windowing behaviour.
LSTMEstimator = TemporalConvEstimator
