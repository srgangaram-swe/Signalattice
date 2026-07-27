"""Causal dynamic asset graphs and a temporal message-passing baseline (SA #15).

Cross-asset dependence is usually captured by a static correlation matrix, which
assumes the dependence structure is the same in 2015 and in a crash. It is not.
This module builds a *dynamic* graph — re-estimated on a rolling trailing window
— and passes messages over it.

Two leakage hazards are specific to graphs, and both are closed by construction
rather than by discipline:

**Time leakage.** Every edge at date ``t`` is estimated from a window ending at
``t-1`` and inclusive of nothing later. The builder never sees ``t``.

**Graph leakage.** This is the subtle one. Message passing lets information flow
between assets, so if an edge were built using the *target* asset's own future,
a neighbour could carry that future back. Edges here are functions of lagged
returns only, so the target's future cannot enter through any path length.

The message-passing model is deliberately compact — one or two rounds of
neighbour averaging followed by a linear readout — because the question is
whether cross-asset structure carries incremental information at all, not
whether a large graph network can be tuned to a benchmark. A model that needs
depth to show an effect on a few hundred assets is fitting noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Refusal thresholds, not tuning knobs.
MAX_NODES = 200
MAX_ROUNDS = 3
MIN_GRAPH_WINDOW = 20

EdgeMethod = Literal["correlation", "partial_correlation", "lagged_causality"]


class AssetGraphError(ValueError):
    """Raised when a graph request is unusable or out of bounds."""


@dataclass(frozen=True)
class DynamicGraph:
    """One causal adjacency estimated from strictly lagged information.

    Attributes:
        assets: Node labels in canonical (sorted) order.
        adjacency: ``(n_assets, n_assets)`` non-negative weights, zero diagonal,
            row-normalized so message passing is an averaging operator and
            cannot inflate signal magnitude with node degree.
        as_of: The date whose *forecast* this graph may be used for; it was
            built from observations strictly before it.
        window: Trailing observations used.
        method: Edge estimator.
        density: Fraction of retained off-diagonal edges.
    """

    assets: tuple[str, ...]
    adjacency: FloatArray
    as_of: pd.Timestamp
    window: int
    method: str
    density: float

    def __post_init__(self) -> None:
        size = len(self.assets)
        if self.adjacency.shape != (size, size):
            raise AssetGraphError("adjacency must be square over the asset set")
        if not np.isfinite(self.adjacency).all():
            raise AssetGraphError("adjacency must be finite")
        if (self.adjacency < -1e-12).any():
            raise AssetGraphError("adjacency weights must be non-negative")
        if np.abs(np.diag(self.adjacency)).max() > 1e-12:
            raise AssetGraphError("adjacency must have a zero diagonal; self-loops are implicit")

    def neighbours(self, asset: str, *, top: int = 5) -> list[tuple[str, float]]:
        """Return an asset's strongest neighbours, for inspection."""
        if asset not in self.assets:
            raise AssetGraphError(f"unknown asset {asset!r}")
        row = self.adjacency[self.assets.index(asset)]
        order = np.argsort(-row)[:top]
        return [(self.assets[index], float(row[index])) for index in order if row[index] > 0.0]


def build_dynamic_graph(
    returns: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    window: int = 63,
    method: EdgeMethod = "correlation",
    threshold: float = 0.3,
    max_degree: int = 8,
) -> DynamicGraph:
    """Estimate one graph from the ``window`` observations strictly before ``as_of``.

    Args:
        returns: Wide frame indexed by date with one column per asset.
        as_of: The forecast date. Observations on or after it are **excluded**.
        window: Trailing observations used.
        method: ``correlation`` (marginal), ``partial_correlation`` (conditional
            on all other assets, which removes the common-factor edge that makes
            every equity look connected to every other), or ``lagged_causality``
            (correlation of a neighbour's lagged return with the target's
            current return — directed, and the only one that can express lead-lag).
        threshold: Absolute edge weight below which an edge is dropped.
        max_degree: Retained neighbours per node; bounds message-passing cost and
            stops a hub from dominating every update.

    Raises:
        AssetGraphError: On an unusable window or out-of-range bound.
    """
    if window < MIN_GRAPH_WINDOW:
        raise AssetGraphError(f"window must be at least {MIN_GRAPH_WINDOW} observations")
    if not 0.0 <= threshold < 1.0:
        raise AssetGraphError("threshold must lie in [0, 1)")
    if max_degree < 1:
        raise AssetGraphError("max_degree must be at least one")
    if returns.shape[1] > MAX_NODES:
        raise AssetGraphError(f"graph exceeds the {MAX_NODES}-node ceiling")

    history = returns.loc[returns.index < as_of]
    if len(history) < window:
        raise AssetGraphError(
            f"only {len(history)} observations strictly before {as_of}; need {window}"
        )
    trailing = history.iloc[-window:]
    assets = tuple(str(column) for column in trailing.columns)

    if method == "lagged_causality":
        # Directed: neighbour j's return at t-1 against target i's return at t.
        lagged = trailing.shift(1).iloc[1:]
        current = trailing.iloc[1:]
        weights = _cross_correlation(current.to_numpy(float), lagged.to_numpy(float))
    else:
        correlation = np.corrcoef(trailing.to_numpy(float), rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0)
        weights = correlation if method == "correlation" else _partial_correlation(correlation)

    adjacency = np.abs(np.nan_to_num(weights, nan=0.0))
    np.fill_diagonal(adjacency, 0.0)
    adjacency[adjacency < threshold] = 0.0
    adjacency = _keep_top_degree(adjacency, max_degree)
    retained = float((adjacency > 0.0).sum())
    possible = max(len(assets) * (len(assets) - 1), 1)

    # Row-normalize: message passing must average neighbours, not sum them, or a
    # high-degree node's update grows with degree rather than with information.
    row_sums = adjacency.sum(axis=1, keepdims=True)
    normalized = np.divide(adjacency, row_sums, out=np.zeros_like(adjacency), where=row_sums > 0.0)

    return DynamicGraph(
        assets=assets,
        adjacency=normalized,
        as_of=pd.Timestamp(as_of),
        window=window,
        method=method,
        density=retained / possible,
    )


def _cross_correlation(current: FloatArray, lagged: FloatArray) -> FloatArray:
    """Return ``w[i, j]`` = corr(current_i, lagged_j)."""
    current_centred = current - current.mean(axis=0)
    lagged_centred = lagged - lagged.mean(axis=0)
    current_scale = np.maximum(current_centred.std(axis=0), 1e-12)
    lagged_scale = np.maximum(lagged_centred.std(axis=0), 1e-12)
    covariance = (current_centred.T @ lagged_centred) / current.shape[0]
    return np.asarray(covariance / np.outer(current_scale, lagged_scale), dtype=np.float64)


def _partial_correlation(correlation: FloatArray) -> FloatArray:
    """Return partial correlations by inverting the correlation matrix.

    Ridge-regularized before inversion: an empirical correlation matrix on a
    short window is frequently near-singular, and inverting it unregularized
    turns estimation noise into enormous spurious partial correlations.
    """
    size = correlation.shape[0]
    regularized = correlation + 1e-3 * np.eye(size)
    try:
        precision = np.linalg.inv(regularized)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by ridge
        raise AssetGraphError("correlation matrix is singular even after ridge") from exc
    diagonal = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    return np.asarray(-precision / np.outer(diagonal, diagonal), dtype=np.float64)


def _keep_top_degree(adjacency: FloatArray, max_degree: int) -> FloatArray:
    """Zero all but each node's strongest ``max_degree`` edges."""
    trimmed = np.zeros_like(adjacency)
    for row in range(adjacency.shape[0]):
        weights = adjacency[row]
        if not weights.any():
            continue
        keep = np.argsort(-weights)[:max_degree]
        trimmed[row, keep] = weights[keep]
    return trimmed


@dataclass(frozen=True)
class MessagePassingResult:
    """Predictions and provenance from one temporal message-passing fit."""

    predictions: pd.DataFrame
    n_rounds: int
    self_weight: float
    edge_method: str
    graph_density: float
    n_graphs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""
        return {
            "n_rounds": self.n_rounds,
            "self_weight": self.self_weight,
            "edge_method": self.edge_method,
            "graph_density": self.graph_density,
            "n_graphs": self.n_graphs,
        }


def temporal_message_passing(
    returns: pd.DataFrame,
    *,
    window: int = 63,
    n_rounds: int = 1,
    self_weight: float = 0.5,
    method: EdgeMethod = "correlation",
    threshold: float = 0.3,
    max_degree: int = 8,
    refit_every: int = 21,
) -> MessagePassingResult:
    """Predict each asset's next return by message passing over its causal graph.

    At each date the node feature is the asset's own most recent observed
    return; one round mixes in its neighbours' most recent returns, weighted by
    the graph. The readout is the mixed feature itself, so the model has **no
    fitted output layer** — it is a structural baseline that isolates the graph's
    contribution rather than letting a regression head absorb it.

    ``self_weight`` interpolates between "ignore the graph" (1.0, which reduces
    exactly to the lagged-return baseline) and "ignore the asset" (0.0). That
    reduction is what makes the ablation meaningful and is asserted by test.

    Args:
        refit_every: Bars between graph re-estimations. The graph is stale for at
            most this many bars, which is a cost/accuracy trade stated rather
            than hidden.

    Raises:
        AssetGraphError: On out-of-range bounds or an unusable panel.
    """
    if not 1 <= n_rounds <= MAX_ROUNDS:
        raise AssetGraphError(f"n_rounds must be in [1, {MAX_ROUNDS}]")
    if not 0.0 <= self_weight <= 1.0:
        raise AssetGraphError("self_weight must lie in [0, 1]")
    if refit_every < 1:
        raise AssetGraphError("refit_every must be at least one bar")

    frame = returns.sort_index()
    dates = frame.index
    predictions = pd.DataFrame(np.nan, index=dates, columns=frame.columns, dtype=float)
    graph: DynamicGraph | None = None
    densities: list[float] = []
    n_graphs = 0

    for position in range(window + 1, len(dates)):
        as_of = dates[position]
        if graph is None or (position - window - 1) % refit_every == 0:
            graph = build_dynamic_graph(
                frame,
                as_of=as_of,
                window=window,
                method=method,
                threshold=threshold,
                max_degree=max_degree,
            )
            densities.append(graph.density)
            n_graphs += 1
        # The node feature is the last *observed* return, strictly before as_of.
        feature = frame.iloc[position - 1].to_numpy(dtype=float)
        feature = np.nan_to_num(feature, nan=0.0)
        state = feature.copy()
        for _ in range(n_rounds):
            state = self_weight * state + (1.0 - self_weight) * (graph.adjacency @ state)
        predictions.iloc[position] = state

    return MessagePassingResult(
        predictions=predictions,
        n_rounds=n_rounds,
        self_weight=self_weight,
        edge_method=method,
        graph_density=float(np.mean(densities)) if densities else 0.0,
        n_graphs=n_graphs,
    )


def evaluate_cross_asset_generalization(
    predictions: pd.DataFrame, realized: pd.DataFrame
) -> pd.DataFrame:
    """Score predictions against realized *next-period* returns.

    The alignment is the thing to get right: a prediction stamped at date ``t``
    is a forecast for the return realized over ``t -> t+1``, so it is scored
    against ``realized.shift(-1)``. Scoring against the same-date return would
    manufacture a perfect result out of an off-by-one.
    """
    future = realized.shift(-1)
    aligned = predictions.reindex_like(future)
    rows: list[dict[str, Any]] = []
    for asset in future.columns:
        predicted = aligned[asset].to_numpy(dtype=float)
        actual = future[asset].to_numpy(dtype=float)
        usable = np.isfinite(predicted) & np.isfinite(actual)
        if usable.sum() < 20:
            continue
        rows.append(
            {
                "ticker": str(asset),
                "n_observations": int(usable.sum()),
                "information_coefficient": _safe_correlation(predicted[usable], actual[usable]),
                "hit_rate": float(np.mean(np.sign(predicted[usable]) == np.sign(actual[usable]))),
            }
        )
    return pd.DataFrame(rows)


def _safe_correlation(left: FloatArray, right: FloatArray) -> float:
    """Return Pearson correlation, or NaN when either side is constant."""
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])
