"""Seaborn renderings of time-frequency tensors (SF-S3-MR2).

An image of a spectrogram is an **inspection aid**. Per the issue's explicit
non-goal, an attractive picture is not evidence of predictability, and every
figure this module produces is annotated so that it cannot be quoted out of
context: the caption always carries the channel, the window geometry, the
frequency unit, the normalization and its fit interval, and the data provenance.

Design rules that follow from that:

* Sequential, perceptually uniform, colourblind-safe colour maps only. A
  diverging map on a non-normalized power surface would invent a meaningful
  midpoint that does not exist.
* Axes are always labelled with units. A time-frequency image with an unlabelled
  frequency axis is unreadable and therefore unfalsifiable.
* Masked samples are rendered as explicit gaps, never as the lowest colour —
  "no data" and "low power" must not look alike.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from quant_platform.features.time_frequency import (  # noqa: E402
    TimeFrequencyError,
    TimeFrequencyTensor,
)
from quant_platform.reporting.plots import _save  # noqa: E402

#: Sequential, perceptually uniform, colourblind-safe.
POWER_COLORMAP = "mako"

#: Colour used for masked (unobserved) cells, distinct from every map value.
MASKED_COLOR = "#bdbdbd"


def _caption(tensor: TimeFrequencyTensor, *, ticker: str, observation: str) -> str:
    """Build the honest provenance caption attached to every figure."""
    metadata = tensor.metadata
    window = metadata.window_parameters
    normalization = str(metadata.normalization)
    if metadata.fitted_state is not None:
        state = metadata.fitted_state
        normalization = (
            f"{normalization} fit on {state.fit_start}..{state.fit_end} "
            f"(n={state.sample_count})"
        )
    return (
        f"{metadata.representation} | {ticker} @ {observation} | "
        f"window length={window['length']} segment={window['segment_length']} "
        f"hop={window['hop']} taper={window['taper']} detrend={window['detrend']} | "
        f"normalization: {normalization}\n"
        "Causal trailing windows; time axis runs oldest to most recent within the window. "
        "Rendering is an inspection aid, not evidence of predictability."
    )


def plot_time_frequency_sample(
    tensor: TimeFrequencyTensor,
    path: Path,
    *,
    sample: int,
    channel: str,
    ticker: str,
    observation: str,
) -> Path:
    """Render one ``(sample, channel)`` surface as a labelled heatmap.

    Args:
        tensor: Source tensor.
        path: Destination PNG path.
        sample: Sample index to render.
        channel: Channel name; must be present in the tensor metadata.
        ticker: Ticker label for the caption.
        observation: Observation date label for the caption.

    Raises:
        TimeFrequencyError: If the sample or channel is out of range, or the
            selected sample is masked — an unobserved window has nothing to show
            and rendering it would produce a blank image that reads as real.
    """
    metadata = tensor.metadata
    if channel not in metadata.channels:
        raise TimeFrequencyError(f"unknown channel {channel!r}; have {list(metadata.channels)}")
    if not 0 <= sample < metadata.n_samples:
        raise TimeFrequencyError(f"sample {sample} out of range [0, {metadata.n_samples})")
    channel_index = metadata.channels.index(channel)
    if not bool(tensor.mask[sample, channel_index]):
        raise TimeFrequencyError(
            f"sample {sample} channel {channel!r} is masked (warm-up or missing data); "
            "there is nothing to render"
        )

    surface = tensor.values[sample, channel_index]
    frequencies = np.asarray(metadata.frequency_values, dtype=np.float64)
    axis_label = (
        "frequency (cycles per bar)"
        if metadata.frequency_axis == "cycles_per_bar"
        else "period (bars)"
    )
    colorbar_label = "standardized log power" if metadata.normalization != "none" else "power"

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        colormap = sns.color_palette(POWER_COLORMAP, as_cmap=True).with_extremes(bad=MASKED_COLOR)
        sns.heatmap(
            np.ma.masked_invalid(surface),
            cmap=colormap,
            ax=ax,
            cbar_kws={"label": colorbar_label},
            yticklabels=[f"{value:.3g}" for value in frequencies],
            xticklabels=[str(step) for step in range(1, metadata.n_time + 1)],
        )
        ax.set_title(f"{metadata.representation.title()} — {ticker} {channel}")
        ax.set_xlabel("sub-window position (oldest → most recent)")
        ax.set_ylabel(axis_label)
        ax.invert_yaxis()
        fig.text(
            0.5,
            -0.16,
            _caption(tensor, ticker=ticker, observation=observation),
            ha="center",
            va="top",
            fontsize=7.5,
            wrap=True,
        )
    return _save(fig, Path(path))


def plot_channel_coverage(tensor: TimeFrequencyTensor, path: Path) -> Path:
    """Render observed-sample coverage per channel.

    Coverage is the first thing to check before reading any surface: a channel
    that is 40% masked produces images that look fine individually and a biased
    aggregate. Showing it explicitly makes the warm-up and missing-data cost
    visible rather than buried in a mask array.
    """
    metadata = tensor.metadata
    coverage = tensor.mask.mean(axis=0)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        sns.barplot(
            x=list(metadata.channels),
            y=[float(value) for value in coverage],
            hue=list(metadata.channels),
            legend=False,
            palette="colorblind",
            ax=ax,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Observed sample coverage per channel")
        ax.set_xlabel("channel")
        ax.set_ylabel("fraction of samples observed")
        for position, value in enumerate(coverage):
            ax.text(position, float(value) + 0.02, f"{float(value):.1%}", ha="center", fontsize=9)
        fig.text(
            0.5,
            -0.06,
            f"n={metadata.n_samples} samples over {metadata.coverage_start}.."
            f"{metadata.coverage_end}; masked samples are warm-up or missing data, "
            "never imputed.",
            ha="center",
            va="top",
            fontsize=7.5,
        )
    return _save(fig, Path(path))
