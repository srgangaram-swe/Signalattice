"""Shared utilities: reproducibility, hashing, paths and git metadata."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

type PathLike = str | os.PathLike[str]


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and the ``PYTHONHASHSEED`` env var for reproducibility.

    Note: this seeds the standard library and NumPy global RNGs. Library code
    should additionally pass explicit ``random_state``/``seed`` values to
    estimators so results do not depend on call ordering.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def project_root() -> Path:
    """Best-effort project root.

    Walk upwards from this file looking for a ``pyproject.toml``. Falls back to
    the current working directory when not found (e.g. installed wheel).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def resolve_path(path: PathLike, base: PathLike | None = None) -> Path:
    """Resolve ``path`` to an absolute :class:`~pathlib.Path`.

    Absolute paths are returned as-is. Relative paths are resolved against
    ``base`` (defaults to the current working directory) so the platform never
    hardcodes machine-specific absolute paths.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    base_path = Path(base).expanduser() if base is not None else Path.cwd()
    return (base_path / p).resolve()


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hash_dataframe(df: pd.DataFrame, *, length: int = 12) -> str:
    """Deterministic short hash of a DataFrame's contents and schema.

    Used to produce a dataset version/fingerprint for experiment tracking so a
    run can be tied back to the exact data it consumed.
    """
    hasher = hashlib.sha256()
    # Schema + shape.
    hasher.update(str(list(df.columns)).encode())
    hasher.update(str(df.shape).encode())
    # Index (e.g. dates) and values, via pandas' stable hashing.
    index_hash = pd.util.hash_pandas_object(df.index, index=True).to_numpy(dtype=np.uint64)
    value_hash = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
    hasher.update(index_hash.tobytes())
    hasher.update(value_hash.tobytes())
    return hasher.hexdigest()[:length]


def hash_dict(mapping: dict[str, Any], *, length: int = 12) -> str:
    """Deterministic short hash of a JSON-serialisable mapping."""
    import json

    payload = json.dumps(mapping, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def git_commit_hash(short: bool = True) -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
    if not short:
        args = ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=str(project_root()),
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git not installed / not a repo
        return None
    return None


def annualization_factor(periods_per_year: int = 252) -> float:
    """Square-root-of-time annualisation factor for volatility scaling."""
    return float(np.sqrt(periods_per_year))
