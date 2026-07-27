"""Content-addressed, integrity-checked store for time-frequency tensors (SF-S3-MR2).

Tensors are expensive to build and are consumed repeatedly, so they are cached.
A cache of scientific artifacts is only useful if a hit is provably the same
thing a rebuild would produce, which is what this store enforces:

* **Content addressing.** The object id is the tensor metadata's semantic
  identity — representation, channels, frequency grid, window contract, coverage,
  normalization state, and the array digests. Any change to any of those is a
  different object, so a stale artifact cannot be served under a new
  configuration.
* **Verification on read.** Every load re-hashes the bytes and compares against
  the manifest. Silent bit-rot, a truncated write, or hand-edited data is an
  error, not a slightly different result three stages downstream.
* **Atomic publication.** Objects are staged in a temporary directory and moved
  into place with a single rename, so a crash mid-write leaves no half-object
  that a later run would treat as a cache hit.
* **Path containment.** The root may not be a symlink, and every path inside a
  manifest must be a contained relative path. Object ids are validated as hex
  digests before touching the filesystem, so an identity cannot escape the root.
* **Bounded storage.** A write above the configured byte ceiling is refused
  before it is attempted.

Arrays are stored as ``.npy`` — a stable, self-describing, dependency-free
format whose bytes are deterministic for a given array.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath

import numpy as np

from quant_platform.features.registry import canonical_json
from quant_platform.features.time_frequency import (
    TensorBoundsError,
    TensorIntegrityError,
    TimeFrequencyError,
    TimeFrequencyMetadata,
    TimeFrequencyTensor,
    digest_array,
)

OBJECT_ID_RE = re.compile(r"^[0-9a-f]{64}$")

VALUES_FILE = "values.npy"
MASK_FILE = "mask.npy"
MANIFEST_FILE = "manifest.json"


class TimeFrequencyStore:
    """Local immutable store for content-addressed time-frequency tensors.

    Args:
        root: Store root directory. Created if absent; may not be a symlink.
        max_object_bytes: Refusal threshold on a single object's array bytes.

    Raises:
        TimeFrequencyError: If the root is a symlink or the ceiling is invalid.
    """

    def __init__(self, root: str | Path, *, max_object_bytes: int = 2_000_000_000) -> None:
        if max_object_bytes < 1:
            raise TimeFrequencyError("max_object_bytes must be positive")
        candidate = Path(root).expanduser()
        if candidate.is_symlink():
            # A symlinked root would let a manifest-relative path resolve
            # outside the tree the caller believes it is writing to.
            raise TimeFrequencyError("time-frequency store root cannot be a symlink")
        self.root = candidate.resolve()
        self.objects_dir = self.root / "objects"
        self.staging_dir = self.root / ".staging"
        self.max_object_bytes = max_object_bytes
        for directory in (self.root, self.objects_dir, self.staging_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def object_path(self, object_id: str) -> Path:
        """Return the contained directory for ``object_id``.

        Raises:
            TimeFrequencyError: If the id is not a full lowercase SHA-256, or if
                the resolved path would escape the objects directory.
        """
        if not OBJECT_ID_RE.fullmatch(object_id):
            raise TimeFrequencyError("object id must be a lowercase full SHA-256 digest")
        path = (self.objects_dir / object_id).resolve()
        if path.parent != self.objects_dir:
            raise TimeFrequencyError("resolved object path escapes the store")
        return path

    def exists(self, object_id: str) -> bool:
        """Return whether a published object directory exists."""
        return (self.object_path(object_id) / MANIFEST_FILE).is_file()

    def write(self, tensor: TimeFrequencyTensor) -> str:
        """Publish a tensor atomically and return its object id.

        Re-publishing an identical tensor is a no-op that returns the existing
        id; re-publishing *different* bytes under the same identity is
        impossible by construction, because the digests are part of the identity.

        Raises:
            TensorBoundsError: If the arrays exceed ``max_object_bytes``.
        """
        tensor.verify()
        if tensor.nbytes > self.max_object_bytes:
            raise TensorBoundsError(
                f"tensor of {tensor.nbytes} bytes exceeds the "
                f"{self.max_object_bytes} byte ceiling"
            )
        object_id = tensor.metadata.identity
        destination = self.object_path(object_id)
        if destination.exists():
            return object_id

        stage = self.staging_dir / f"{object_id}.{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        try:
            _write_array(stage / VALUES_FILE, tensor.values)
            _write_array(stage / MASK_FILE, tensor.mask)
            _write_json(stage / MANIFEST_FILE, tensor.metadata.model_dump(mode="json"))
            _fsync_directory(stage)
            os.replace(stage, destination)
            _fsync_directory(self.objects_dir)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return object_id

    def read(self, object_id: str) -> TimeFrequencyTensor:
        """Load and verify a published tensor.

        Raises:
            TimeFrequencyError: If the object is absent or its manifest is
                unreadable or invalid.
            TensorIntegrityError: If stored bytes do not match the manifest, or
                the manifest's own identity does not match the id it is filed
                under.
        """
        directory = self.object_path(object_id)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.is_file():
            raise TimeFrequencyError(f"no time-frequency object {object_id}")
        try:
            raw = manifest_path.read_bytes()
        except OSError as exc:
            raise TimeFrequencyError(f"unreadable tensor manifest for {object_id}") from exc
        try:
            # JSON-mode validation: the contract is strict, but JSON has no
            # native date or tuple, so those are decoded from their canonical
            # string and array forms rather than rejected.
            metadata = TimeFrequencyMetadata.model_validate_json(raw)
        except ValueError as exc:
            raise TimeFrequencyError(f"invalid tensor manifest for {object_id}") from exc
        if metadata.identity != object_id:
            # The directory name is derived from the metadata, so a mismatch
            # means the manifest was altered after publication.
            raise TensorIntegrityError(
                f"manifest identity {metadata.identity[:12]} does not match object id "
                f"{object_id[:12]}"
            )
        values = _read_array(directory / VALUES_FILE, np.float64)
        mask = _read_array(directory / MASK_FILE, np.bool_)
        if digest_array(values) != metadata.values_sha256:
            raise TensorIntegrityError(f"tensor values digest mismatch for {object_id}")
        if digest_array(mask) != metadata.mask_sha256:
            raise TensorIntegrityError(f"tensor mask digest mismatch for {object_id}")
        return TimeFrequencyTensor(values=values, mask=mask, metadata=metadata)

    def list_objects(self) -> tuple[str, ...]:
        """Return published object ids in deterministic order."""
        return tuple(
            sorted(
                entry.name
                for entry in self.objects_dir.iterdir()
                if entry.is_dir() and OBJECT_ID_RE.fullmatch(entry.name)
            )
        )


def _write_array(path: Path, array: np.ndarray) -> None:
    """Write a C-ordered ``.npy`` file without pickling."""
    with path.open("wb") as handle:
        # allow_pickle=False: a stored tensor must never be an executable
        # deserialization boundary.
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _read_array(path: Path, dtype: type[np.generic]) -> np.ndarray:
    """Read a ``.npy`` array, refusing pickled payloads and wrong dtypes."""
    if not path.is_file():
        raise TimeFrequencyError(f"missing tensor array {PurePosixPath(path.name)}")
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise TensorIntegrityError(f"unreadable tensor array {path.name}") from exc
    if array.dtype != np.dtype(dtype):
        raise TensorIntegrityError(
            f"tensor array {path.name} has dtype {array.dtype}, expected {np.dtype(dtype)}"
        )
    return np.asarray(array)


def _write_json(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json(payload))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
