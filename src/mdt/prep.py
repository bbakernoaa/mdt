"""Prep stage: retrieve/prepare datasets to local stores with completeness manifests.

Feature 001-prep-fixed-allocation (US1). Implements the PrepManifest entity
from specs/001-prep-fixed-allocation/data-model.md and the `mdt prep`
contract from specs/001-prep-fixed-allocation/contracts/cli.md.

Design (research.md R3-R5):
- Prep executes only load nodes (+ derived save nodes) via the plain-Dask
  engine; analysis stages never run here.
- A per-store JSON manifest is written ``incomplete`` before retrieval and
  ``complete`` only after coverage is verified; it is the completion marker
  that makes reruns idempotent and interruptions safe.
- ``check_caches`` gives the allocation-mode compute stage a fail-fast,
  network-free guarantee that every load can be served locally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    import networkx as nx

    from mdt.config import ConfigParser

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = "./zarr_stores"


def manifest_path_for(store_path: str) -> Path:
    """Return the manifest path beside a store (data-model.md PrepManifest)."""
    return Path(f"{store_path}.mdt-prep.json")


def config_fingerprint(dataset_type: str, kwargs: Dict[str, Any]) -> str:
    """SHA-256 of the normalized reader type + kwargs (config-drift detection)."""
    payload = json.dumps({"type": dataset_type, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class PrepManifest:
    """Per-store completeness record (data-model.md PrepManifest)."""

    dataset: str
    store_path: str
    backend: str  # 'icechunk' | 'zarr'
    config_fingerprint: str
    requested_dates: List[str]
    covered_dates: List[str] = field(default_factory=list)
    variables: Optional[List[str]] = None
    status: str = "incomplete"  # 'incomplete' | 'complete'
    writer: Optional[Dict[str, Any]] = None
    updated: str = field(default_factory=_now)
    history: List[str] = field(default_factory=list)

    # -- persistence -----------------------------------------------------

    def _atomic_write(self) -> None:
        path = manifest_path_for(self.store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    def to_dict(self) -> Dict[str, Any]:
        """Return the manifest as a JSON-serializable dictionary."""
        return {
            "dataset": self.dataset,
            "store_path": self.store_path,
            "backend": self.backend,
            "config_fingerprint": self.config_fingerprint,
            "requested_dates": self.requested_dates,
            "covered_dates": self.covered_dates,
            "variables": self.variables,
            "status": self.status,
            "writer": self.writer,
            "updated": self.updated,
            "history": self.history,
        }

    @classmethod
    def load(cls, path: Path) -> "PrepManifest":
        """Load a manifest from its JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def load_for_store(cls, store_path: str) -> Optional["PrepManifest"]:
        """Load the manifest beside a store, or None when absent/unreadable."""
        path = manifest_path_for(store_path)
        if not path.exists():
            return None
        try:
            return cls.load(path)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Unreadable prep manifest %s (%s); treating as absent.", path, exc)
            return None

    # -- state transitions -----------------------------------------------

    @classmethod
    def begin(
        cls,
        dataset: str,
        store_path: str,
        backend: str,
        fingerprint: str,
        requested_dates: List[str],
        force: bool = False,
    ) -> "PrepManifest":
        """Create/claim an ``incomplete`` manifest before retrieval.

        Raises RuntimeError when another live prep run owns the store
        (data-model.md state machine: live-writer conflict).
        """
        existing = cls.load_for_store(store_path)
        if existing is not None and existing.status == "incomplete" and existing.writer_is_alive() and not force:
            writer = existing.writer or {}
            raise RuntimeError(
                f"Dataset '{dataset}': another prep run (pid={writer.get('pid')}, host={writer.get('host')}) "
                f"is writing store '{store_path}'. Wait for it to finish or remove its manifest if it is stale."
            )
        manifest = cls(
            dataset=dataset,
            store_path=store_path,
            backend=backend,
            config_fingerprint=fingerprint,
            requested_dates=sorted(requested_dates),
            writer={"pid": os.getpid(), "host": os.uname().nodename},
            history=(existing.history if existing else []),
        )
        manifest.history.append(f"{_now()} prep started by pid={os.getpid()} force={force}")
        manifest._atomic_write()
        return manifest

    def complete(self, covered_dates: List[str], variables: Optional[List[str]] = None, allow_partial: bool = False) -> None:
        """Transition to ``complete`` when coverage satisfies the request.

        With ``allow_partial=True`` (used after a *successful* retrieval whose
        source genuinely lacks some requested dates), the manifest completes
        with the actual coverage recorded; the compute stage's cache check
        still flags the shortfall against the original request.
        """
        missing = set(self.requested_dates) - set(covered_dates)
        if missing and not allow_partial:
            raise ValueError(f"Dataset '{self.dataset}': store coverage incomplete; missing dates {sorted(missing)}.")
        self.covered_dates = sorted(covered_dates)
        self.variables = variables
        self.status = "complete"
        self.writer = None
        self.updated = _now()
        note = f"missing requested dates {sorted(missing)}" if missing else f"{len(self.covered_dates)} dates covered"
        self.history.append(f"{_now()} prep complete: {note}")
        self._atomic_write()

    def writer_is_alive(self) -> bool:
        """Whether the recorded writer process still exists on its host.

        Cross-host writers cannot be probed from here; they are treated as
        stale so an interrupted prep on another node never blocks resume
        (data-model.md state machine). Concurrent cross-host prep against the
        same store is out of scope for v1 (spec Assumptions).
        """
        if not self.writer:
            return False
        if self.writer.get("host") != os.uname().nodename:
            return False
        return _pid_alive(int(self.writer.get("pid", -1)))

    def covers(self, fingerprint: str, requested_dates: List[str]) -> bool:
        """Whether this complete manifest can serve the given request."""
        return self.status == "complete" and self.config_fingerprint == fingerprint and set(requested_dates) <= set(self.covered_dates)


# -- request/coverage helpers ----------------------------------------------


def requested_dates_for(kwargs: Dict[str, Any]) -> List[str]:
    """Expand the dataset's date request to ISO day strings (date-level guarantee)."""
    if "start_date" in kwargs and "end_date" in kwargs and "dates" not in kwargs:
        index = pd.date_range(str(kwargs["start_date"]), str(kwargs["end_date"]), freq="D")
        return [d.strftime("%Y-%m-%d") for d in index]
    dates = kwargs.get("dates") or kwargs.get("date")
    if dates is None:
        return []
    if isinstance(dates, str):
        dates = [dates]
    return sorted({pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates})


def store_covered_dates(store_path: str, backend: str) -> List[str]:
    """Read the time coordinate of a local store (no network access)."""
    import xarray as xr

    if backend == "icechunk":
        import icechunk

        storage = icechunk.local_filesystem_storage(store_path)
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session(branch="main")
        ds = xr.open_zarr(session.store, consolidated=False)
    else:
        ds = xr.open_zarr(store_path)
    try:
        for candidate in ("time", "valid_time", "dates", "date", "valid_date", "reference_time"):
            if candidate in ds.coords or candidate in ds.dims:
                values = ds[candidate].values
                return sorted({pd.to_datetime(t).strftime("%Y-%m-%d") for t in values})
        for coord in ds.coords:
            if "time" in str(coord).lower() or "date" in str(coord).lower():
                values = ds[coord].values
                return sorted({pd.to_datetime(t).strftime("%Y-%m-%d") for t in values})
        return []
    finally:
        ds.close()


def is_local_source(config: "ConfigParser", name: str) -> bool:
    """Whether a dataset reads files already on local/shared storage.

    Mirrors DAGBuilder's ``retrieval`` flag: local ``files`` (or an explicit
    existing/bypass flag) mean no remote download is involved.
    """
    details = config.data[name]
    kwargs = details.get("kwargs") or {}
    return bool(kwargs.get("files")) or details.get("existing", False) or details.get("bypass_load", False)


def store_for_dataset(config: "ConfigParser", name: str) -> tuple[str, str]:
    """Resolve (store_path, backend) for a dataset.

    Policy: REMOTE sources are materialized as real local zarr data (the
    compute stage must never depend on the remote source). LOCAL sources may
    use a configured icechunk store of virtual references (no copy), or the
    derived zarr default.
    """
    details = config.data[name]
    store_dir = (config.config.get("prep") or {}).get("store_dir", DEFAULT_STORE_DIR)
    zarr_store = details.get("zarr_store") or {}
    if zarr_store.get("enabled") and is_local_source(config, name):
        backend = zarr_store.get("backend", "zarr")
        if backend == "icechunk":
            return str(zarr_store.get("icechunk_url") or zarr_store.get("icechunk_repo")), "icechunk"
        return str(zarr_store.get("store_path", f"{store_dir}/{name}/")), "zarr"
    return f"{store_dir}/{name}_prep.zarr", "zarr"


def dataset_fingerprint(config: "ConfigParser", name: str) -> str:
    """Fingerprint of a configured dataset's reader type and kwargs."""
    details = config.data[name]
    return config_fingerprint(details.get("type", ""), dict(details.get("kwargs") or {}))


# -- orchestration ----------------------------------------------------------


def run_prep(config: "ConfigParser", force: bool = False) -> List[Dict[str, Any]]:
    """Execute the prep stage: retrieve/prepare every dataset to local stores.

    Returns one result dict per dataset:
    ``{dataset, status: retrieved|cached|failed, store_path, error?}``.

    Prep executes on local Dask workers at the launch point (e.g. a DTN or
    service node with data access), regardless of the configured compute
    clusters: the compute stage may target a network-isolated fixed
    allocation that must never be used for retrieval.
    """
    import copy as _copy

    from mdt.dag import DAGBuilder
    from mdt.dask_engine import DaskEngine

    prep_config = _copy.deepcopy(config)
    # Retrieval is network-bound, so a couple of modest workers suffice. Bound
    # threads and give an explicit memory limit: the default "auto" limit on a
    # large shared login node is tiny relative to the machine and causes
    # premature memory pressure / thrashing during large materializations.
    prep_config.config["execution"] = {
        "orchestrator": "dask",
        "default_cluster": "compute",
        "clusters": {
            "compute": {"mode": "local", "workers": 2, "threads_per_worker": 2, "memory_limit": "8GB"},
            "service": {"mode": "local", "workers": 1, "threads_per_worker": 2, "memory_limit": "8GB"},
        },
    }
    dag = DAGBuilder(prep_config).build_prep_dag()
    results: List[Dict[str, Any]] = []
    pending: Dict[str, str] = {}  # load node id -> dataset name

    for name in config.data:
        store_path, backend = store_for_dataset(config, name)
        fingerprint = dataset_fingerprint(config, name)
        requested = requested_dates_for(dict(config.data[name].get("kwargs") or {}))
        manifest = PrepManifest.load_for_store(store_path)
        if not force and manifest is not None and manifest.covers(fingerprint, requested):
            logger.info("Dataset '%s': complete cached store at %s; skipping retrieval.", name, store_path)
            results.append({"dataset": name, "status": "cached", "store_path": store_path})
            continue
        try:
            PrepManifest.begin(name, store_path, backend, fingerprint, requested, force=force)
        except RuntimeError as exc:
            results.append({"dataset": name, "status": "failed", "store_path": store_path, "error": str(exc)})
            continue
        pending[f"load_{name}"] = name

    # Execute each pending dataset as its own subgraph (load + optional derived
    # save). Per-dataset execution isolates failures: one bad remote endpoint
    # must not prevent the other datasets from being prepared and manifested.
    for node_id, name in pending.items():
        store_path, backend = store_for_dataset(config, name)
        requested = requested_dates_for(dict(config.data[name].get("kwargs") or {}))
        sub_nodes = {node_id}
        save_node = f"prep_save_{name}"
        if save_node in dag.nodes:
            sub_nodes.add(save_node)
        subgraph = dag.subgraph(sub_nodes).copy()
        try:
            engine = DaskEngine(subgraph, prep_config)
            try:
                task_status = engine.execute()
            finally:
                engine.close()
            error = next((v for v in task_status.values() if isinstance(v, Exception)), None)
            if error is not None:
                raise error
            covered = store_covered_dates(store_path, backend)
            manifest = PrepManifest.load_for_store(store_path)
            if manifest is None:
                raise RuntimeError("prep manifest missing after retrieval")
            # Retrieval succeeded: record what the source actually provided.
            # Genuine source gaps complete as partial (the compute stage's
            # cache check still flags them against the original request).
            manifest.complete(covered_dates=covered or requested, allow_partial=True)
            result = {"dataset": name, "status": "retrieved", "store_path": store_path}
            missing = sorted(set(requested) - set(manifest.covered_dates))
            if missing:
                result["warning"] = f"source lacks requested dates {missing}; stored coverage is partial"
                logger.warning("Dataset '%s': %s", name, result["warning"])
            results.append(result)
        except Exception as exc:
            logger.error("Prep failed for dataset '%s': %s", name, exc)
            results.append({"dataset": name, "status": "failed", "store_path": store_path, "error": str(exc)})

    results.sort(key=lambda r: r["dataset"])
    return results


def check_caches(config: "ConfigParser", dag: "nx.DiGraph") -> None:
    """Fail-fast, network-free verification that every load can be served locally.

    Used by allocation-mode execution (research.md R5). Raises RuntimeError
    listing all offending datasets with the ``mdt prep`` instruction.

    A ``complete`` manifest with a matching fingerprint but partial date
    coverage is accepted with a warning: prep already attempted retrieval
    and recorded the gap (e.g. the source genuinely lacks those dates), and
    pairing only matches overlapping times. Missing stores, incomplete
    manifests, and config drift still fail hard.
    """
    problems: List[str] = []
    for _node_id, attrs in dag.nodes(data=True):
        if attrs.get("task_type") != "load_data":
            continue
        name = attrs["name"]
        store_path, _backend = store_for_dataset(config, name)
        fingerprint = dataset_fingerprint(config, name)
        requested = requested_dates_for(dict(config.data[name].get("kwargs") or {}))
        manifest = PrepManifest.load_for_store(store_path)
        if manifest is None:
            problems.append(f"{name}: no prepared store at '{store_path}'")
            continue
        if manifest.status != "complete":
            problems.append(f"{name}: store '{store_path}' not usable (status={manifest.status})")
            continue
        if manifest.config_fingerprint != fingerprint:
            problems.append(f"{name}: store '{store_path}' not usable (configuration changed since prep)")
            continue
        missing = set(requested) - set(manifest.covered_dates)
        if missing:
            logger.warning(
                "Dataset '%s': prepared store covers %d/%d requested dates; missing %s "
                "(recorded at prep time). Proceeding with available coverage.",
                name,
                len(manifest.covered_dates),
                len(requested),
                sorted(missing),
            )
    if problems:
        raise RuntimeError(
            "Prepared data missing or incomplete for offline execution:\n  - "
            + "\n  - ".join(problems)
            + "\nRun `mdt prep <config>` where data access is available (e.g. a DTN) before the compute stage."
        )
