"""Prep-stage tests (feature 001-prep-fixed-allocation, US1).

Scenarios per specs/001-prep-fixed-allocation/quickstart.md §A:
- test_prep_dag_prunes_analysis (FR-002)
- manifest lifecycle / idempotency / resume (FR-003, FR-005)
- summary + exit codes (FR-004)
- local end-to-end cached rerun (SC-002)
"""

import json
import os
import time

import numpy as np
import pytest
import xarray as xr

from mdt.config import ConfigParser
from mdt.dag import DAGBuilder
from mdt.prep import (
    PrepManifest,
    check_caches,
    manifest_path_for,
    run_prep,
)


def _write_config(tmp_path, body):
    path = tmp_path / "wf.yaml"
    path.write_text(body)
    return ConfigParser(str(path))


LOCAL_TWO_SOURCES = """\
data:
  obs:
    type: aeronet
    cluster: compute
    kwargs: {files: dummy_obs}
  model:
    type: gefs
    cluster: compute
    kwargs: {files: dummy_model}
statistics:
  s:
    input: obs
    metrics: [mb]
    kwargs: {obs_var: o, mod_var: m}
plots:
  p:
    input: obs
    type: scatter
    kwargs: {var1: o, var2: m}
execution:
  orchestrator: dask
  clusters:
    compute: {mode: local, workers: 1}
    service: {mode: local, workers: 1}
"""


def _fake_dataset():
    return xr.Dataset(
        {"o": ("x", [1.0, 2.0]), "m": ("x", [1.5, 2.5])},
        coords={"time": ("x", np.array(["2025-08-01", "2025-08-02"], dtype="datetime64[ns]"))},
    )


# --- T007: PrepDAG pruning (FR-002) ---


def test_prep_dag_prunes_analysis(tmp_path):
    config = _write_config(tmp_path, LOCAL_TWO_SOURCES)
    dag = DAGBuilder(config).build_prep_dag()
    types = {attrs["task_type"] for _, attrs in dag.nodes(data=True)}
    assert types == {"load_data", "save_data"}
    assert set(dag.nodes) == {"load_obs", "load_model", "prep_save_obs", "prep_save_model"}
    assert dag.nodes["prep_save_obs"]["url"].endswith("obs_prep.zarr")
    assert dag.nodes["prep_save_obs"]["backend"] == "zarr"
    assert list(dag.predecessors("prep_save_obs")) == ["load_obs"]


def test_prep_dag_skips_derived_save_for_local_store_writing_loads(tmp_path):
    body = LOCAL_TWO_SOURCES.replace(
        "    kwargs: {files: dummy_model}",
        "    kwargs: {files: dummy_model}\n    zarr_store: {enabled: true, backend: icechunk, icechunk_url: ./store_model}",
    )
    config = _write_config(tmp_path, body)
    dag = DAGBuilder(config).build_prep_dag()
    assert "prep_save_model" not in dag.nodes
    assert "prep_save_obs" in dag.nodes


def test_prep_dag_remote_source_materializes_local_zarr(tmp_path):
    """Remote sources download real data to local zarr; virtual-reference
    ingestion options are stripped so compute never needs the remote source."""
    body = LOCAL_TWO_SOURCES.replace(
        "  model:\n    type: gefs\n    cluster: compute\n    kwargs: {files: dummy_model}",
        "  model:\n    type: gefs\n    cluster: compute\n    kwargs: {start_date: '2025-08-01', end_date: '2025-08-02'}\n"
        "    zarr_store: {enabled: true, backend: icechunk, icechunk_url: ./s3refs}",
    )
    config = _write_config(tmp_path, body)
    dag = DAGBuilder(config).build_prep_dag()
    assert dag.nodes["load_model"]["retrieval"] is True
    kwargs = dag.nodes["load_model"]["kwargs"]
    assert "use_icechunk" not in kwargs and "icechunk_url" not in kwargs
    assert "prep_save_model" in dag.nodes
    assert dag.nodes["prep_save_model"]["url"].endswith("model_prep.zarr")


def test_prep_dag_respects_prep_store_dir(tmp_path):
    config = _write_config(tmp_path, LOCAL_TWO_SOURCES + "prep:\n  store_dir: ./my_prep\n")
    dag = DAGBuilder(config).build_prep_dag()
    assert dag.nodes["prep_save_obs"]["url"] == "./my_prep/obs_prep.zarr"


# --- T008: manifest lifecycle (FR-003/005, data-model.md PrepManifest) ---


def test_manifest_lifecycle(tmp_path):
    store = tmp_path / "obs.zarr"
    m = PrepManifest.begin(dataset="obs", store_path=str(store), backend="zarr", fingerprint="fp1", requested_dates=["2025-08-01"])
    assert m.status == "incomplete"
    assert m.writer == {"pid": os.getpid(), "host": os.uname().nodename}
    loaded = PrepManifest.load(manifest_path_for(str(store)))
    assert loaded.status == "incomplete"

    m.complete(covered_dates=["2025-08-01"], variables=["o", "m"])
    loaded = PrepManifest.load(manifest_path_for(str(store)))
    assert loaded.status == "complete"
    assert loaded.writer is None
    assert loaded.covered_dates == ["2025-08-01"]
    assert loaded.history  # provenance appended (FR-013)


def test_manifest_complete_requires_coverage(tmp_path):
    store = tmp_path / "obs.zarr"
    m = PrepManifest.begin(dataset="obs", store_path=str(store), backend="zarr", fingerprint="fp1", requested_dates=["2025-08-01", "2025-08-02"])
    with pytest.raises(ValueError, match="coverage"):
        m.complete(covered_dates=["2025-08-01"])
    assert PrepManifest.load(manifest_path_for(str(store))).status == "incomplete"


def test_manifest_stale_writer_resumable(tmp_path):
    store = tmp_path / "obs.zarr"
    PrepManifest.begin(dataset="obs", store_path=str(store), backend="zarr", fingerprint="fp1", requested_dates=[])
    data = json.loads(manifest_path_for(str(store)).read_text())
    data["writer"] = {"pid": 999999999, "host": "other-host"}
    manifest_path_for(str(store)).write_text(json.dumps(data))
    loaded = PrepManifest.load(manifest_path_for(str(store)))
    assert loaded.writer_is_alive() is False


def test_manifest_live_writer_conflict(tmp_path):
    store = tmp_path / "obs.zarr"
    PrepManifest.begin(dataset="obs", store_path=str(store), backend="zarr", fingerprint="fp1", requested_dates=[])
    with pytest.raises(RuntimeError, match="another prep run"):
        PrepManifest.begin(dataset="obs", store_path=str(store), backend="zarr", fingerprint="fp1", requested_dates=[])


# --- T009/T010: run_prep end-to-end, idempotency, summary, failures ---


def _patch_load(monkeypatch, fake_load):
    monkeypatch.setattr("mdt.tasks.data.load_data", fake_load)
    monkeypatch.setattr("mdt.dask_engine.load_data", fake_load)


def _call_marker(tmp_path):
    """Cross-process call counter: loads run in Dask worker processes, so an
    in-memory list would stay empty in the test process."""
    marker_dir = tmp_path / "load_calls"
    marker_dir.mkdir(exist_ok=True)

    def record(name):
        (marker_dir / f"{name}-{time.monotonic_ns()}").write_text(name)

    def count():
        return len(list(marker_dir.iterdir()))

    def reset():
        for f in marker_dir.iterdir():
            f.unlink()

    return record, count, reset


def test_run_prep_local_end_to_end_and_idempotent(tmp_path, monkeypatch):
    config = _write_config(tmp_path, LOCAL_TWO_SOURCES)
    monkeypatch.chdir(tmp_path)
    record, count, reset = _call_marker(tmp_path)

    def fake_load(name, dataset_type, kwargs):
        record(name)
        return _fake_dataset()

    _patch_load(monkeypatch, fake_load)

    results = run_prep(config)
    assert {r["dataset"]: r["status"] for r in results} == {"obs": "retrieved", "model": "retrieved"}
    assert count() == 2
    for name in ("obs", "model"):
        store = tmp_path / "zarr_stores" / f"{name}_prep.zarr"
        assert store.is_dir()
        m = PrepManifest.load(manifest_path_for(str(store)))
        assert m.status == "complete"

    # second run: zero retrievals, all cached (SC-002)
    reset()
    t0 = time.monotonic()
    results2 = run_prep(config)
    assert {r["dataset"]: r["status"] for r in results2} == {"obs": "cached", "model": "cached"}
    assert count() == 0
    assert time.monotonic() - t0 < 60


def test_run_prep_force_bypasses_cache(tmp_path, monkeypatch):
    config = _write_config(tmp_path, LOCAL_TWO_SOURCES)
    monkeypatch.chdir(tmp_path)
    record, count, reset = _call_marker(tmp_path)

    def fake_load(name, dataset_type, kwargs):
        record(name)
        return _fake_dataset()

    _patch_load(monkeypatch, fake_load)
    run_prep(config)
    reset()
    results = run_prep(config, force=True)
    assert all(r["status"] == "retrieved" for r in results)
    assert count() == 2


def test_run_prep_failure_reported(tmp_path, monkeypatch):
    config = _write_config(tmp_path, LOCAL_TWO_SOURCES)
    monkeypatch.chdir(tmp_path)

    def fake_load(name, dataset_type, kwargs):
        if name == "model":
            raise ValueError("remote endpoint exploded")
        return _fake_dataset()

    _patch_load(monkeypatch, fake_load)
    results = run_prep(config)
    by_name = {r["dataset"]: r for r in results}
    assert by_name["obs"]["status"] == "retrieved"
    assert by_name["model"]["status"] == "failed"
    assert "remote endpoint exploded" in by_name["model"]["error"]


def test_run_prep_interrupted_store_not_served_complete(tmp_path, monkeypatch):
    config = _write_config(tmp_path, LOCAL_TWO_SOURCES)
    monkeypatch.chdir(tmp_path)
    _patch_load(monkeypatch, lambda *a, **k: _fake_dataset())
    run_prep(config)

    # simulate interruption: revert manifest to incomplete with dead writer
    store = tmp_path / "zarr_stores" / "obs_prep.zarr"
    path = manifest_path_for(str(store))
    data = json.loads(path.read_text())
    data["status"] = "incomplete"
    data["writer"] = {"pid": 999999999, "host": "other-host"}
    path.write_text(json.dumps(data))

    record, count, _ = _call_marker(tmp_path)

    def counting_load(name, dataset_type, kwargs):
        record(name)
        return _fake_dataset()

    _patch_load(monkeypatch, counting_load)
    results = run_prep(config)
    by_name = {r["dataset"]: r["status"] for r in results}
    assert by_name["obs"] == "retrieved"  # re-prepped, not served from incomplete store
    assert by_name["model"] == "cached"
    assert count() == 1


def test_check_caches_fails_fast_on_incomplete(tmp_path, monkeypatch):
    config = _write_config(tmp_path, LOCAL_TWO_SOURCES)
    monkeypatch.chdir(tmp_path)
    _patch_load(monkeypatch, lambda *a, **k: _fake_dataset())
    run_prep(config)

    store = tmp_path / "zarr_stores" / "obs_prep.zarr"
    path = manifest_path_for(str(store))
    data = json.loads(path.read_text())
    data["status"] = "incomplete"
    path.write_text(json.dumps(data))

    dag = DAGBuilder(config).build()
    with pytest.raises(RuntimeError, match=r"obs[\s\S]*mdt prep"):
        check_caches(config, dag)


def test_prep_summary_and_exit_code(tmp_path, monkeypatch, capsys):
    from mdt.cli import main

    cfg_file = tmp_path / "wf.yaml"
    cfg_file.write_text(LOCAL_TWO_SOURCES)
    monkeypatch.chdir(tmp_path)
    _patch_load(monkeypatch, lambda *a, **k: _fake_dataset())
    monkeypatch.setattr("sys.argv", ["mdt", "prep", str(cfg_file)])
    main()
    out = capsys.readouterr().out
    assert "obs: retrieved -> " in out
    assert "PREP SUMMARY: 2 retrieved, 0 cached, 0 failed" in out


def test_prep_cli_exit_1_on_failure(tmp_path, monkeypatch):
    from mdt.cli import main

    cfg_file = tmp_path / "wf.yaml"
    cfg_file.write_text(LOCAL_TWO_SOURCES)
    monkeypatch.chdir(tmp_path)

    def fake_load(name, dataset_type, kwargs):
        raise ValueError("boom")

    _patch_load(monkeypatch, fake_load)
    monkeypatch.setattr("sys.argv", ["mdt", "prep", str(cfg_file)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1