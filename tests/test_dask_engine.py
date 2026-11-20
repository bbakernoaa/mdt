"""Tests for the plain-Dask orchestrator (no Prefect server/state)."""

from unittest.mock import MagicMock

import pytest
import xarray as xr

from mdt.config import ConfigParser
from mdt.dag import DAGBuilder
from mdt.dask_engine import DaskEngine, wait_for_workers
from mdt.dask_runtime import resolve_task_resource
from mdt.engine_registry import EngineRegistry


def test_dask_engine_registered():
    """The dask orchestrator resolves to DaskEngine."""
    assert EngineRegistry.get_engine("dask") is DaskEngine


def test_resolve_task_resource_routing():
    """Retrieval loads route to service; cached loads to cache_cluster; explicit wins."""
    exec_cfg = {"default_cluster": "compute", "cache_cluster": "compute"}
    retrieval = {"task_type": "load_data", "cluster": "service", "retrieval": True}
    cached = {"task_type": "load_data", "cluster": "service", "retrieval": False}
    explicit = {"task_type": "load_data", "cluster": "service", "cluster_explicit": True, "retrieval": False}
    stats = {"task_type": "compute_statistics", "cluster": "compute"}
    assert resolve_task_resource(retrieval, exec_cfg) == "SERVICE"
    assert resolve_task_resource(cached, exec_cfg) == "COMPUTE"
    assert resolve_task_resource(explicit, exec_cfg) == "SERVICE"
    assert resolve_task_resource(stats, exec_cfg) == "COMPUTE"
    # Without cache_cluster, cached loads keep their configured cluster.
    assert resolve_task_resource(cached, {"default_cluster": "compute"}) == "SERVICE"


def test_dag_marks_retrieval_vs_cached_loads(tmp_path):
    """DAG load nodes carry the retrieval flag from their source kind."""
    config_file = tmp_path / "wf.yaml"
    config_file.write_text(
        """
data:
  remote_model:
    type: gefs
    kwargs: {start_date: '2025-08-01', end_date: '2025-08-02'}
    zarr_store: {enabled: true, backend: icechunk, icechunk_url: ./store}
  cached_model:
    type: gefs
    existing: true
    kwargs: {files: '/tmp/*.grib2'}
  local_files:
    type: gefs
    kwargs: {files: '/tmp/*.grib2'}
"""
    )
    dag = DAGBuilder(ConfigParser(str(config_file))).build()
    assert dag.nodes["load_remote_model"]["retrieval"] is True
    assert dag.nodes["load_cached_model"]["retrieval"] is False
    assert dag.nodes["load_local_files"]["retrieval"] is False


def test_wait_for_workers_returns_when_resources_present():
    """wait_for_workers returns once all required resources registered."""
    client = MagicMock()
    client.scheduler_info.return_value = {
        "workers": {
            "w1": {"resources": {"COMPUTE": 1}},
            "w2": {"resources": {"SERVICE": 1}},
        }
    }
    wait_for_workers(client, {"COMPUTE": 1, "SERVICE": 1}, timeout=1)
    client.scheduler_info.assert_called()


def test_wait_for_workers_times_on_missing_resource(monkeypatch):
    """wait_for_workers raises TimeoutError naming the missing resource."""
    monkeypatch.setattr("mdt.dask_engine.time.sleep", lambda s: None)
    client = MagicMock()
    client.scheduler_info.return_value = {"workers": {"w1": {"resources": {"COMPUTE": 1}}}}
    with pytest.raises(TimeoutError, match="SERVICE"):
        wait_for_workers(client, {"SERVICE": 1}, timeout=0)


def _local_dag(tmp_path):
    config_file = tmp_path / "wf.yaml"
    config_file.write_text(
        """
data:
  obs:
    type: aeronet
    cluster: compute
    kwargs: {files: dummy}
statistics:
  s:
    input: obs
    metrics: [mb]
    kwargs: {obs_var: o, mod_var: m}
execution:
  orchestrator: dask
  clusters:
    compute: {mode: local, workers: 1}
"""
    )
    config = ConfigParser(str(config_file))
    return DAGBuilder(config).build(), config


def test_dask_engine_executes_local_dag_without_gather(tmp_path, monkeypatch):
    """End-to-end on a local cluster: results stay on workers (no .result())."""
    pytest.importorskip("distributed")
    dag, config = _local_dag(tmp_path)

    obs = xr.Dataset({"o": ("x", [1.0, 2.0]), "m": ("x", [1.5, 2.5])})
    monkeypatch.setattr("mdt.dask_engine.load_data", lambda name, dataset_type, kwargs: obs)

    engine = DaskEngine(dag, config)
    results = engine.execute()

    assert results["load_obs"] == "finished"
    assert results["stats_s"] == "finished"


def test_dask_engine_reports_task_failure(tmp_path, monkeypatch):
    """A failing task surfaces as RuntimeError with the task's error."""
    pytest.importorskip("distributed")
    dag, config = _local_dag(tmp_path)

    def boom(name, dataset_type, kwargs):
        raise ValueError("load exploded")

    monkeypatch.setattr("mdt.dask_engine.load_data", boom)
    engine = DaskEngine(dag, config)
    with pytest.raises(RuntimeError, match="load exploded"):
        engine.execute()
