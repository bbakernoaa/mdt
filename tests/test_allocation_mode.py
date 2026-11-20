"""Fixed-allocation execution tests (feature 001-prep-fixed-allocation, US2).

Scenarios per specs/001-prep-fixed-allocation/quickstart.md §A:
- env validation / layout adaptation (FR-10, SC-007)
- no batch submissions (FR-007, SC-003)
- fail-fast missing cache (FR-009, SC-007)
- SLURMRunner wiring (research.md R1)
"""

from unittest.mock import MagicMock

import pytest

from mdt.allocation import adapt_layout, detect_allocation
from mdt.config import ConfigParser
from mdt.dag import DAGBuilder
from mdt.dask_runtime import DaskClusterManager
from mdt.prep import PrepManifest

ALLOCATION_YAML = """\
data:
  obs:
    type: aeronet
    cluster: compute
    kwargs: {files: dummy_obs}
statistics:
  s:
    input: obs
    metrics: [mb]
    kwargs: {obs_var: o, mod_var: m}
execution:
  orchestrator: dask
  dask_interface: hsn0
  allocation:
    nodes: 1
    ntasks_per_node: 8
    cpus_per_task: 8
    memory: 200GB
    account: bil-fire3
  clusters:
    compute: {mode: allocation}
"""


def _config(tmp_path, body=ALLOCATION_YAML):
    path = tmp_path / "wf.yaml"
    path.write_text(body)
    return ConfigParser(str(path))


SLURM_ENV = {
    "SLURM_JOB_ID": "12345",
    "SLURM_NTASKS": "8",
    "SLURM_PROCID": "1",
    "SLURM_JOB_NUM_NODES": "1",
    "SLURM_CPUS_PER_TASK": "8",
}


# --- T017: env detection + layout adaptation (FR-010, SC-007) ---


def test_detect_allocation_reads_slurm_env():
    """SLURM env vars parse into AllocationInfo with n_workers = ntasks - 2."""
    info = detect_allocation(env=SLURM_ENV)
    assert info.job_id == 12345
    assert info.ntasks == 8
    assert info.proc_id == 1
    assert info.num_nodes == 1
    assert info.cpus_per_task == 8
    assert info.n_workers == 6  # ntasks - scheduler - client


@pytest.mark.parametrize("missing", ["SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_PROCID"])
def test_detect_allocation_missing_env_errors(missing):
    """Missing required SLURM variables raise an actionable error naming them."""
    env = dict(SLURM_ENV)
    del env[missing]
    with pytest.raises(RuntimeError, match=missing):
        detect_allocation(env=env)


def test_detect_allocation_world_too_small():
    """Allocations with fewer than 3 tasks fail fast (scheduler + client + worker)."""
    env = dict(SLURM_ENV, SLURM_NTASKS="2")
    with pytest.raises(RuntimeError, match="at least 3"):
        detect_allocation(env=env)


def test_adapt_layout_clamps_to_allocation(caplog):
    """Over-requested workers are clamped to the allocation with a warning."""
    info = detect_allocation(env=dict(SLURM_ENV, SLURM_NTASKS="4"))  # 2 workers available
    layout = adapt_layout(info, requested_workers=8, cpus_per_task=8)
    assert layout["n_workers"] == 2
    assert layout["nthreads"] == 8
    assert any("clamp" in r.message.lower() or "allocat" in r.message.lower() for r in caplog.records)


def test_adapt_layout_no_clamp_when_fits():
    """Requests that fit the allocation are honored unchanged."""
    info = detect_allocation(env=SLURM_ENV)  # 6 workers
    layout = adapt_layout(info, requested_workers=4, cpus_per_task=8)
    assert layout["n_workers"] == 4


# --- T018: no submissions + routing (FR-007, SC-003) ---


def test_allocation_mode_never_submits_batch_jobs(tmp_path, monkeypatch):
    """Direct setup in allocation mode must not sbatch; it directs to the runner entry."""
    config = _config(tmp_path)
    manager = DaskClusterManager(config)
    submit = MagicMock()
    monkeypatch.setattr("mdt.dask_runtime.subprocess.run", submit)
    with pytest.raises(RuntimeError, match="mdt.allocation"):
        manager.setup_dask_clusters()
    submit.assert_not_called()
    manager.close()


def test_allocation_engine_omits_resource_annotations(tmp_path, monkeypatch):
    """With an external runner client, submits carry no per-cluster resources."""
    from mdt.dask_engine import DaskEngine

    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    dag = DAGBuilder(config).build()
    # satisfy cache check
    store = tmp_path / "zarr_stores" / "obs_prep.zarr"
    store.mkdir(parents=True)
    m = PrepManifest.begin("obs", str(store), "zarr", "fp", [])
    import mdt.prep as prep_mod

    monkeypatch.setattr(prep_mod, "dataset_fingerprint", lambda c, n: "fp")
    m.complete(covered_dates=[])

    client = MagicMock()
    future = MagicMock(status="finished")
    client.submit.return_value = future
    monkeypatch.setattr("mdt.dask_engine.dask_wait", lambda futures: None, raising=False)
    import mdt.dask_engine as de

    monkeypatch.setattr(de, "wait_for_workers", MagicMock())
    engine = DaskEngine(dag, config, client=client)
    monkeypatch.setattr(engine, "_setup_dask_clusters", MagicMock())
    results = engine.execute()
    assert client.submit.call_count == len(dag.nodes)
    for call in client.submit.call_args_list:
        assert "resources" not in call.kwargs
    assert all(v == "finished" for v in results.values())


# --- T019: fail-fast cache check (FR-009, SC-007) ---


def test_allocation_engine_fails_fast_before_submit(tmp_path, monkeypatch):
    """Missing caches abort before any task submission, naming mdt prep."""
    from mdt.dask_engine import DaskEngine

    config = _config(tmp_path)
    dag = DAGBuilder(config).build()
    client = MagicMock()
    engine = DaskEngine(dag, config, client=client)
    monkeypatch.setattr(engine, "_setup_dask_clusters", MagicMock())
    with pytest.raises(RuntimeError, match="mdt prep"):
        engine.execute()
    client.submit.assert_not_called()


def test_allocation_engine_strips_retrieval_kwargs(tmp_path, monkeypatch):
    """Load kwargs that enable remote retrieval are removed in allocation mode."""
    from mdt.dask_engine import DaskEngine

    body = ALLOCATION_YAML.replace("kwargs: {files: dummy_obs}", "kwargs: {files: dummy_obs, download: true, retries: 8}")
    config = _config(tmp_path, body)
    monkeypatch.chdir(tmp_path)
    dag = DAGBuilder(config).build()
    store = tmp_path / "zarr_stores" / "obs_prep.zarr"
    store.mkdir(parents=True)
    import mdt.prep as prep_mod

    monkeypatch.setattr(
        prep_mod, "dataset_fingerprint", lambda c, n: prep_mod.config_fingerprint("aeronet", {"files": "dummy_obs", "download": True, "retries": 8})
    )
    m = PrepManifest.begin("obs", str(store), "zarr", prep_mod.dataset_fingerprint(config, "obs"), [])
    m.complete(covered_dates=[])

    client = MagicMock()
    client.submit.return_value = MagicMock(status="finished")
    import mdt.dask_engine as de

    monkeypatch.setattr(de, "wait_for_workers", MagicMock())
    monkeypatch.setattr(de, "dask_wait", lambda futures: None)
    engine = DaskEngine(dag, config, client=client)
    monkeypatch.setattr(engine, "_setup_dask_clusters", MagicMock())
    engine.execute()
    load_calls = [c for c in client.submit.call_args_list if c.kwargs.get("dataset_type") == "aeronet"]
    assert load_calls, "expected a load submit"
    kwargs = load_calls[0].kwargs["kwargs"]
    assert "download" not in kwargs
    assert "retries" not in kwargs
    assert kwargs.get("existing_zarr") is True


# --- T020: SLURMRunner wiring (research.md R1) ---


def test_runner_entry_wires_slurmrunner(tmp_path, monkeypatch):
    """The entry point wires SLURMRunner options and runs the engine on the client rank."""
    import mdt.allocation as alloc

    config_path = tmp_path / "wf.yaml"
    config_path.write_text(ALLOCATION_YAML)
    monkeypatch.chdir(tmp_path)

    runner_cls = MagicMock()
    runner_instance = runner_cls.return_value
    runner_instance.scheduler_address = "tcp://127.0.0.1:8786"
    monkeypatch.setattr(alloc, "SLURMRunner", runner_cls)
    client_cls = MagicMock()
    monkeypatch.setattr(alloc, "Client", client_cls)
    engine_cls = MagicMock()
    engine_cls.return_value.execute.return_value = {"load_obs": "finished"}
    monkeypatch.setattr(alloc, "DaskEngine", engine_cls)
    monkeypatch.setattr(alloc, "check_caches", MagicMock())
    for key, value in SLURM_ENV.items():
        monkeypatch.setenv(key, value)

    alloc.main([str(config_path)])

    kwargs = runner_cls.call_args.kwargs
    assert "scheduler-12345.json" in str(kwargs.get("scheduler_file", ""))
    worker_options = kwargs.get("worker_options", {})
    assert worker_options.get("interface") == "hsn0"
    assert worker_options.get("nthreads") == 8
    # client rank connected and ran the engine with the external client
    client_cls.assert_called_once()
    assert engine_cls.call_args.kwargs.get("client") is client_cls.return_value
    engine_cls.return_value.execute.assert_called_once()
