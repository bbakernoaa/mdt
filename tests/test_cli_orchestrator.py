"""Unit tests for CLI orchestrator selection logic.

Validates: Requirements 7.1, 7.2, 7.3
"""

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
import yaml


@pytest.fixture
def mock_dag():
    """Return a trivial DAG for CLI tests."""
    return nx.DiGraph()


@pytest.fixture
def mock_config_factory():
    """Factory that builds a mock ConfigParser with a configurable orchestrator."""

    def _make(orchestrator="prefect"):
        cfg = MagicMock()
        cfg.orchestrator = orchestrator
        return cfg

    return _make


@pytest.fixture
def mock_engine_class():
    """Return a mock engine class whose instances have an .execute() method."""
    engine_cls = MagicMock()
    engine_cls.return_value.execute.return_value = {"status": "ok"}
    return engine_cls


class TestCLIOrchestratorSelection:
    """Verify the CLI resolves the orchestrator in the correct priority order."""

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_cli_arg_overrides_yaml_config(
        self, mock_load_config, mock_dag_builder, mock_registry, mock_config_factory, mock_dag, mock_engine_class
    ):
        """CLI --orchestrator ecflow overrides YAML config that says prefect."""
        # YAML says "prefect", CLI says "ecflow" → ecflow wins
        config = mock_config_factory(orchestrator="prefect")
        mock_load_config.return_value = config
        mock_dag_builder.return_value.build.return_value = mock_dag
        mock_registry.get_engine.return_value = mock_engine_class

        with patch("sys.argv", ["mdt", "run", "config.yaml", "--orchestrator", "ecflow"]):
            from mdt.cli import main

            main()

        mock_registry.get_engine.assert_called_once_with("ecflow")

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_yaml_config_used_when_cli_arg_absent(
        self, mock_load_config, mock_dag_builder, mock_registry, mock_config_factory, mock_dag, mock_engine_class
    ):
        """No --orchestrator flag → YAML config value is used."""
        config = mock_config_factory(orchestrator="ecflow")
        mock_load_config.return_value = config
        mock_dag_builder.return_value.build.return_value = mock_dag
        mock_registry.get_engine.return_value = mock_engine_class

        with patch("sys.argv", ["mdt", "run", "config.yaml"]):
            from mdt.cli import main

            main()

        mock_registry.get_engine.assert_called_once_with("ecflow")

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_default_falls_back_to_prefect(
        self, mock_load_config, mock_dag_builder, mock_registry, mock_config_factory, mock_dag, mock_engine_class
    ):
        """No --orchestrator flag and YAML defaults to 'prefect' → prefect is used."""
        config = mock_config_factory(orchestrator="prefect")
        mock_load_config.return_value = config
        mock_dag_builder.return_value.build.return_value = mock_dag
        mock_registry.get_engine.return_value = mock_engine_class

        with patch("sys.argv", ["mdt", "run", "config.yaml"]):
            from mdt.cli import main

            main()

        mock_registry.get_engine.assert_called_once_with("prefect")

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_successful_run_uses_normal_shutdown(self, mock_load, mock_builder, mock_registry, mock_config_factory, mock_dag, mock_engine_class):
        mock_load.return_value = mock_config_factory()
        mock_builder.return_value.build.return_value = mock_dag
        mock_registry.get_engine.return_value = mock_engine_class
        with patch("os._exit") as forced_exit, patch("sys.argv", ["mdt", "run", "config.yaml"]):
            from mdt.cli import main

            main()
        forced_exit.assert_not_called()
        mock_engine_class.return_value.close.assert_called_once()

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_run_failure_returns_nonzero(self, mock_load, mock_builder, mock_registry, mock_config_factory, mock_dag):
        mock_load.return_value = mock_config_factory()
        mock_builder.return_value.build.return_value = mock_dag
        engine_class = MagicMock()
        engine_class.return_value.execute.side_effect = RuntimeError("failed node pair_bad")
        mock_registry.get_engine.return_value = engine_class
        from mdt.cli import main

        with patch("sys.argv", ["mdt", "run", "config.yaml"]), pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    @patch("mdt.cli.EngineRegistry")
    @patch("mdt.cli.DAGBuilder")
    @patch("mdt.cli.load_config")
    def test_validate_does_not_initialize_engine(self, mock_load, mock_builder, mock_registry, mock_config_factory, mock_dag):
        mock_load.return_value = mock_config_factory()
        mock_builder.return_value.build.return_value = mock_dag
        with patch("sys.argv", ["mdt", "validate", "config.yaml"]):
            from mdt.cli import main

            main()
        mock_registry.get_engine.assert_not_called()

    def test_template_is_valid_yaml_with_local_defaults(self, tmp_path):
        target = tmp_path / "template.yaml"
        with patch("sys.argv", ["mdt", "template", "--output", str(target)]):
            from mdt.cli import main

            main()
        config = yaml.safe_load(target.read_text())
        assert config["execution"]["clusters"]["local"]["workers"] == 1
        assert config["execution"]["pairing_mode"] == "serial"

    def test_pairing_mode_selects_serial_or_concurrent_local_path(self):
        from mdt.engine import _serialize_local_pairing_tasks

        local_clusters = {"compute": {"mode": "local"}, "service": {"mode": "local"}}
        assert _serialize_local_pairing_tasks(local_clusters, "serial") is True
        assert _serialize_local_pairing_tasks(local_clusters, "concurrent") is False
        assert _serialize_local_pairing_tasks({"compute": {"mode": "slurm"}}, "serial") is False

    def test_cluster_setup_does_not_mutate_nested_cluster_kwargs(self, monkeypatch):
        dask_distributed = pytest.importorskip("dask.distributed")

        from mdt.engine import PrefectEngine

        cluster_kwargs = {"env_extra": ["export USER_FLAG=1"]}
        original = {"env_extra": ["export USER_FLAG=1"]}
        execution = {
            "clusters": {
                "compute": {"mode": "local", "workers": 2, "cluster_kwargs": cluster_kwargs},
                "service": {"mode": "local", "workers": 1},
            }
        }
        config = MagicMock(execution=execution)
        cluster = MagicMock(scheduler_address="tcp://127.0.0.1:8786", new_spec={"cls": object, "options": {}}, worker_spec={})
        local_cluster = MagicMock(return_value=cluster)
        monkeypatch.setattr(dask_distributed, "LocalCluster", local_cluster)
        engine = PrefectEngine(nx.DiGraph(), config)

        engine._setup_dask_clusters()

        assert cluster_kwargs == original
        assert local_cluster.call_args.kwargs["n_workers"] == 0
        # Workers are spawned via managed specs carrying per-cluster resources.
        assert [spec["options"]["resources"] for spec in cluster.worker_spec.values()] == [
            {"COMPUTE": 1},
            {"COMPUTE": 1},
            {"SERVICE": 1},
        ]
        cluster.scale.assert_called_once_with(3)
        engine.close()

    def test_mixed_local_and_hpc_clusters_start_tagged_local_workers(self, monkeypatch):
        distributed = pytest.importorskip("dask.distributed")
        from mdt.engine import PrefectEngine

        scheduler = MagicMock(scheduler_address="tcp://127.0.0.1:8786", new_spec={"cls": object, "options": {}}, worker_spec={})
        monkeypatch.setattr(distributed, "LocalCluster", MagicMock(return_value=scheduler))
        cfg = MagicMock(
            execution={
                "clusters": {
                    "service": {"mode": "local", "workers": 2},
                    "compute": {"mode": "slurm", "workers": 1},
                }
            }
        )
        engine = PrefectEngine(nx.DiGraph(), cfg)
        hpc_cluster = MagicMock(job_script=MagicMock(return_value="script"), scheduler_address="dummy")
        monkeypatch.setattr("mdt.dask_runtime.HPCProfileFactory.create_cluster", MagicMock(return_value=hpc_cluster))
        monkeypatch.setattr("mdt.dask_runtime.subprocess.run", MagicMock(return_value=MagicMock(stdout="12345\n")))

        engine._setup_dask_clusters()

        assert len(scheduler.worker_spec) == 2
        assert all(spec["options"]["resources"] == {"SERVICE": 1} for spec in scheduler.worker_spec.values())
        engine.close()

    def test_dask_interface_propagates_to_scheduler_and_hpc_workers(self, monkeypatch):
        """execution.dask_interface pins scheduler and worker networks (e.g. Gaea hsn0).

        The scheduler must advertise the chosen interface (not the default
        front-end address) and HPC workers inherit it unless overridden.
        """
        distributed = pytest.importorskip("dask.distributed")
        from mdt.engine import PrefectEngine

        scheduler = MagicMock(scheduler_address="tcp://172.25.95.5:8786")
        local_cluster = MagicMock(return_value=scheduler)
        monkeypatch.setattr(distributed, "LocalCluster", local_cluster)
        hpc_cluster = MagicMock(job_script=MagicMock(return_value="script"), scheduler_address="dummy")
        factory = MagicMock(return_value=hpc_cluster)
        monkeypatch.setattr("mdt.dask_runtime.HPCProfileFactory.create_cluster", factory)
        monkeypatch.setattr("mdt.dask_runtime.subprocess.run", MagicMock(return_value=MagicMock(stdout="12345\n")))
        cfg = MagicMock(
            execution={
                "dask_interface": "hsn0",
                "clusters": {
                    "gaea_dtn": {"mode": "gaea", "workers": 1},
                    "service": {"mode": "gaea", "workers": 1, "cluster_kwargs": {"interface": "bond0"}},
                },
            }
        )
        engine = PrefectEngine(nx.DiGraph(), cfg)

        engine._setup_dask_clusters()

        assert local_cluster.call_args.kwargs.get("interface") == "hsn0"
        assert "host" not in local_cluster.call_args.kwargs
        by_name = {call.kwargs["cluster_name"]: call.kwargs for call in factory.call_args_list}
        assert by_name["gaea_dtn"]["interface"] == "hsn0"
        assert by_name["service"]["interface"] == "bond0"  # explicit override wins
        engine.close()

    def test_hpc_cluster_merges_prologue_and_env_extra_with_resource_tag(self, monkeypatch):
        """Resource tags must be case-preserving and survive user prologues.

        DASK_DISTRIBUTED__WORKER__RESOURCES__<NAME> env vars get lowercased by
        dask.config, breaking uppercase annotations, so tags go through the
        worker CLI's --resources flag. dask-jobqueue >= 0.8 ignores env_extra
        whenever job_script_prologue is present, so both must be merged.
        """
        distributed = pytest.importorskip("dask.distributed")
        from mdt.engine import PrefectEngine

        scheduler = MagicMock(scheduler_address="tcp://127.0.0.1:8786")
        monkeypatch.setattr(distributed, "LocalCluster", MagicMock(return_value=scheduler))
        hpc_cluster = MagicMock(job_script=MagicMock(return_value="script"), scheduler_address="dummy")
        factory = MagicMock(return_value=hpc_cluster)
        monkeypatch.setattr("mdt.dask_runtime.HPCProfileFactory.create_cluster", factory)
        monkeypatch.setattr("mdt.dask_runtime.subprocess.run", MagicMock(return_value=MagicMock(stdout="12345\n")))
        cluster_kwargs = {
            "job_script_prologue": ["cd /repo"],
            "env_extra": ["export USER_FLAG=1"],
        }
        original = {
            "job_script_prologue": ["cd /repo"],
            "env_extra": ["export USER_FLAG=1"],
        }
        cfg = MagicMock(execution={"clusters": {"gaea_dtn": {"mode": "gaea", "workers": 1, "cluster_kwargs": cluster_kwargs}}})
        engine = PrefectEngine(nx.DiGraph(), cfg)

        engine._setup_dask_clusters()

        kwargs = factory.call_args.kwargs
        assert "env_extra" not in kwargs
        assert kwargs["job_script_prologue"] == ["cd /repo", "export USER_FLAG=1"]
        # Case-preserving resource tag via the worker CLI (env vars get lowercased).
        assert kwargs["worker_extra_args"][-2:] == ["--resources", "GAEA_DTN=1"]
        # Caller's kwargs must not be mutated.
        assert cluster_kwargs == original
        engine.close()

    def test_hpc_worker_jobs_get_unique_names(self, monkeypatch):
        """Identical generated scripts must not reuse worker names across jobs.

        The scheduler rejects duplicate worker names ('name taken'), killing
        every job after the first when a cluster submits multiple workers.
        """
        distributed = pytest.importorskip("dask.distributed")
        from mdt.engine import PrefectEngine

        scheduler = MagicMock(scheduler_address="tcp://127.0.0.1:8786")
        monkeypatch.setattr(distributed, "LocalCluster", MagicMock(return_value=scheduler))
        hpc_cluster = MagicMock(
            job_script=MagicMock(return_value="dask-worker tcp://1.2.3.4:9999 --name dummy-name --nworkers 2"),
            scheduler_address="tcp://1.2.3.4:9999",
        )
        monkeypatch.setattr("mdt.dask_runtime.HPCProfileFactory.create_cluster", MagicMock(return_value=hpc_cluster))
        submit = MagicMock(return_value=MagicMock(stdout="12345\n"))
        monkeypatch.setattr("mdt.dask_runtime.subprocess.run", submit)
        cfg = MagicMock(execution={"clusters": {"compute": {"mode": "slurm", "workers": 2}}})
        engine = PrefectEngine(nx.DiGraph(), cfg)

        engine._setup_dask_clusters()

        scripts = [call.kwargs["input"] for call in submit.call_args_list]
        assert "--name compute-0" in scripts[0]
        assert "--name compute-1" in scripts[1]
        assert "--name dummy-name" not in scripts[0]
        engine.close()

    def test_partial_hpc_submission_cancels_submitted_jobs_and_closes_resources(self, monkeypatch):
        distributed = pytest.importorskip("dask.distributed")
        from mdt.engine import PrefectEngine

        scheduler = MagicMock(scheduler_address="tcp://127.0.0.1:8786")
        monkeypatch.setattr(distributed, "LocalCluster", MagicMock(return_value=scheduler))
        hpc_cluster = MagicMock(job_script=MagicMock(return_value="script"), scheduler_address="dummy")
        monkeypatch.setattr("mdt.dask_runtime.HPCProfileFactory.create_cluster", MagicMock(return_value=hpc_cluster))
        submit = MagicMock(side_effect=[MagicMock(stdout="12345\n"), __import__("subprocess").CalledProcessError(1, "sbatch")])
        monkeypatch.setattr("mdt.dask_runtime.subprocess.run", submit)
        cfg = MagicMock(execution={"clusters": {"compute": {"mode": "slurm", "workers": 2}}})
        engine = PrefectEngine(nx.DiGraph(), cfg)

        with pytest.raises(RuntimeError, match="Failed to submit HPC worker job"):
            engine._setup_dask_clusters()

        assert [call.args[0] for call in submit.call_args_list] == [
            ["sbatch", "--parsable"],
            ["sbatch", "--parsable"],
            ["scancel", "12345"],
        ]
        scheduler.close.assert_called_once()
        hpc_cluster.close.assert_called_once()
