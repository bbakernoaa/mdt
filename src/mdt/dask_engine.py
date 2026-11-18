"""Plain-Dask execution engine for MDT.

Executes the task DAG directly on a Dask distributed cluster without an
orchestration server (no Prefect API/SQLite state). Task results stay on the
workers as Dask futures: datasets are never gathered back to the launching
node, so cached loads (Icechunk/Zarr on shared storage) feed pairing,
statistics and plotting on the same workers via Dask's data-locality policy.
Only remote retrieval needs the service (e.g. DTN) cluster.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict

import networkx as nx
from distributed import wait as dask_wait

from mdt.dask_runtime import DaskClusterManager, resolve_task_resource
from mdt.engine_registry import Engine
from mdt.tasks.data import load_data, save_data
from mdt.tasks.pairing import combine_paired_data, pair_data
from mdt.tasks.plotting import generate_plot
from mdt.tasks.reductions import calculate_reduction
from mdt.tasks.statistics import compute_statistics

if TYPE_CHECKING:
    from mdt.config import ConfigParser

logger = logging.getLogger(__name__)

#: Seconds between polls while waiting for batch workers to register.
_WORKER_POLL_INTERVAL = 5.0


def wait_for_workers(client: Any, required: Dict[str, int], timeout: float = 600.0) -> None:
    """Block until every required cluster resource has enough workers.

    Parameters
    ----------
    client : distributed.Client
        Connected client used to query scheduler worker state.
    required : dict
        Mapping of uppercase resource names to minimum worker counts.
    timeout : float
        Maximum seconds to wait before raising.

    Raises
    ------
    TimeoutError
        If the required workers do not register within ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while True:
        workers = client.scheduler_info().get("workers", {})
        counts: Dict[str, int] = {}
        for worker in workers.values():
            for resource in worker.get("resources", {}):
                counts[resource] = counts.get(resource, 0) + 1
        missing = {name: count - counts.get(name, 0) for name, count in required.items() if counts.get(name, 0) < count}
        if not missing:
            logger.info("All required Dask workers registered: %s", counts)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout:.0f}s waiting for Dask workers with resources: {missing}")
        logger.info("Waiting for Dask workers to register (missing: %s)", missing)
        time.sleep(_WORKER_POLL_INTERVAL)


def _task_call(node_data: Dict[str, Any], predecessors: list, outputs: Dict[str, Any], dag: nx.DiGraph) -> tuple:
    """Map a DAG node to (function, kwargs) with future-valued inputs.

    Inputs are passed as the Dask futures stored in ``outputs``; distributed
    resolves them on the worker before invoking the function, so datasets
    move worker-to-worker (or not at all) instead of through the client.
    """
    task_type = node_data["task_type"]
    input_node = predecessors[0] if predecessors else None

    if task_type == "load_data":
        return load_data, {"name": node_data["name"], "dataset_type": node_data["dataset_type"], "kwargs": node_data["kwargs"]}

    if task_type == "pair_data":
        source_names = node_data.get("source_names")
        if not isinstance(source_names, list):
            single_source = node_data.get("source_name")
            source_names = [single_source] if isinstance(single_source, str) else []
        target_name = node_data.get("target_name")
        source_node_ids = {
            source_name: next((p for p in predecessors if dag.nodes[p].get("name") == source_name), None)
            for source_name in source_names
            if isinstance(source_name, str)
        }
        target_node_id = next((p for p in predecessors if dag.nodes[p].get("name") == target_name), None)
        if any(value is None for value in source_node_ids.values()) or not target_node_id:
            raise ValueError(f"Failed to resolve predecessors for pairing '{node_data['name']}': {source_node_ids}, target={target_node_id}")
        source_output: Any = {name: outputs[nid] for name, nid in source_node_ids.items() if nid is not None}
        if len(source_output) == 1:
            source_output = next(iter(source_output.values()))
        return pair_data, {
            "name": node_data["name"],
            "method": node_data["method"],
            "source_data": source_output,
            "target_data": outputs.get(target_node_id),
            "kwargs": {**node_data["kwargs"], "model_dim": node_data.get("model_dim", "model")},
            "mask": node_data.get("mask"),
        }

    if task_type == "combine_paired_data":
        paired_data_inputs = {}
        for source_pair_name in node_data.get("sources", []):
            pred_node_id = f"pair_{source_pair_name}"
            if pred_node_id in outputs:
                paired_data_inputs[source_pair_name] = outputs[pred_node_id]
            else:
                logger.warning("Predecessor %s not found for combine task '%s'", pred_node_id, node_data["name"])
        return combine_paired_data, {"paired_data": paired_data_inputs, "dim": node_data.get("dim", "model")}

    if task_type == "compute_statistics":
        return compute_statistics, {
            "name": node_data["name"],
            "metrics": node_data["metrics"],
            "input_data": outputs.get(input_node),
            "kwargs": node_data["kwargs"],
        }

    if task_type == "generate_plot":
        return generate_plot, {
            "name": node_data["name"],
            "plot_type": node_data["plot_type"],
            "input_data": outputs.get(input_node),
            "kwargs": node_data["kwargs"],
        }

    if task_type == "calculate_reduction":
        return calculate_reduction, {
            "name": node_data["name"],
            "data": outputs.get(input_node),
            "method": node_data["method"],
            "dim": node_data["dim"],
            "force_weighted": node_data["force_weighted"],
            "kwargs": node_data["kwargs"],
        }

    if task_type == "save_data":
        return save_data, {
            "name": node_data["name"],
            "data": outputs.get(input_node),
            "backend": node_data["backend"],
            "url": node_data["url"],
            "kwargs": node_data["kwargs"],
        }

    raise ValueError(f"Unknown task type '{task_type}'")


#: Load kwargs that enable remote retrieval; stripped in allocation mode so
#: no code path can attempt a download on network-isolated compute nodes.
_RETRIEVAL_KWARGS = ("download", "retries", "backoff_factor", "n_chunks", "max_scan_attempts", "network_timeout")


class DaskEngine(Engine):
    """Executes a NetworkX DAG directly on Dask distributed (no orchestrator server).

    Parameters
    ----------
    dag : networkx.DiGraph
        Task graph to execute.
    config : ConfigParser
        Parsed workflow configuration.
    client : distributed.Client, optional
        External client (fixed-allocation mode via ``mdt.allocation``). When
        provided, the engine does not create clusters, does not annotate
        per-cluster resources (single worker pool), and fail-fast verifies
        that every load can be served from prepared local stores.
    """

    def __init__(self, dag: nx.DiGraph, config: "ConfigParser", client: Any = None):
        self.dag = dag
        self.config = config
        self._external_client = client
        self._manager = DaskClusterManager(config)
        # Attribute aliases kept for parity with PrefectEngine introspection.
        self.clusters = self._manager.clusters
        self._primary_cluster: Any = None
        self._submitted_jobs = self._manager._submitted_jobs

    def close(self) -> None:
        """Close all Dask clusters and cancel submitted worker jobs."""
        self._manager.close()
        self._primary_cluster = None

    def _setup_dask_clusters(self) -> Any:
        """Initialize the shared Dask topology (see DaskClusterManager)."""
        cluster = self._manager.setup_dask_clusters()
        self._primary_cluster = self._manager._primary_cluster
        return cluster

    def execute(self) -> Dict[str, Any]:
        """Run the DAG on Dask and close resources even when tasks fail."""
        from distributed import Client

        exec_cfg = self.config.execution
        allocation_mode = self._external_client is not None

        if allocation_mode:
            # Fixed allocation (FR-009): verify prepared caches before any
            # submission; strip retrieval options and point every load at its
            # prepared store so no code path can attempt a download.
            from mdt.prep import check_caches, store_for_dataset

            check_caches(self.config, self.dag)
            for _, node_data in self.dag.nodes(data=True):
                if node_data.get("task_type") == "load_data":
                    kwargs = dict(node_data.get("kwargs") or {})
                    for key in _RETRIEVAL_KWARGS:
                        kwargs.pop(key, None)
                    store_path, backend = store_for_dataset(self.config, node_data["name"])
                    kwargs["existing_zarr"] = True
                    if backend == "icechunk":
                        kwargs["use_icechunk"] = True
                        kwargs["icechunk_url"] = store_path
                    else:
                        kwargs["store_path"] = store_path
                    node_data["kwargs"] = kwargs
            client = self._external_client
        else:
            cluster = self._setup_dask_clusters()
            client = Client(cluster)

        resources_by_node = {node_id: resolve_task_resource(self.dag.nodes[node_id], exec_cfg) for node_id in self.dag.nodes}
        try:
            if not allocation_mode:
                # Batch worker jobs need time to start; wait for every cluster the
                # DAG routes tasks to, so submissions never queue on absent workers.
                # One worker can run several tasks; require at least one per cluster.
                required = {name: 1 for name in set(resources_by_node.values())}
                wait_for_workers(client, required, timeout=float(exec_cfg.get("worker_wait_timeout", 600)))

            outputs: Dict[str, Any] = {}
            for node_id in nx.topological_sort(self.dag):
                node_data = self.dag.nodes[node_id]
                predecessors = list(self.dag.predecessors(node_id))
                func, call_kwargs = _task_call(node_data, predecessors, outputs, self.dag)
                key = f"{node_data['task_type']}-{node_id}"
                if allocation_mode:
                    # Single worker pool inside the allocation: per-cluster
                    # resource tags would never be satisfied (research.md R1).
                    future = client.submit(func, key=key, **call_kwargs)
                    logger.info("Submitted task '%s' (allocation pool)", node_id)
                else:
                    resource = resources_by_node[node_id]
                    future = client.submit(func, resources={resource: 1}, key=key, **call_kwargs)
                    logger.info("Submitted task '%s' to cluster resource %s", node_id, resource)
                outputs[node_id] = future

            # Wait for completion without gathering datasets to this node.
            dask_wait(list(outputs.values()))
            results: Dict[str, Any] = {}
            failed: Dict[str, Exception] = {}
            for node_id, future in outputs.items():
                if future.status == "error":
                    exc = future.exception()
                    logger.error("Task %s failed: %s", node_id, exc)
                    results[node_id] = exc
                    failed[node_id] = exc
                else:
                    # Keep results on workers; only record completion state.
                    results[node_id] = future.status
            if failed:
                summary = ", ".join(f"{node_id}: {error}" for node_id, error in failed.items())
                raise RuntimeError(f"Workflow failed in {len(failed)} task(s): {summary}") from next(iter(failed.values()))
            logger.info("All tasks completed.")
            return results
        finally:
            if not allocation_mode:
                client.close()
                self.close()