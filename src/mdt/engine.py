"""Prefect execution engine and task wrappers for MDT."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, cast

import networkx as nx

from mdt.dask_runtime import DaskClusterManager, resolve_task_resource
from mdt.engine_registry import Engine
from mdt.tasks.data import load_data, save_data
from mdt.tasks.pairing import combine_paired_data, pair_data
from mdt.tasks.plotting import generate_plot
from mdt.tasks.reductions import calculate_reduction
from mdt.tasks.statistics import compute_statistics

if TYPE_CHECKING:
    from prefect.futures import PrefectFuture
    from prefect.task_runners import TaskRunner

    from mdt.config import ConfigParser

logger = logging.getLogger(__name__)


def _serialize_local_pairing_tasks(clusters_cfg: Dict[str, Any], pairing_mode: str) -> bool:
    """Return whether local pair tasks must run inline for compatibility."""
    return pairing_mode == "serial" and all(cfg.get("mode", "local") == "local" for cfg in clusters_cfg.values())


def _resolve_task_output(value: Any) -> Any:
    """Resolve nested future inputs without changing xarray objects or model names."""
    if hasattr(value, "result") and callable(value.result):
        value = value.result()
    if isinstance(value, dict):
        return {key: _resolve_task_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_task_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_task_output(item) for item in value)
    return value


def prefect_calculate_reduction(
    name: str,
    data: Any,
    method: str,
    dim: Any,
    force_weighted: bool,
    kwargs: Dict[str, Any],
) -> Any:
    """Prefect task wrapper for calculate_reduction."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Reducing dataset '{name}' using method '{method}' over dim: {dim}")
    return calculate_reduction(data, method, dim, force_weighted, **kwargs)


def prefect_save_data(name: str, data: Any, backend: str, url: str, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for save_data."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Saving dataset '{name}' using backend '{backend}' to '{url}'")
    return save_data(name, data, backend, url, kwargs)


def prefect_load_data(name: str, dataset_type: str, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for load_data."""
    from prefect import get_run_logger

    logger = get_run_logger()
    use_virtualizarr = kwargs.get("use_virtualizarr", False)
    if use_virtualizarr:
        backend = kwargs.get("virtualizarr_backend", "N/A")
        store_path = kwargs.get("store_path", "N/A")
        icechunk_repo = kwargs.get("icechunk_repo", "N/A")
        logger.info(
            f"Loading data with VirtualiZarr: {name} "
            f"(type: {dataset_type}, backend: {backend}, store: {store_path}, icechunk_repo: {icechunk_repo})"
        )
    else:
        logger.info(f"Loading data: {name} of type {dataset_type}")
    return load_data(name, dataset_type, kwargs)


def prefect_pair_data(name: str, method: str, source_data: Any, target_data: Any, kwargs: Dict[str, Any], mask: str | None = None) -> Any:
    """Prefect task wrapper for pair_data."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Pairing data: {name} using {method}")
    return pair_data(name, method, source_data, target_data, kwargs, mask=mask)


def prefect_combine_paired_data(paired_data: Dict[str, Any], dim: str = "model") -> Any:
    """Prefect task wrapper for combine_paired_data."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Combining {len(paired_data)} paired datasets along '{dim}'")
    return combine_paired_data(paired_data, dim=dim)


def prefect_compute_statistics(name: str, metrics: List[str], input_data: Any, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for compute_statistics."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Computing statistics: {name} for metrics {metrics}")
    return compute_statistics(name, metrics, input_data, kwargs)


def prefect_generate_plot(name: str, plot_type: str, input_data: Any, kwargs: Dict[str, Any]) -> Any:
    """Prefect task wrapper for generate_plot."""
    from prefect import get_run_logger

    logger = get_run_logger()
    logger.info(f"Generating plot: {name} of type {plot_type}")
    return generate_plot(name, plot_type, input_data, kwargs)


class PrefectEngine(Engine):
    """Executes a NetworkX DAG as a Prefect Flow."""

    def __init__(self, dag: nx.DiGraph, config: "ConfigParser"):
        self.dag = dag
        self.config = config
        self._manager = DaskClusterManager(config)
        # Shared state with the cluster manager so introspection keeps working.
        self.clusters: Dict[str, Any] = self._manager.clusters
        self._primary_cluster: Any = None
        self._submitted_jobs: List[tuple[str, str]] = self._manager._submitted_jobs

    def close(self) -> None:
        """Close all Dask clusters owned by this engine."""
        self._manager.close()
        self._primary_cluster = None

    def _setup_dask_clusters(self) -> Any:
        """
        Initializes Dask clusters based on the execution configuration.

        It starts a primary LocalCluster (scheduler) and then scales HPC clusters
        which connect their workers back to this primary scheduler.
        Each HPC cluster's workers are tagged with the cluster name using `resources`.

        The topology logic is shared with the plain-Dask engine via
        :class:`mdt.dask_runtime.DaskClusterManager`.
        """
        cluster = self._manager.setup_dask_clusters()
        self._primary_cluster = self._manager._primary_cluster
        return cluster

    def execute(self) -> Dict[str, Any]:
        """Run the flow and close resources even when setup or tasks fail."""
        try:
            return self._execute()
        finally:
            self.close()

    def _execute(self) -> Dict[str, Any]:
        """
        Builds the Dask cluster topology and executes the Prefect flow.

        Returns
        -------
        dict
            A dictionary mapping node IDs from the task graph to the Dask futures
            representing their completion state and results.
        """
        from prefect import flow, task

        # Prefect Task Wrappers — defined here so the @task decorator is only
        # evaluated when Prefect is actually installed and execute() is called.
        # Requirement 3.5: We maintain lazy imports.
        from prefect.cache_policies import NONE as NO_CACHE

        p_load_data = task(name="Load Data", cache_policy=NO_CACHE)(prefect_load_data)
        p_pair_data = task(name="Pair Data", cache_policy=NO_CACHE)(prefect_pair_data)
        p_combine_paired_data = task(name="Combine Paired Data", cache_policy=NO_CACHE)(prefect_combine_paired_data)
        p_compute_statistics = task(name="Compute Statistics", cache_policy=NO_CACHE)(prefect_compute_statistics)
        p_generate_plot = task(name="Generate Plot", cache_policy=NO_CACHE)(prefect_generate_plot)
        p_calculate_reduction = task(name="Calculate Reduction", cache_policy=NO_CACHE)(prefect_calculate_reduction)
        p_save_data = task(name="Save Data", cache_policy=NO_CACHE)(prefect_save_data)

        from contextlib import nullcontext

        import dask

        # Determine if we need Dask resource annotations
        exec_cfg = self.config.execution
        clusters_cfg = exec_cfg.get("clusters", {})
        use_dask_runner = len(clusters_cfg) > 1 or any(
            cfg.get("mode", "local") != "local" or cfg.get("workers", 1) > 1 for cfg in clusters_cfg.values()
        )
        pairing_mode = exec_cfg.get("pairing_mode", "serial")
        serialize_pair_tasks = _serialize_local_pairing_tasks(clusters_cfg, pairing_mode)

        # Snapshot mutable engine state into local variables so the flow closure
        # does not capture `self` (which may hold non-picklable Dask client state).
        dag = self.dag

        # Define the Prefect flow inline to capture the instance variables
        @flow(name="MDT Verification Workflow")  # type: ignore
        def mdt_flow() -> Dict[str, Any]:
            # Dictionary to store the output futures of each task
            task_outputs: Dict[str, Any] = {}
            last_pair_future: Any = None

            # Use topological sort to process nodes in the correct dependency order
            for node_id in nx.topological_sort(dag):
                node_data = dag.nodes[node_id]
                task_type = node_data["task_type"]
                target_cluster = node_data.get("cluster")

                # Retrieve the inputs from the dependencies that have already run
                predecessors = list(dag.predecessors(node_id))

                # Route via shared policy: retrieval loads -> service cluster,
                # cached loads -> execution.cache_cluster when configured,
                # everything else -> the node's cluster (default compute).
                res_key = resolve_task_resource(node_data, exec_cfg)
                logger.debug(f"Routing task '{node_id}' to cluster resource: {res_key}")

                # Route the task to the specific worker pool via Dask Resource Annotation
                # Only use dask.annotate when running with DaskTaskRunner
                annotation_ctx = dask.annotate(resources={res_key: 1}) if use_dask_runner else nullcontext()
                with annotation_ctx:
                    if task_type == "load_data":
                        name = node_data["name"]
                        kwargs = node_data["kwargs"]
                        use_virtualizarr = kwargs.get("use_virtualizarr", False)

                        task_options: Dict[str, Any] = {}
                        if use_virtualizarr:
                            task_options["tags"] = ["virtualizarr"]
                            task_options["task_run_name"] = f"Load Data: {name} [VirtualiZarr]"

                        future = p_load_data.with_options(**task_options).submit(name=name, dataset_type=node_data["dataset_type"], kwargs=kwargs)
                        task_outputs[node_id] = future

                    elif task_type == "pair_data":
                        # Resolve source and target node IDs from DAG metadata
                        source_names = node_data.get("source_names")
                        if not isinstance(source_names, list):
                            single_source = node_data.get("source_name")
                            source_names = [single_source] if isinstance(single_source, str) else []
                        target_name = node_data.get("target_name")

                        # Resolve node IDs from the actual predecessors in the DAG
                        # We look for nodes where the 'name' attribute matches the requested source/target names
                        source_node_ids = {
                            source_name: next((p for p in predecessors if dag.nodes[p].get("name") == source_name), None)
                            for source_name in source_names
                            if isinstance(source_name, str)
                        }
                        target_node_id = next((p for p in predecessors if dag.nodes[p].get("name") == target_name), None)

                        if any(value is None for value in source_node_ids.values()) or not target_node_id:
                            logger.error(
                                f"Failed to resolve predecessors for pairing '{node_id}'. Sources: {source_node_ids}, Target: {target_node_id}"
                            )

                        source_output = {name: task_outputs[node_id] for name, node_id in source_node_ids.items() if node_id is not None}
                        if len(source_output) == 1:
                            source_output = next(iter(source_output.values()))
                        target_output = task_outputs.get(target_node_id) if target_node_id else None

                        if serialize_pair_tasks:
                            if last_pair_future is not None:
                                _resolve_task_output(last_pair_future)

                            # Keep Monet pairing on the flow thread for local runs.
                            # This avoids ESMF VM/thread initialization issues in Prefect workers
                            # while preserving the same pairing implementation path.
                            pair_result = prefect_pair_data(
                                name=node_data["name"],
                                method=node_data["method"],
                                source_data=_resolve_task_output(source_output),
                                target_data=_resolve_task_output(target_output),
                                kwargs={**node_data["kwargs"], "model_dim": node_data.get("model_dim", "model")},
                                mask=node_data.get("mask"),
                            )
                            task_outputs[node_id] = pair_result
                            last_pair_future = pair_result
                        else:
                            future = p_pair_data.submit(
                                name=node_data["name"],
                                method=node_data["method"],
                                source_data=source_output,
                                target_data=target_output,
                                kwargs={**node_data["kwargs"], "model_dim": node_data.get("model_dim", "model")},
                                mask=node_data.get("mask"),
                            )
                            task_outputs[node_id] = future

                    elif task_type == "combine_paired_data":
                        # Collect all inputs from predecessors (pairing tasks)
                        paired_data_inputs = {}
                        # Get the list of pairing names executed
                        sources = node_data.get("sources", [])

                        for source_pair_name in sources:
                            # The node ID is constructed as pair_{name} in DAGBuilder
                            pred_node_id = f"pair_{source_pair_name}"
                            if pred_node_id in task_outputs:
                                # Use the pairing name (or a label) as the key
                                paired_data_inputs[source_pair_name] = task_outputs[pred_node_id]
                            else:
                                logger.warning(f"Predecessor {pred_node_id} not found for combine task {node_id}")

                        future = p_combine_paired_data.submit(
                            paired_data=paired_data_inputs,
                            dim=node_data.get("dim", "model"),
                        )
                        task_outputs[node_id] = future

                    elif task_type == "compute_statistics":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = p_compute_statistics.submit(
                            name=node_data["name"],
                            metrics=node_data["metrics"],
                            input_data=task_outputs.get(input_node) if input_node else None,
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

                    elif task_type == "generate_plot":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = p_generate_plot.submit(
                            name=node_data["name"],
                            plot_type=node_data["plot_type"],
                            input_data=task_outputs.get(input_node) if input_node else None,
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

                    elif task_type == "calculate_reduction":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = p_calculate_reduction.submit(
                            name=node_data["name"],
                            data=task_outputs.get(input_node) if input_node else None,
                            method=node_data["method"],
                            dim=node_data["dim"],
                            force_weighted=node_data["force_weighted"],
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

                    elif task_type == "save_data":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = p_save_data.submit(
                            name=node_data["name"],
                            data=task_outputs.get(input_node) if input_node else None,
                            backend=node_data["backend"],
                            url=node_data["url"],
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

            logger.info("All tasks submitted to Prefect.")
            return task_outputs

        # Setup execution environment
        logger.info("Starting Prefect flow execution...")
        if use_dask_runner:
            # Setup Dask clusters and configure Prefect to use the central scheduler
            from prefect_dask.task_runners import DaskTaskRunner

            cluster = self._setup_dask_clusters()
            logger.info(f"Central Dask Scheduler address: {cluster.scheduler_address}")
            task_futures: Dict[str, Any] = mdt_flow.with_options(task_runner=cast(Any, DaskTaskRunner(address=cluster.scheduler_address)))()
        else:
            # Single local worker — use ConcurrentTaskRunner (no Dask serialization overhead)
            from prefect.task_runners import ConcurrentTaskRunner

            # Prefect's runner generic is invariant, although concurrent futures
            # implement the PrefectFuture interface required by Flow.with_options.
            runner = cast("TaskRunner[PrefectFuture[Any]]", ConcurrentTaskRunner())
            task_futures = mdt_flow.with_options(task_runner=runner)()

        # Requirement: Ensure the CLI waits for task completion.
        # Prefect .submit() returns futures. We need to wait for them.
        logger.info("Waiting for tasks to complete...")
        final_results: Dict[str, Any] = {}
        failed_tasks: Dict[str, Exception] = {}
        for node_id, future in task_futures.items():
            try:
                # This will block until the specific task is done.
                final_results[node_id] = _resolve_task_output(future)
            except Exception as e:
                logger.error(f"Task {node_id} failed: {e}")
                final_results[node_id] = e
                failed_tasks[node_id] = e

        logger.info("All tasks completed.")
        if failed_tasks:
            failure_summary = ", ".join(f"{node_id}: {error}" for node_id, error in failed_tasks.items())
            raise RuntimeError(f"Workflow failed in {len(failed_tasks)} task(s): {failure_summary}") from next(iter(failed_tasks.values()))
        return final_results
