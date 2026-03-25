"""Prefect execution engine and task wrappers for MDT."""

import logging

import dask
import dask.distributed
import networkx as nx
from prefect import flow, get_run_logger, task
from prefect_dask.task_runners import DaskTaskRunner

from mdt.hpc import HPCProfileFactory
from mdt.tasks.data import load_data
from mdt.tasks.pairing import pair_data
from mdt.tasks.plotting import generate_plot
from mdt.tasks.statistics import compute_statistics

logger = logging.getLogger(__name__)


# Prefect Task Wrappers
@task(name="Load Data")
def prefect_load_data(name, dataset_type, kwargs):
    """
    Prefect wrapper for the load_data function.

    Parameters
    ----------
    name : str
        The configuration identifier for this data.
    dataset_type : str
        The name of the monetio dataset (e.g., 'cmaq', 'aeronet', 'gcafs').
    kwargs : dict
        Additional keyword arguments to pass to the dataset reader.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        The loaded data object.
    """
    logger = get_run_logger()
    logger.info(f"Loading data: {name} of type {dataset_type}")
    return load_data(name, dataset_type, kwargs)


@task(name="Pair Data")
def prefect_pair_data(name, method, source_data, target_data, kwargs):
    """
    Prefect wrapper for the pair_data function.

    Parameters
    ----------
    name : str
        Identifier for this pairing task.
    method : str
        The method to use (e.g., 'interpolate', 'regrid', 'point_to_grid').
    source_data : xarray.Dataset or pandas.DataFrame
        The source data object (typically a model).
    target_data : xarray.Dataset or pandas.DataFrame
        The target data object or grid.
    kwargs : dict
        Additional arguments to pass to the pairing function.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        The paired dataset object.
    """
    logger = get_run_logger()
    logger.info(f"Pairing data: {name} using {method}")
    return pair_data(name, method, source_data, target_data, kwargs)


@task(name="Compute Statistics")
def prefect_compute_statistics(name, metrics, input_data, kwargs):
    """
    Prefect wrapper for the compute_statistics function.

    Parameters
    ----------
    name : str
        Identifier for this statistics task.
    metrics : list of str
        A list of metric names to compute (e.g., ['rmse', 'bias', 'corr']).
    input_data : pandas.DataFrame or xarray.Dataset
        The paired dataset to analyze.
    kwargs : dict
        Additional arguments passed to the monet_stats functions.

    Returns
    -------
    dict
        A mapping of computed metric names to their results.
    """
    logger = get_run_logger()
    logger.info(f"Computing statistics: {name} for metrics {metrics}")
    return compute_statistics(name, metrics, input_data, kwargs)


@task(name="Generate Plot")
def prefect_generate_plot(name, plot_type, input_data, kwargs):
    """
    Prefect wrapper for the generate_plot function.

    Parameters
    ----------
    name : str
        Identifier for this plotting task.
    plot_type : str
        Type of plot (e.g., 'spatial', 'scatter', 'timeseries').
    input_data : xarray.Dataset, pandas.DataFrame, or dict
        The data to plot.
    kwargs : dict
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    object
        The plot or figure object.
    """
    logger = get_run_logger()
    logger.info(f"Generating plot: {name} of type {plot_type}")
    return generate_plot(name, plot_type, input_data, kwargs)


class PrefectEngine:
    """Executes a NetworkX DAG as a Prefect Flow."""

    def __init__(self, dag, config):
        self.dag = dag
        self.config = config
        self.client = None
        self.clusters = {}

    def _setup_dask_clusters(self):
        """
        Initializes Dask clusters based on the execution configuration.

        It starts a primary LocalCluster (scheduler) and then scales HPC clusters
        which connect their workers back to this primary scheduler.
        Each HPC cluster's workers are tagged with the cluster name using `resources`.
        """
        exec_cfg = self.config.execution
        clusters_cfg = exec_cfg.get("clusters", {})

        import subprocess

        # If there's only one cluster and it's local, keep it simple
        if len(clusters_cfg) == 1 and list(clusters_cfg.values())[0].get("mode", "local") == "local":
            logger.info("Setting up a single local Dask cluster.")
            cluster_name = list(clusters_cfg.keys())[0]
            workers = list(clusters_cfg.values())[0].get("workers", 1)
            # Assign the requested numeric resource to the local workers to prevent hanging
            return dask.distributed.LocalCluster(n_workers=workers, resources={cluster_name.upper(): 1})

        # Multi-cluster or HPC mode:
        # Create a central scheduler on the node where MDT is executed.
        logger.info("Setting up central Dask Scheduler.")
        primary_cluster = dask.distributed.LocalCluster(
            n_workers=0,  # The local cluster is just the scheduler
            host="0.0.0.0",  # Allow external connections
        )
        scheduler_address = primary_cluster.scheduler_address
        self.client = dask.distributed.Client(primary_cluster)

        for cluster_name, cfg in clusters_cfg.items():
            mode = cfg.get("mode", "local")
            workers = cfg.get("workers", 1)
            kwargs = cfg.get("cluster_kwargs", {})

            # Assign resource annotations so Dask knows which workers belong to which cluster.
            res_name = cluster_name.upper()

            if mode == "local":
                logger.info(f"Scaling local workers for '{cluster_name}'")
                # Scale the central cluster and explicitly assign resources
                primary_cluster.scale(workers, resources={res_name: 1})
            else:
                logger.info(f"Setting up HPC cluster '{cluster_name}' (mode: {mode}) connecting to {scheduler_address}")

                # To support multiple disparate HPC clusters sharing a single DAG and scheduler,
                # we bypass `dask-jobqueue`'s automatic localized scheduler creation to avoid
                # port conflicts ("Address already in use"). We construct an isolated jobqueue
                # object on a random port, extract its generated batch script template, inject
                # our central scheduler address, and manually submit the job script.

                # Tag the workers with this cluster's name as a numeric resource
                if "env_extra" not in kwargs:
                    kwargs["env_extra"] = []
                kwargs["env_extra"].append(f"export DASK_DISTRIBUTED__WORKER__RESOURCES__{res_name}=1")

                # Force the dummy scheduler to a random port to avoid conflicts
                kwargs["scheduler_options"] = {"port": 0}

                # Create the JobQueue object representing this specific profile/queue
                hpc_cluster = HPCProfileFactory.create_cluster(mode, **kwargs)

                # Retrieve the underlying job script formatted for the specific batch system
                job_script = hpc_cluster.job_script()

                # Dask job scripts contain a variable representing the scheduler it connects to.
                # E.g., `dask-worker tcp://127.0.0.1:8786 ...`
                # We replace this dummy scheduler address with our central primary scheduler address.
                dummy_address = hpc_cluster.scheduler_address
                job_script = job_script.replace(dummy_address, scheduler_address)

                # Determine the submission command based on the mode (slurm -> sbatch, pbs -> qsub)
                submit_cmd = "sbatch" if "slurm" in mode or mode in ["hera", "jet", "orion", "hercules", "gaea", "ursa"] else "qsub"
                if mode == "lsf":
                    submit_cmd = "bsub"

                # Manually submit the modified script for the requested number of workers
                logger.info(f"Submitting {workers} {submit_cmd} jobs to '{cluster_name}' queue.")
                for _ in range(workers):
                    try:
                        subprocess.run([submit_cmd], input=job_script.encode("utf-8"), check=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to submit HPC worker job for cluster {cluster_name}: {e}")
                        raise

                self.clusters[cluster_name] = hpc_cluster

        return primary_cluster

    def execute(self):
        """
        Builds the Dask cluster topology and executes the Prefect flow.

        Returns
        -------
        dict
            A dictionary mapping node IDs from the task graph to the Dask futures
            representing their completion state and results.
        """

        # Define the Prefect flow inline to capture the instance variables
        @flow(name="MDT Verification Workflow")
        def mdt_flow():
            logger = get_run_logger()
            logger.info("Starting MDT Workflow...")

            # Dictionary to store the output futures of each task
            task_outputs = {}

            # Use topological sort to process nodes in the correct dependency order
            for node_id in nx.topological_sort(self.dag):
                node_data = self.dag.nodes[node_id]
                task_type = node_data["task_type"]
                target_cluster = node_data.get("cluster")

                # Retrieve the inputs from the dependencies that have already run
                predecessors = list(self.dag.predecessors(node_id))

                # We will use Prefect's `.with_options` to inject standard Dask resources.
                # If target_cluster is None, default to "COMPUTE"
                res_key = target_cluster.upper() if target_cluster else "COMPUTE"

                # In Prefect, we can't directly inject `dask_resources` into `.submit()`,
                # but we can use `dask.annotate` if we map it to `client.submit` directly.
                # To maintain Prefect orchestration and Dask scaling, we will
                # use `dask.annotate` and rely on Prefect picking up the context.
                # To fix the context loss, we wrap the native python functions as `dask.delayed`
                # or just use `prefect_task.with_options(tags=[res_key]).submit()`.

                # For robust HPC mapping, let's submit natively to Dask while tracking:

                # Route the task to the specific worker pool via Dask Resource Annotation
                with dask.annotate(resources={res_key: 1}):
                    if task_type == "load_data":
                        future = prefect_load_data.submit(
                            name=node_data["name"], dataset_type=node_data["dataset_type"], kwargs=node_data["kwargs"]
                        )
                        task_outputs[node_id] = future

                    elif task_type == "pair_data":
                        # Find which predecessor is source and which is target
                        # Based on our DAG builder, predecessors are the load nodes for source and target
                        source_node = None
                        target_node = None
                        pairing_name = node_data["name"]
                        pairing_details = self.config.pairing.get(pairing_name, {})
                        source_name = pairing_details.get("source")
                        target_name = pairing_details.get("target")

                        if source_name:
                            source_node = f"load_{source_name}"
                        if target_name:
                            target_node = f"load_{target_name}"

                        future = prefect_pair_data.submit(
                            name=node_data["name"],
                            method=node_data["method"],
                            source_data=task_outputs.get(source_node) if source_node else None,
                            target_data=task_outputs.get(target_node) if target_node else None,
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

                    elif task_type == "compute_statistics":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = prefect_compute_statistics.submit(
                            name=node_data["name"],
                            metrics=node_data["metrics"],
                            input_data=task_outputs.get(input_node) if input_node else None,
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

                    elif task_type == "generate_plot":
                        # Only one predecessor (input)
                        input_node = predecessors[0] if predecessors else None
                        future = prefect_generate_plot.submit(
                            name=node_data["name"],
                            plot_type=node_data["plot_type"],
                            input_data=task_outputs.get(input_node) if input_node else None,
                            kwargs=node_data["kwargs"],
                        )
                        task_outputs[node_id] = future

            logger.info("All tasks submitted to Prefect.")
            return task_outputs

        # Setup Dask clusters and configure Prefect to use the central scheduler
        cluster = self._setup_dask_clusters()

        # Execute the Prefect flow, explicitly overriding the task_runner with our central cluster
        results = mdt_flow.with_options(task_runner=DaskTaskRunner(address=cluster.scheduler_address))()

        return results
