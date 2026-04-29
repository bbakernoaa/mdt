"""Prefect execution engine and task wrappers for MDT."""

import logging

import networkx as nx

from mdt.engine_registry import Engine
from mdt.hpc import HPCProfileFactory
from mdt.tasks.data import load_data
from mdt.tasks.pairing import combine_paired_data, pair_data
from mdt.tasks.plotting import generate_plot
from mdt.tasks.statistics import compute_statistics

logger = logging.getLogger(__name__)


class PrefectEngine(Engine):
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
        import dask.distributed

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

            # Let the HPCProfileFactory know the logical name of this cluster (e.g. 'service')
            # so it can choose appropriate partitions.
            kwargs["cluster_name"] = cluster_name

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
        from prefect import flow, get_run_logger, task
        from prefect_dask.task_runners import DaskTaskRunner

        # Prefect Task Wrappers — defined here so the @task decorator is only
        # evaluated when Prefect is actually installed and execute() is called.

        @task(name="Load Data")
        def prefect_load_data(name, dataset_type, kwargs):
            logger = get_run_logger()
            logger.info(f"Loading data: {name} of type {dataset_type}")
            return load_data(name, dataset_type, kwargs)

        @task(name="Pair Data")
        def prefect_pair_data(name, method, source_data, target_data, kwargs):
            logger = get_run_logger()
            logger.info(f"Pairing data: {name} using {method}")
            return pair_data(name, method, source_data, target_data, kwargs)

        @task(name="Combine Paired Data")
        def prefect_combine_paired_data(paired_data, dim="model"):
            logger = get_run_logger()
            logger.info(f"Combining {len(paired_data)} paired datasets along '{dim}'")
            return combine_paired_data(paired_data, dim=dim)

        @task(name="Compute Statistics")
        def prefect_compute_statistics(name, metrics, input_data, kwargs):
            logger = get_run_logger()
            logger.info(f"Computing statistics: {name} for metrics {metrics}")
            return compute_statistics(name, metrics, input_data, kwargs)

        @task(name="Generate Plot")
        def prefect_generate_plot(name, plot_type, input_data, kwargs):
            logger = get_run_logger()
            logger.info(f"Generating plot: {name} of type {plot_type}")
            return generate_plot(name, plot_type, input_data, kwargs)

        import dask

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

                        future = prefect_combine_paired_data.submit(
                            paired_data=paired_data_inputs,
                            dim=node_data.get("dim", "model"),
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
