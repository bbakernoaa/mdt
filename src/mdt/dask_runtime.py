"""Shared Dask cluster topology management for MDT execution engines.

Both the Prefect and plain-Dask engines start the same topology: a central
scheduler on the launching node plus local workers and/or batch-scheduler
(dask-jobqueue) worker jobs tagged with per-cluster Dask resources. Keeping
this logic in one place guarantees identical routing behavior across engines.
"""

import copy
import logging
import re
import subprocess
from typing import TYPE_CHECKING, Any, Dict, List
from urllib.parse import urlparse

from mdt.hpc import HPCProfileFactory

if TYPE_CHECKING:
    from mdt.config import ConfigParser

logger = logging.getLogger(__name__)


def resolve_task_resource(node_data: Dict[str, Any], exec_cfg: Dict[str, Any]) -> str:
    """Resolve the uppercase Dask resource name a DAG node routes to.

    When ``execution.cache_cluster`` is configured (e.g. a compute cluster),
    loads that can be served from an existing local cache route there instead
    of the service cluster, keeping cached data on the workers that compute
    with it. Retrieval loads (remote downloads / GRIB scans) stay on the
    service cluster (e.g. HPC data transfer nodes). Explicit per-task
    ``cluster`` assignments always win.
    """
    default_cluster = exec_cfg.get("default_cluster", "compute")
    # DAGBuilder stamps cluster_explicit on load nodes (it fills 'cluster'
    # with the service default even when the user did not set one).
    explicit = node_data.get("cluster_explicit", False)
    if node_data.get("task_type") == "load_data" and not explicit:
        cache_cluster = exec_cfg.get("cache_cluster")
        if cache_cluster and not node_data.get("retrieval", True):
            return str(cache_cluster).upper()
        return str(node_data.get("cluster") or "service").upper()
    return str(node_data.get("cluster") or default_cluster).upper()


class DaskClusterManager:
    """Owns the Dask scheduler/clusters and submitted batch worker jobs."""

    def __init__(self, config: "ConfigParser"):
        self.config = config
        self.clusters: Dict[str, Any] = {}
        self._primary_cluster: Any = None
        self._submitted_jobs: List[tuple[str, str]] = []

    def close(self) -> None:
        """Close all Dask clusters owned by this manager and cancel worker jobs."""
        resources = [self._primary_cluster]
        for resource in self.clusters.values():
            resources.extend(resource if isinstance(resource, (list, tuple)) else [resource])
        seen: set[int] = set()
        for resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.warning("Failed to close Dask resource %s: %s", type(resource).__name__, exc)
        for cancel_cmd, job_id in reversed(self._submitted_jobs):
            try:
                subprocess.run([cancel_cmd, job_id], check=True, capture_output=True, text=True)  # noqa: S603
            except Exception as exc:
                logger.warning("Failed to cancel submitted worker job %s: %s", job_id, exc)
        self._submitted_jobs.clear()
        self._primary_cluster = None
        self.clusters.clear()

    def setup_dask_clusters(self) -> Any:
        """
        Initializes Dask clusters based on the execution configuration.

        It starts a primary LocalCluster (scheduler) and then scales HPC clusters
        which connect their workers back to this primary scheduler.
        Each HPC cluster's workers are tagged with the cluster name using `resources`.
        """
        exec_cfg = self.config.execution
        clusters_cfg = exec_cfg.get("clusters", {})

        # Fixed-allocation mode (feature 001-prep-fixed-allocation): the whole
        # workload (scheduler + client + workers) is bootstrapped by SLURMRunner
        # inside an existing allocation via `srun python -m mdt.allocation`.
        # This manager must never submit batch jobs for it (FR-007, SC-003).
        if any(str(cfg.get("mode", "local")).lower() == "allocation" for cfg in clusters_cfg.values() if isinstance(cfg, dict)):
            raise RuntimeError(
                "Cluster mode 'allocation' runs inside an existing SLURM allocation, not via job submission. "
                "Launch with: srun -n <ntasks> python -m mdt.allocation <config.yaml> "
                "(see `mdt script` to generate the allocation script)."
            )

        # If there's only one cluster and it's local, keep it simple
        import dask.distributed

        if len(clusters_cfg) == 1 and next(iter(clusters_cfg.values())).get("mode", "local") == "local":
            logger.info("Setting up a single local Dask cluster.")
            cluster_name = next(iter(clusters_cfg.keys()))
            cluster_cfg = next(iter(clusters_cfg.values()))
            workers = cluster_cfg.get("workers", 1)
            memory_limit = cluster_cfg.get("memory_limit", "auto")
            # Assign the requested numeric resource to the local workers to prevent hanging
            cluster = dask.distributed.LocalCluster(
                n_workers=workers,
                resources={cluster_name.upper(): 1},
                memory_limit=memory_limit,
            )
            self._primary_cluster = cluster
            return cluster

        # Local service/compute clusters share one workstation. Expose each logical
        # resource on its workers so annotated tasks cannot wait for a nonexistent tag.
        if clusters_cfg and all(cfg.get("mode", "local") == "local" for cfg in clusters_cfg.values()):
            cluster = dask.distributed.LocalCluster(n_workers=0, host="127.0.0.1")
            self._primary_cluster = cluster
            # Synchronous distributed.Worker construction never starts in
            # current distributed releases, and spawned nanny processes
            # re-import __main__ (breaking scripts without a main guard).
            # Use managed in-process Worker specs instead: workers run as
            # thread pools inside this process, tagged per cluster.
            specs: Dict[str, Any] = {}
            self.clusters.update({name: [] for name in clusters_cfg})
            for cluster_name, cfg in clusters_cfg.items():
                workers = int(cfg.get("workers", 1))
                for index in range(workers):
                    specs[f"{cluster_name}-{index}"] = {
                        "cls": dask.distributed.Worker,
                        "options": {
                            "nthreads": int(cfg.get("threads_per_worker", 1)),
                            "resources": {cluster_name.upper(): 1},
                            "memory_limit": cfg.get("memory_limit", "auto"),
                        },
                    }
            cluster.worker_spec.update(specs)
            cluster.scale(len(specs))
            return cluster

        # Multi-cluster or HPC mode:
        # Create a central scheduler on the node where MDT is executed.
        # `execution.dask_interface` selects the network the scheduler
        # advertises and remote workers use (e.g. 'hsn0' on Gaea: the default
        # front-end 'fen' address is not routable from DTNs/compute nodes).
        interface = exec_cfg.get("dask_interface")
        logger.info("Setting up central Dask Scheduler.")
        scheduler_kwargs: Dict[str, Any] = {"n_workers": 0}  # The local cluster is just the scheduler
        if interface:
            scheduler_kwargs["interface"] = interface
        else:
            scheduler_kwargs["host"] = "0.0.0.0"  # noqa: S104 — Allow external connections
        primary_cluster = dask.distributed.LocalCluster(**scheduler_kwargs)
        self._primary_cluster = primary_cluster
        scheduler_address = primary_cluster.scheduler_address

        for cluster_name, cfg in clusters_cfg.items():
            mode = cfg.get("mode", "local")
            workers = cfg.get("workers", 1)
            kwargs = copy.deepcopy(cfg.get("cluster_kwargs", {}))

            # Let the HPCProfileFactory know the logical name of this cluster (e.g. 'service')
            # so it can choose appropriate partitions.
            kwargs["cluster_name"] = cluster_name

            # Assign resource annotations so Dask knows which workers belong to which cluster.
            res_name = cluster_name.upper()

            if mode == "local":
                logger.info("Starting %s tagged local Dask worker(s) for '%s'", workers, cluster_name)
                # Managed in-process Worker specs (see all-local branch above).
                specs = {}
                for index in range(int(workers)):
                    specs[f"{cluster_name}-{index}"] = {
                        "cls": dask.distributed.Worker,
                        "options": {
                            "nthreads": int(cfg.get("threads_per_worker", 1)),
                            "resources": {res_name: 1},
                            "memory_limit": cfg.get("memory_limit", "auto"),
                        },
                    }
                primary_cluster.worker_spec.update(specs)
                primary_cluster.scale(len(primary_cluster.worker_spec))
                self.clusters[cluster_name] = list(specs)
            else:
                logger.info(f"Setting up HPC cluster '{cluster_name}' (mode: {mode}) connecting to {scheduler_address}")

                # Remote workers must talk on the same network as the scheduler;
                # inherit execution.dask_interface unless the cluster overrides it.
                if interface:
                    kwargs.setdefault("interface", interface)

                # To support multiple disparate HPC clusters sharing a single DAG and scheduler,
                # we bypass `dask-jobqueue`'s automatic localized scheduler creation to avoid
                # port conflicts ("Address already in use"). We construct an isolated jobqueue
                # object on a random port, extract its generated batch script template, inject
                # our central scheduler address, and manually submit the job script.

                # Tag the workers with this cluster's name as a numeric resource.
                # Use the worker CLI's --resources flag: DASK_DISTRIBUTED__WORKER__
                # RESOURCES__<NAME> env vars are lowercased by dask.config, which
                # silently breaks the uppercase resource annotations below and
                # deadlocks every task. --resources preserves the key's case.
                worker_extra = list(kwargs.pop("worker_extra_args", []) or [])
                worker_extra.extend(["--resources", f"{res_name}=1"])
                kwargs["worker_extra_args"] = worker_extra

                # dask-jobqueue >= 0.8 renamed env_extra to job_script_prologue and
                # silently ignores env_extra whenever job_script_prologue is set.
                # Merge any legacy env_extra lines into job_script_prologue so user
                # setup commands are not dropped.
                prologue = list(kwargs.pop("job_script_prologue", []) or [])
                prologue.extend(kwargs.pop("env_extra", []) or [])
                if prologue:
                    kwargs["job_script_prologue"] = prologue

                # Force the dummy scheduler to a random port to avoid conflicts
                kwargs["scheduler_options"] = {"port": 0}

                # Create the JobQueue object representing this specific profile/queue
                hpc_cluster = HPCProfileFactory.create_cluster(mode, **kwargs)
                self.clusters[cluster_name] = hpc_cluster

                # Retrieve the underlying job script formatted for the specific batch system
                hpc_job_script = hpc_cluster.job_script()

                # Dask job scripts contain a variable representing the scheduler it connects to.
                # E.g., `dask-worker tcp://127.0.0.1:8786 ...`
                # We replace this dummy scheduler address with our central primary scheduler address.
                dummy_address = hpc_cluster.scheduler_address
                hpc_job_script = hpc_job_script.replace(dummy_address, scheduler_address)

                # Determine the submission command based on the mode (slurm -> sbatch, pbs -> qsub)
                submit_cmd = "sbatch" if "slurm" in mode or mode in ["hera", "jet", "orion", "hercules", "gaea", "ursa"] else "qsub"
                cancel_cmd = "scancel" if submit_cmd == "sbatch" else "qdel"
                if mode == "lsf":
                    submit_cmd = "bsub"
                    cancel_cmd = "bkill"
                submit_args = [submit_cmd, "--parsable"] if submit_cmd == "sbatch" else [submit_cmd]

                # Manually submit the modified script for the requested number of workers
                logger.info(f"Submitting {workers} {submit_cmd} jobs to '{cluster_name}' queue.")
                for job_index in range(workers):
                    # Every submitted job carries the same generated worker name
                    # (e.g. 'dummy-name'); duplicates are rejected by the scheduler
                    # ("name taken"). Give each job a unique name suffix.
                    job_script = hpc_job_script.replace("--name dummy-name", f"--name {cluster_name}-{job_index}")
                    try:
                        completed = subprocess.run(  # noqa: S603
                            submit_args, input=job_script, check=True, capture_output=True, text=True
                        )
                        self._submitted_jobs.append((cancel_cmd, self._parse_job_id(submit_cmd, completed.stdout or "")))
                    except Exception as exc:
                        logger.error("Failed to submit HPC worker job for cluster %s: %s", cluster_name, exc)
                        self.close()
                        raise RuntimeError(f"Failed to submit HPC worker job for cluster '{cluster_name}': {exc}") from exc

        return primary_cluster

    @staticmethod
    def _parse_job_id(submit_cmd: str, output: str) -> str:
        """Extract a scheduler job identifier from the submission response."""
        if submit_cmd == "bsub":
            match = re.search(r"Job\s+<(\d+)>", output)
            job_id = match.group(1) if match else ""
        else:
            job_id = output.strip().split()[0].split(";")[0] if output.strip() else ""
        if not job_id:
            raise RuntimeError(f"Could not parse {submit_cmd} job identifier from response: {output!r}")
        return job_id