"""Fixed-allocation execution entry point (feature 001-prep-fixed-allocation, US2).

Run inside an already-granted SLURM allocation:

    srun -n <ntasks> python -m mdt.allocation workflow.yaml

Every rank constructs dask-jobqueue's ``SLURMRunner`` (research.md R1):
rank 0 becomes the Dask scheduler, rank 1 the client (which executes the
analysis DAG), and the remaining ranks become workers. Coordination happens
through a ``scheduler-<job_id>.json`` file on shared storage — no additional
batch jobs are submitted and no external network access is needed: all data
must already be prepared via ``mdt prep`` (fail-fast checked before any task
is submitted).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from dask.distributed import Client
from dask_jobqueue.slurm import SLURMRunner

from mdt.config import load_config
from mdt.dag import DAGBuilder
from mdt.dask_engine import DaskEngine
from mdt.prep import check_caches

logger = logging.getLogger(__name__)

_REQUIRED_ENV = ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_JOB_ID")


@dataclass
class AllocationInfo:
    """Runtime allocation facts from the SLURM environment (data-model.md AllocationSpec)."""

    job_id: int
    ntasks: int
    proc_id: int
    num_nodes: int
    cpus_per_task: int

    @property
    def n_workers(self) -> int:
        """Worker tasks available after scheduler (rank 0) and client (rank 1)."""
        return self.ntasks - 2


def detect_allocation(env: Optional[Mapping[str, str]] = None) -> AllocationInfo:
    """Read and validate the SLURM allocation environment (FR-010, SC-007).

    Raises RuntimeError with actionable guidance when required variables are
    missing or the allocation is too small for the runner (needs >= 3 tasks).
    """
    env = dict(os.environ if env is None else env)
    missing = [key for key in _REQUIRED_ENV if key not in env]
    if missing:
        raise RuntimeError(
            f"Allocation mode requires the SLURM environment variables {missing}. "
            "Launch inside an allocation, e.g.: salloc -N 1 --ntasks=4 --cpus-per-task=8, "
            "then: srun -n 4 python -m mdt.allocation <config.yaml>"
        )
    ntasks = int(env["SLURM_NTASKS"])
    if ntasks < 3:
        raise RuntimeError(
            f"Allocation has SLURM_NTASKS={ntasks}; SLURMRunner needs at least 3 tasks "
            "(one each for scheduler, client, and a worker). Request more tasks, e.g. --ntasks=4."
        )
    return AllocationInfo(
        job_id=int(env["SLURM_JOB_ID"]),
        ntasks=ntasks,
        proc_id=int(env["SLURM_PROCID"]),
        num_nodes=int(env.get("SLURM_JOB_NUM_NODES", "1")),
        cpus_per_task=int(env.get("SLURM_CPUS_PER_TASK", "1")),
    )


def adapt_layout(info: AllocationInfo, requested_workers: int, cpus_per_task: int) -> Dict[str, int]:
    """Clamp the requested worker layout to the actual allocation (FR-010).

    Warns instead of waiting for resources that will never arrive.
    """
    n_workers = min(requested_workers, info.n_workers)
    if n_workers < requested_workers:
        logger.warning(
            "Requested %d workers but the allocation provides only %d worker tasks "
            "(SLURM_NTASKS=%d minus scheduler and client); clamping to the allocation.",
            requested_workers,
            info.n_workers,
            info.ntasks,
        )
    return {"n_workers": n_workers, "nthreads": cpus_per_task}


def main(argv: Optional[List[str]] = None) -> None:
    """SLURMRunner entry: all ranks bootstrap; the client rank runs the DAG."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: python -m mdt.allocation CONFIG_PATH")
    config_path = argv[0]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    info = detect_allocation()
    config = load_config(config_path)
    exec_cfg = config.execution
    allocation_cfg = exec_cfg.get("allocation") or {}
    interface = allocation_cfg.get("interface") or exec_cfg.get("dask_interface")
    cpus_per_task = int(allocation_cfg.get("cpus_per_task", info.cpus_per_task))

    # Worker layout: SLURMRunner gives one worker per remaining rank; nthreads
    # comes from the allocation's cpus-per-task (adapt_layout logs any clamp).
    layout = adapt_layout(info, info.n_workers, cpus_per_task)

    worker_options: Dict[str, Any] = {"nthreads": layout["nthreads"]}
    if interface:
        worker_options["interface"] = interface
    scheduler_options: Dict[str, Any] = {}
    if interface:
        scheduler_options["interface"] = interface

    # All ranks construct the runner; only the client rank returns past this
    # point (scheduler/worker ranks serve until the client finishes).
    runner = SLURMRunner(
        scheduler_file=f"scheduler-{info.job_id}.json",
        worker_options=worker_options,
        scheduler_options=scheduler_options,
    )

    # Client rank: execute the analysis DAG on the runner's cluster.
    client = Client(runner.scheduler_address)
    exit_code = 0
    try:
        dag = DAGBuilder(config).build()
        check_caches(config, dag)
        engine = DaskEngine(dag, config, client=client)
        engine.execute()
        logger.info("Allocation workflow completed successfully.")
    except Exception as exc:
        logger.error("Allocation workflow failed: %s", exc)
        exit_code = 1
    finally:
        client.close()
        try:
            runner.close()
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            logger.warning("Runner shutdown issue: %s", exc)
        scheduler_file = getattr(runner, "scheduler_file", None)
        if isinstance(scheduler_file, (str, os.PathLike)):
            try:
                os.remove(scheduler_file)
            except OSError:
                pass
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()