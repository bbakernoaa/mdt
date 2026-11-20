"""SLURM batch-script generation (feature 001-prep-fixed-allocation, US3).

Renders deterministic, ready-to-submit batch scripts from the workflow
configuration (contracts/cli.md `mdt script`):

- ``--stage compute`` (default): requests the fixed allocation from
  ``execution.allocation`` and bootstraps the whole Dask workload inside it
  via ``srun python -m mdt.allocation`` — no per-worker job submissions.
- ``--stage prep``: a single-task retrieval job for a node with data access
  (head/service node or DTN), sized from ``execution.prep_allocation`` or
  conservative defaults. Gaea DTN defaults respect the dtn QoS 4-CPU/job cap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdt.config import ConfigParser


def render_prep_script(config: "ConfigParser", config_path: str) -> str:
    """Render the prep-stage batch script (pure, deterministic, no side effects).

    Resources come from ``execution.prep_allocation`` when present, otherwise
    from conservative single-task defaults appropriate for a DTN/service node
    (4 CPUs honors the Gaea dtn QoS per-job cap). Site directives fall back to
    ``execution.allocation`` values (account, python, prologue) when set.
    """
    exec_cfg = config.execution
    prep_cfg = exec_cfg.get("prep_allocation") or {}
    allocation = exec_cfg.get("allocation") or {}

    cpus = int(prep_cfg.get("cpus", 4))
    walltime = prep_cfg.get("walltime", "04:00:00")
    account = prep_cfg.get("account") or allocation.get("account")
    if not account:
        raise ValueError(
            "Prep script generation requires an account: set execution.prep_allocation.account "
            "or execution.allocation.account."
        )
    python = prep_cfg.get("python") or allocation.get("python") or sys.executable
    prologue = list(prep_cfg.get("prologue") or allocation.get("prologue") or [f"cd {Path.cwd()}"])
    force = bool(prep_cfg.get("force", False))
    abs_config = Path(config_path).resolve()

    lines = [
        "#!/bin/bash",
        "#SBATCH -J mdt-prep",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH -t {walltime}",
        f"#SBATCH -A {account}",
    ]
    memory = prep_cfg.get("memory")
    if memory:
        lines.append(f"#SBATCH --mem={memory}")
    if prep_cfg.get("partition"):
        lines.append(f"#SBATCH -p {prep_cfg['partition']}")
    if prep_cfg.get("qos"):
        lines.append(f"#SBATCH --qos={prep_cfg['qos']}")
    if prep_cfg.get("slurm_cluster"):
        lines.append(f"#SBATCH -M {prep_cfg['slurm_cluster']}")
    for directive in prep_cfg.get("extra_directives") or []:
        lines.append(f"#SBATCH {directive}")
    lines.append("")
    lines.extend(prologue)
    lines.append("")
    force_flag = " --force" if force else ""
    lines.append(f"{python} -m mdt prep {abs_config}{force_flag}")
    lines.append("")
    return "\n".join(lines)


def render_allocation_script(config: "ConfigParser", config_path: str) -> str:
    """Render the allocation batch script text (pure, deterministic, no side effects).

    Raises ValueError when execution.allocation is absent or lacks the fields
    required for generation (nodes, account).
    """
    allocation = config.execution.get("allocation")
    if not allocation:
        raise ValueError(
            "Script generation requires an 'execution.allocation' section "
            "(nodes, ntasks_per_node, cpus_per_task, memory, walltime, account). "
            "See docs/hpc_usage.md and specs/001-prep-fixed-allocation/contracts/cli.md."
        )
    missing = [key for key in ("nodes", "account") if key not in allocation]
    if missing:
        raise ValueError(f"Script generation requires execution.allocation fields: {missing}.")

    nodes = int(allocation["nodes"])
    ntasks_per_node = int(allocation.get("ntasks_per_node", 4))
    cpus_per_task = int(allocation.get("cpus_per_task", 1))
    memory = allocation.get("memory")  # optional: some sites (e.g. Gaea c6) don't configure consumable memory
    walltime = allocation.get("walltime", "02:00:00")
    account = allocation["account"]
    python = allocation.get("python") or sys.executable
    prologue = list(allocation.get("prologue") or [f"cd {Path.cwd()}"])
    abs_config = Path(config_path).resolve()

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J mdt-analysis",
        f"#SBATCH -N {nodes}",
        f"#SBATCH --ntasks-per-node={ntasks_per_node}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH -t {walltime}",
        f"#SBATCH -A {account}",
    ]
    if memory:
        lines.append(f"#SBATCH --mem={memory}")
    if allocation.get("partition"):
        lines.append(f"#SBATCH -p {allocation['partition']}")
    if allocation.get("qos"):
        lines.append(f"#SBATCH --qos={allocation['qos']}")
    if allocation.get("slurm_cluster"):
        lines.append(f"#SBATCH -M {allocation['slurm_cluster']}")
    lines.append("")
    lines.extend(prologue)
    lines.append("")
    lines.append(f"srun -n {nodes * ntasks_per_node} {python} -m mdt.allocation {abs_config}")
    lines.append("")
    return "\n".join(lines)