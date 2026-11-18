"""SLURM allocation-script generation (feature 001-prep-fixed-allocation, US3).

Renders a deterministic, ready-to-submit batch script from the workflow's
``execution.allocation`` section (contracts/cli.md `mdt script`). The script
requests the fixed allocation and bootstraps the whole Dask workload inside
it via ``srun python -m mdt.allocation`` — no per-worker job submissions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdt.config import ConfigParser


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