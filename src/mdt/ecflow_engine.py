"""ecFlow orchestrator engine for MDT.

Translates a NetworkX DAG into an ecFlow suite definition, generates
per-task wrapper scripts, and submits the suite to an ecFlow server.

The ``ecflow`` package is imported lazily so that this module can be
loaded without ecFlow installed.
"""

import json
import os

import networkx as nx

from mdt.engine_registry import Engine

#: Maps DAG ``task_type`` values to ecFlow family names.
_FAMILY_MAP: dict[str, str] = {
    "load_data": "load",
    "pair_data": "pair",
    "combine_paired_data": "combine",
    "compute_statistics": "statistics",
    "generate_plot": "plot",
}


#: Template for the dispatch block inside each wrapper script.
#: Maps ``TASK_TYPE`` values to the import + call statements.
_DISPATCH_BLOCKS: dict[str, str] = {
    "load_data": (
        "    from mdt.tasks.data import load_data\n"
        "    load_data(name=task_name, dataset_type=dataset_type, kwargs=kwargs)"
    ),
    "pair_data": (
        "    from mdt.tasks.pairing import pair_data\n"
        "    pair_data(name=task_name, method=kwargs.pop('method', ''),\n"
        "             source_data=None, target_data=None, kwargs=kwargs)"
    ),
    "combine_paired_data": (
        "    from mdt.tasks.pairing import combine_paired_data\n"
        "    combine_paired_data(paired_data=kwargs.pop('paired_data', {}),\n"
        "                        dim=kwargs.pop('dim', 'model'))"
    ),
    "compute_statistics": (
        "    from mdt.tasks.statistics import compute_statistics\n"
        "    compute_statistics(name=task_name, metrics=metrics,\n"
        "                       input_data=None, kwargs=kwargs)"
    ),
    "generate_plot": (
        "    from mdt.tasks.plotting import generate_plot\n"
        "    generate_plot(name=task_name, plot_type=plot_type,\n"
        "                  input_data=None, kwargs=kwargs)"
    ),
}


def _build_wrapper_script(node_id: str) -> str:
    """Return the full text of an ``.ecf`` wrapper script for *node_id*.

    The script uses ecFlow ``%VAR%`` substitution tokens that the ecFlow
    server replaces at runtime with the variable values set on the task
    node.
    """
    lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated ecFlow wrapper for task %TASK_NAME%."""',
        "import json, os, sys",
        "import ecflow",
        "",
        'client = ecflow.Client(os.environ.get("ECF_HOST", "localhost"),',
        '                       int(os.environ.get("ECF_PORT", "3141")))',
        "try:",
        '    client.init(os.environ["ECF_NAME"], os.environ["ECF_PASS"])',
        "",
        "    # --- task parameters (substituted by ecFlow) ---",
        '    task_type = "%TASK_TYPE%"',
        '    task_name = "%TASK_NAME%"',
        '    dataset_type = "%DATASET_TYPE%"',
        "    kwargs = json.loads('%TASK_KWARGS%')",
        "    metrics = json.loads('%METRICS%')",
        '    plot_type = "%PLOT_TYPE%"',
        "",
        "    # --- dispatch to the correct MDT task function ---",
    ]

    # Build an if/elif chain for all known task types.
    first = True
    for task_type, block in _DISPATCH_BLOCKS.items():
        keyword = "if" if first else "elif"
        lines.append(f'    {keyword} task_type == "{task_type}":')
        lines.append(block)
        first = False
    lines.append("    else:")
    lines.append('        raise ValueError(f"Unknown task_type: {task_type}")')

    lines += [
        "",
        "    client.complete()",
        "except Exception as e:",
        "    client.abort(str(e))",
        "    sys.exit(1)",
        "",
    ]
    return "\n".join(lines) + "\n"


class EcFlowEngine(Engine):
    """Engine implementation that executes DAGs via ECMWF ecFlow.

    Parameters
    ----------
    dag : nx.DiGraph
        The task dependency graph built by :class:`~mdt.dag.DAGBuilder`.
    config : object
        The parsed MDT configuration object (typically
        :class:`~mdt.config.ConfigParser`).

    Raises
    ------
    ImportError
        If the ``ecflow`` package is not installed.
    """

    def __init__(self, dag: nx.DiGraph, config):
        try:
            import ecflow  # lazy import — keep ecflow optional
        except ImportError:
            raise ImportError(
                "ecFlow is not installed. Install with: pip install mdt[ecflow]"
            )

        self.dag = dag
        self.config = config
        self.ecflow = ecflow

        exec_cfg = config.execution
        self.host: str = exec_cfg.get("ecflow_host", "localhost")
        self.port: int = int(exec_cfg.get("ecflow_port", 3141))
        self.suite_name: str = exec_cfg.get("suite_name", "mdt")
        self.task_script_dir: str = exec_cfg.get("task_script_dir", "./ecflow_tasks/")

    # ------------------------------------------------------------------
    # Engine ABC
    # ------------------------------------------------------------------

    def execute(self) -> dict:
        """Build the suite, generate wrappers, and start the suite.

        Returns
        -------
        dict
            ``{"suite": <suite_name>, "status": "started"}``
        """
        defs = self.build_suite()
        self.generate_task_wrappers()
        self._load_and_start(defs)
        return {"suite": self.suite_name, "status": "started"}

    # ------------------------------------------------------------------
    # Stubs — implemented in later tasks
    # ------------------------------------------------------------------

    def build_suite(self):
        """Translate the DAG into an ``ecflow.Defs`` suite definition.

        Creates one ecFlow family per task type and one ecFlow task node
        per DAG node.  Trigger expressions are derived from DAG edges and
        ecFlow variables carry the parameters each wrapper script needs.

        Returns
        -------
        ecflow.Defs
            The fully constructed suite definition ready to be loaded into
            an ecFlow server.
        """
        ecflow = self.ecflow

        defs = ecflow.Defs()
        suite = ecflow.Suite(self.suite_name)

        # Create one family per task type.
        families: dict[str, object] = {}
        for family_name in _FAMILY_MAP.values():
            family = ecflow.Family(family_name)
            families[family_name] = family
            suite.add_family(family)

        # Build lookups for trigger generation.
        node_family: dict[str, str] = {}
        node_tasks: dict[str, object] = {}

        for node_id, data in self.dag.nodes(data=True):
            task_type = data["task_type"]
            family_name = _FAMILY_MAP[task_type]
            node_family[node_id] = family_name

            task = ecflow.Task(node_id)

            # --- ecFlow variables ---
            task.add_variable("TASK_TYPE", data["task_type"])
            task.add_variable("TASK_NAME", data["name"])
            task.add_variable("DATASET_TYPE", data.get("dataset_type") or "")
            task.add_variable("TASK_KWARGS", json.dumps(data.get("kwargs") or {}))
            task.add_variable("METRICS", json.dumps(data.get("metrics") or []))
            task.add_variable("PLOT_TYPE", data.get("plot_type") or "")
            task.add_variable("CLUSTER", data.get("cluster") or "")

            families[family_name].add_task(task)
            node_tasks[node_id] = task

        # --- trigger expressions from DAG edges ---
        for node_id in self.dag.nodes:
            predecessors = list(self.dag.predecessors(node_id))
            if not predecessors:
                continue

            parts = [
                f"{node_family[pred]}/{pred} == complete"
                for pred in sorted(predecessors)
            ]
            trigger_expr = " and ".join(parts)
            node_tasks[node_id].add_trigger(trigger_expr)

        defs.add_suite(suite)
        return defs

    def generate_task_wrappers(self) -> list[str]:
        """Generate one ``.ecf`` wrapper script per DAG node.

        Each script is a self-contained Python program that:

        1. Connects to the ecFlow server and calls ``client.init()``.
        2. Dispatches to the correct ``mdt.tasks.*`` function based on
           the ``TASK_TYPE`` ecFlow variable.
        3. Calls ``client.complete()`` on success.
        4. Calls ``client.abort(error_message)`` and exits with code 1
           on any exception.

        Returns
        -------
        list[str]
            Paths of the generated ``.ecf`` files.
        """
        os.makedirs(self.task_script_dir, exist_ok=True)

        generated: list[str] = []
        for node_id in self.dag.nodes:
            script_path = os.path.join(self.task_script_dir, f"{node_id}.ecf")
            with open(script_path, "w") as fh:
                fh.write(_build_wrapper_script(node_id))
            generated.append(script_path)

        return generated

    def _load_and_start(self, defs) -> None:
        """Load the suite definition into the ecFlow server and begin it.

        Parameters
        ----------
        defs : ecflow.Defs
            The fully constructed suite definition.

        Raises
        ------
        RuntimeError
            If the connection to the ecFlow server fails.
        """
        try:
            client = self.ecflow.Client(self.host, self.port)
            client.load(defs)
            client.begin_suite(self.suite_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to ecFlow server at "
                f"{self.host}:{self.port} — {exc}"
            ) from exc
