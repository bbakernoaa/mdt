"""Command-line interface for the Model Development Tool (MDT)."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from mdt import __version__
from mdt.config import load_config
from mdt.dag import DAGBuilder
from mdt.engine_registry import EngineRegistry


def setup_logging(debug: bool = False) -> None:
    """Configures the standard logging format for MDT."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> None:
    """
    Entry point for the MDT CLI application.

    Parses command-line arguments and executes the requested subcommands.
    """
    parser = argparse.ArgumentParser(description="Monet Environmental Verification System (MDT)")
    parser.add_argument("--version", action="version", version=f"MDT {__version__}")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and show full stack traces on error.")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # The 'run' command
    run_parser = subparsers.add_parser("run", help="Run a verification workflow from a YAML config.")
    run_parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    run_parser.add_argument(
        "--orchestrator",
        choices=["prefect", "dask", "ecflow"],
        default=None,
        help="Override the orchestrator backend specified in the YAML configuration.",
    )

    # The 'prep' command (feature 001-prep-fixed-allocation): retrieve/prepare
    # datasets to local icechunk/zarr stores; never runs analysis stages.
    prep_parser = subparsers.add_parser("prep", help="Prepare (retrieve/cache) all datasets to local stores without running analysis.")
    prep_parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    prep_parser.add_argument(
        "--orchestrator",
        choices=["dask", "prefect"],
        default=None,
        help="Override the orchestrator backend (prep defaults to the plain-Dask engine).",
    )
    prep_parser.add_argument("--force", action="store_true", help="Ignore cached stores and re-retrieve all datasets.")

    # The 'script' command (feature 001-prep-fixed-allocation): generate a
    # SLURM allocation script for the compute stage; never submits it.
    script_parser = subparsers.add_parser("script", help="Generate a SLURM allocation script for the workflow's compute stage.")
    script_parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    script_parser.add_argument("--output", "-o", type=str, default="mdt_allocation.slurm", help="Output script path.")
    script_parser.add_argument("--stage", choices=["compute"], default="compute", help="Workflow stage to run in the allocation.")

    # The 'validate' command
    validate_parser = subparsers.add_parser("validate", help="Validate a YAML configuration file.")
    validate_parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")

    # The 'template' command
    template_parser = subparsers.add_parser("template", help="Generate a sample YAML configuration file.")
    template_parser.add_argument("--output", "-o", type=str, default="config_template.yaml", help="Output filename.")

    args = parser.parse_args()
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    if args.command == "run":
        engine = None
        try:
            # 1. Load the configuration
            logger.info(f"Loading configuration from {args.config_path}")
            config = load_config(args.config_path)

            # 2. Build the Directed Acyclic Graph (DAG)
            logger.info("Constructing Task Graph")
            builder = DAGBuilder(config)
            dag = builder.build()

            # 3. Resolve orchestrator: CLI arg → YAML config → default ("prefect")
            orchestrator_name = args.orchestrator or config.orchestrator

            # 4. Initialize the execution engine via the registry and run the graph
            logger.info(f"Initializing '{orchestrator_name}' execution engine")
            engine_cls = EngineRegistry.get_engine(orchestrator_name)
            engine = engine_cls(dag, config)

            logger.info("Executing workflow")
            engine.execute()

            logger.info("Workflow execution completed successfully.")

        except Exception as e:
            if args.debug:
                logger.exception("Error executing MDT workflow")
            else:
                logger.error(f"Error executing MDT workflow: {e}")
            sys.exit(1)
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as cleanup_error:
                    logger.warning("Failed to cleanly close execution engine: %s", cleanup_error)

    elif args.command == "prep":
        try:
            from mdt.prep import run_prep

            logger.info(f"Loading configuration from {args.config_path}")
            config = load_config(args.config_path)
            logger.info("Preparing datasets (retrieval only; analysis stages are not executed)")
            results = run_prep(config, force=args.force)

            counts = {"retrieved": 0, "cached": 0, "failed": 0}
            for result in results:
                counts[result["status"]] += 1
                line = f"{result['dataset']}: {result['status']} -> {result['store_path']}"
                if result.get("error"):
                    line += f" ({result['error']})"
                if result.get("warning"):
                    line += f" [warning: {result['warning']}]"
                print(line)
            print(f"PREP SUMMARY: {counts['retrieved']} retrieved, {counts['cached']} cached, {counts['failed']} failed")

            summary_file = (config.config.get("prep") or {}).get("summary_file")
            if summary_file:
                import json

                Path(summary_file).write_text(json.dumps(results, indent=2))
                logger.info("Wrote prep summary to %s", summary_file)

            if counts["failed"]:
                sys.exit(1)
        except SystemExit:
            raise
        except Exception as e:
            if args.debug:
                logger.exception("Error preparing datasets")
            else:
                logger.error(f"Error preparing datasets: {e}")
            sys.exit(1)

    elif args.command == "script":
        try:
            from mdt.scriptgen import render_allocation_script

            config = load_config(args.config_path)
            script = render_allocation_script(config, args.config_path)
            Path(args.output).write_text(script)
            print(f"Wrote allocation script to {args.output}")
            print(f"Submit with: sbatch {args.output}")
        except Exception as e:
            if args.debug:
                logger.exception("Error generating allocation script")
            else:
                logger.error(f"Error generating allocation script: {e}")
            sys.exit(1)

    elif args.command == "validate":
        try:
            logger.info(f"Validating configuration: {args.config_path}")
            config = load_config(args.config_path)
            # Try building the DAG as well, to catch reference errors
            builder = DAGBuilder(config)
            builder.build()
            print(f"SUCCESS: Configuration '{args.config_path}' is valid.")
        except Exception as e:
            if args.debug:
                logger.exception("Validation failed")
            else:
                logger.error(f"Validation failed: {e}")
            sys.exit(1)

    elif args.command == "template":
        template = {
            "data": {
                "my_model": {"type": "cmaq", "kwargs": {"fname": "path/to/cmaq_output.nc"}},
                "my_obs": {"type": "aeronet", "kwargs": {"fname": "path/to/aeronet_data.nc"}},
            },
            "pairing": {"eval_pair": {"source": "my_model", "target": "my_obs", "method": "interpolate"}},
            "statistics": {
                "basic_stats": {"input": "eval_pair", "metrics": ["rmse", "bias", "corr"], "kwargs": {"obs_var": "obs", "mod_var": "mod"}}
            },
            "plots": {"spatial_eval": {"input": "eval_pair", "type": "spatial", "kwargs": {"savename": "spatial_plot.png"}}},
            "execution": {
                "orchestrator": "prefect",
                "default_cluster": "local",
                "pairing_mode": "serial",
                "clusters": {"local": {"mode": "local", "workers": 1}},
            },
        }
        try:
            with Path(args.output).open("w") as f:
                yaml.safe_dump(template, f, sort_keys=False)
                f.write(
                    "\n# --- Optional: two-stage prep + fixed-allocation execution ---\n"
                    "# Stage 1 (where data access exists, e.g. a DTN):  mdt prep this_file.yaml\n"
                    "# Stage 2 (inside a fixed SLURM allocation):\n"
                    "#   mdt script this_file.yaml -o run.slurm && sbatch run.slurm\n"
                    "# prep:\n"
                    "#   store_dir: ./zarr_stores\n"
                    "#   summary_file: prep_summary.json\n"
                    "# execution:\n"
                    "#   orchestrator: dask\n"
                    "#   dask_interface: hsn0        # Gaea: Slingshot HSN\n"
                    "#   allocation:\n"
                    "#     nodes: 1\n"
                    "#     ntasks_per_node: 4        # 1 scheduler + 1 client + 2 workers\n"
                    "#     cpus_per_task: 8\n"
                    "#     memory: 200GB\n"
                    "#     walltime: '02:00:00'\n"
                    "#     account: my-project\n"
                    "#     partition: batch\n"
                    "#     slurm_cluster: c6\n"
                    "#   clusters:\n"
                    "#     compute: {mode: allocation}\n"
                )
            print(f"Generated template configuration: {args.output}")
        except Exception as e:
            logger.error(f"Failed to generate template: {e}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
