"""Command-line interface for the Model Development Tool (MDT)."""

import argparse
import logging

from mdt.config import load_config
from mdt.dag import DAGBuilder
from mdt.engine_registry import EngineRegistry


def setup_logging():
    """Configures the standard logging format for MDT."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    """
    Entry point for the MDT CLI application.

    Parses command-line arguments and executes the requested subcommands.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Monet Environmental Verification System (MDT)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # The 'run' command
    run_parser = subparsers.add_parser("run", help="Run a verification workflow from a YAML config.")
    run_parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    run_parser.add_argument(
        "--orchestrator",
        choices=["prefect", "ecflow"],
        default=None,
        help="Override the orchestrator backend specified in the YAML configuration.",
    )

    args = parser.parse_args()

    if args.command == "run":
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
            logger.error(f"Error executing MDT workflow: {e}")
            raise
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
