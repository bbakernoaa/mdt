"""Command-line interface for the MONET Verification System (MVS)."""

import argparse
import logging

from mvs.config import load_config
from mvs.dag import DAGBuilder
from mvs.engine import PrefectEngine


def setup_logging():
    """Configures the standard logging format for MVS."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    """
    Entry point for the MVS CLI application.

    Parses command-line arguments and executes the requested subcommands.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Monet Environmental Verification System (MVS)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # The 'run' command
    run_parser = subparsers.add_parser("run", help="Run a verification workflow from a YAML config.")
    run_parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")

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

            # 3. Initialize the Prefect execution engine and run the graph
            logger.info("Initializing Prefect Execution Engine")
            engine = PrefectEngine(dag, config)

            logger.info("Executing workflow")
            engine.execute()

            logger.info("Workflow execution completed successfully.")

        except Exception as e:
            logger.error(f"Error executing MVS workflow: {e}")
            raise
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
