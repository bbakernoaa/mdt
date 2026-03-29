import logging

import networkx as nx

logger = logging.getLogger(__name__)


class DAGBuilder:
    """Builds a NetworkX Directed Acyclic Graph based on MDT configuration."""

    def __init__(self, config):
        self.config = config
        self.graph = nx.DiGraph()

    def build(self):
        """
        Constructs and validates the dependency graph.

        Returns
        -------
        networkx.DiGraph
            The fully constructed Directed Acyclic Graph representing the
            execution workflow.

        Raises
        ------
        ValueError
            If the generated graph contains cycles and is not a valid DAG.
        """
        logger.info("Building task graph...")
        self._add_data_nodes()
        self._add_pairing_nodes()
        self._add_combine_nodes()
        self._add_statistics_nodes()
        self._add_plotting_nodes()

        # Validate that the graph is indeed a DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The generated task graph contains cycles and is not a valid DAG.")

        logger.info(f"Built task graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph

    def _add_data_nodes(self):
        """Adds nodes for loading data (from monetio)."""
        data_cfg = self.config.data
        if not data_cfg:
            logger.warning("No 'data' section found in configuration.")
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in data_cfg.items():
            node_id = f"load_{name}"
            # Type specifies the reader to use from monetio.readers
            dataset_type = details.get("type")
            if not dataset_type:
                logger.warning(f"Data source '{name}' is missing 'type'. Skipping.")
                continue

            kwargs = details.get("kwargs") or {}
            if "use_kerchunk" in details:
                kwargs["use_kerchunk"] = details.get("use_kerchunk")
            if "kerchunk_file" in details:
                kwargs["kerchunk_file"] = details.get("kerchunk_file")

            self.graph.add_node(
                node_id,
                task_type="load_data",
                name=name,
                dataset_type=dataset_type,
                cluster=details.get("cluster", default_cluster),
                kwargs=kwargs,
            )

    def _add_pairing_nodes(self):
        """Adds nodes for pairing datasets together (from monet)."""
        pairing_cfg = self.config.pairing
        if not pairing_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in pairing_cfg.items():
            node_id = f"pair_{name}"
            # E.g. source: load_gcafs, target: load_aeronet
            source = details.get("source")
            target = details.get("target")

            if not source or not target:
                logger.warning(f"Pairing '{name}' is missing 'source' or 'target'. Skipping.")
                continue

            # Dependencies: both datasets must be loaded before they can be paired
            source_node = f"load_{source}"
            target_node = f"load_{target}"

            if source_node not in self.graph:
                logger.error(f"Source data '{source}' for pairing '{name}' not found. Skipping pairing.")
                continue
            if target_node not in self.graph:
                logger.error(f"Target data '{target}' for pairing '{name}' not found. Skipping pairing.")
                continue

            self.graph.add_node(
                node_id,
                task_type="pair_data",
                name=name,
                method=details.get("method", "interpolate"),  # Default to interpolate or similar
                cluster=details.get("cluster", default_cluster),
                kwargs=details.get("kwargs", {}),
            )

            self.graph.add_edge(source_node, node_id)
            self.graph.add_edge(target_node, node_id)

    def _add_combine_nodes(self):
        """Adds nodes for combining multiple paired datasets."""
        combine_cfg = self.config.combine
        if not combine_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in combine_cfg.items():
            node_id = f"combine_{name}"
            sources = details.get("sources", [])
            dim = details.get("dim", "model")

            if not sources:
                logger.warning(f"Combine task '{name}' has no sources. Skipping.")
                continue

            # Check if all sources exist (as paired nodes)
            valid_sources = []
            for source in sources:
                pair_node_id = f"pair_{source}"
                if pair_node_id in self.graph:
                    valid_sources.append(pair_node_id)
                else:
                    logger.warning(f"Source '{source}' for combine task '{name}' not found. Skipping source.")

            if not valid_sources:
                logger.error(f"Combine task '{name}' has no valid sources. Skipping.")
                continue

            self.graph.add_node(
                node_id,
                task_type="combine_paired_data",
                name=name,
                sources=sources,  # List of original names, Engine uses this to look them up
                dim=dim,
                cluster=details.get("cluster", default_cluster),
                kwargs=details.get("kwargs", {}),
            )

            for pair_node_id in valid_sources:
                self.graph.add_edge(pair_node_id, node_id)

    def _add_statistics_nodes(self):
        """Adds nodes for computing statistics on paired data (from monet-stats)."""
        stats_cfg = self.config.statistics
        if not stats_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in stats_cfg.items():
            node_id = f"stats_{name}"
            input_data = details.get("input")  # E.g., a pairing name or just a dataset

            if not input_data:
                logger.warning(f"Statistics task '{name}' is missing 'input'. Skipping.")
                continue

            # We need to map 'input' to a node. It could be a 'pair_' node or a 'load_' node.
            # Usually stats are done on paired data, but we allow either for flexibility.
            possible_nodes = [f"combine_{input_data}", f"pair_{input_data}", f"load_{input_data}"]
            found_edge = False
            target_parent = None
            for p in possible_nodes:
                if p in self.graph:
                    target_parent = p
                    found_edge = True
                    break

            if not found_edge:
                logger.error(f"Input '{input_data}' for statistics '{name}' not found. Skipping stats.")
                continue

            self.graph.add_node(
                node_id,
                task_type="compute_statistics",
                name=name,
                metrics=details.get("metrics", []),
                cluster=details.get("cluster", default_cluster),
                kwargs=details.get("kwargs", {}),
            )

            self.graph.add_edge(target_parent, node_id)

    def _add_plotting_nodes(self):
        """Adds nodes for plotting data or statistics (from monet-plots)."""
        plots_cfg = self.config.plots
        if not plots_cfg:
            return

        default_cluster = self.config.execution.get("default_cluster", "compute")

        for name, details in plots_cfg.items():
            node_id = f"plot_{name}"
            input_data = details.get("input")  # E.g., pair_name, stats_name, or load_name

            if not input_data:
                logger.warning(f"Plot task '{name}' is missing 'input'. Skipping.")
                continue

            # Find the dependency. Could be stats, combine, pair, or load
            possible_nodes = [f"stats_{input_data}", f"combine_{input_data}", f"pair_{input_data}", f"load_{input_data}"]
            found_edge = False
            target_parent = None
            for p in possible_nodes:
                if p in self.graph:
                    target_parent = p
                    found_edge = True
                    break

            if not found_edge:
                logger.error(f"Input '{input_data}' for plot '{name}' not found. Skipping plot.")
                continue

            self.graph.add_node(
                node_id,
                task_type="generate_plot",
                name=name,
                plot_type=details.get("type", "spatial"),  # Type of plot to generate
                cluster=details.get("cluster", default_cluster),
                kwargs=details.get("kwargs", {}),
            )

            self.graph.add_edge(target_parent, node_id)
