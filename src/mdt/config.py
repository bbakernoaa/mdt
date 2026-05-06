"""Module for parsing and validating MDT YAML configuration files."""

import logging
import pathlib
from typing import Any, Dict, Union, cast

import yaml

logger = logging.getLogger(__name__)

VALID_ZARR_BACKENDS = {"kerchunk_json", "kerchunk_parquet", "icechunk"}


class ConfigParser:
    """
    Parses and validates MDT YAML configuration files.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The path to the YAML configuration file.
    """

    def __init__(self, config_path: Union[str, pathlib.Path]):
        self.config_path = pathlib.Path(config_path)
        self.config: Dict[str, Any] = self._load_yaml()
        self._validate_config()

    def _load_yaml(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open() as f:
            try:
                config = yaml.safe_load(f)
                return cast(Dict[str, Any], config) if config is not None else {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration: {e}") from e

    def _validate_config(self) -> None:
        """Basic validation to ensure required top-level keys exist."""
        required_keys = ["data"]
        for key in required_keys:
            if key not in self.config:
                logger.error(f"Configuration is missing top-level key: '{key}'.")
                raise ValueError(f"Configuration is missing top-level key: '{key}'.")

        # Ensure 'data' is a dictionary if it exists
        if "data" in self.config:
            if not isinstance(self.config["data"], dict):
                raise ValueError("'data' section in configuration must be a dictionary.")
            for name, details in self.config["data"].items():
                if not isinstance(details, dict) or "type" not in details:
                    raise ValueError(f"Data source '{name}' must have a 'type' field.")

                zarr_store = details.get("zarr_store")
                if zarr_store is not None:
                    if not isinstance(zarr_store, dict):
                        raise ValueError(f"Data source '{name}': 'zarr_store' must be a mapping.")
                    enabled = zarr_store.get("enabled", False)
                    if enabled:
                        backend = zarr_store.get("backend", "kerchunk_json")
                        if backend not in VALID_ZARR_BACKENDS:
                            raise ValueError(
                                f"Data source '{name}': unsupported zarr_store.backend '{backend}'. Supported: {sorted(VALID_ZARR_BACKENDS)}"
                            )
                        if backend == "icechunk" and not zarr_store.get("icechunk_repo"):
                            raise ValueError(f"Data source '{name}': 'icechunk_repo' is required when zarr_store.backend is 'icechunk'.")

        if "pairing" in self.config:
            if not isinstance(self.config["pairing"], dict):
                raise ValueError("'pairing' section in configuration must be a dictionary.")
            for name, details in self.config["pairing"].items():
                if not isinstance(details, dict):
                    raise ValueError(f"Pairing '{name}' must be a dictionary.")
                if "source" not in details or "target" not in details:
                    raise ValueError(f"Pairing '{name}' must specify both 'source' and 'target'.")

        if "statistics" in self.config:
            if not isinstance(self.config["statistics"], dict):
                raise ValueError("'statistics' section in configuration must be a dictionary.")
            for name, details in self.config["statistics"].items():
                if not isinstance(details, dict):
                    raise ValueError(f"Statistics task '{name}' must be a dictionary.")
                if "input" not in details:
                    raise ValueError(f"Statistics task '{name}' must specify an 'input'.")

        if "plots" in self.config:
            if not isinstance(self.config["plots"], dict):
                raise ValueError("'plots' section in configuration must be a dictionary.")
            for name, details in self.config["plots"].items():
                if not isinstance(details, dict):
                    raise ValueError(f"Plot task '{name}' must be a dictionary.")
                if "input" not in details:
                    raise ValueError(f"Plot task '{name}' must specify an 'input'.")

    @property
    def data(self) -> Dict[str, Any]:
        """dict: The 'data' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("data", {}))

    @property
    def pairing(self) -> Dict[str, Any]:
        """dict: The 'pairing' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("pairing", {}))

    @property
    def combine(self) -> Dict[str, Any]:
        """dict: The 'combine' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("combine", {}))

    @property
    def statistics(self) -> Dict[str, Any]:
        """dict: The 'statistics' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("statistics", {}))

    @property
    def plots(self) -> Dict[str, Any]:
        """dict: The 'plots' section of the configuration."""
        return cast(Dict[str, Any], self.config.get("plots", {}))

    @property
    def orchestrator(self) -> str:
        """str: The orchestrator backend name from the execution section, defaults to 'prefect'."""
        return str(self.execution.get("orchestrator", "prefect"))

    @property
    def execution(self) -> Dict[str, Any]:
        """
        Retrieves the execution configuration, which can define multiple clusters.

        If not specified, defaults to a single local cluster named 'compute'.

        When the orchestrator is ``"ecflow"``, the following ecFlow-specific
        keys are accessible directly from the returned dict:

        - ``ecflow_host`` — ecFlow server hostname (default ``"localhost"``)
        - ``ecflow_port`` — ecFlow server port (default ``3141``)
        - ``suite_name``  — ecFlow suite name (default ``"mdt"``)
        - ``task_script_dir`` — directory for generated ``.ecf`` wrapper
          scripts (default ``"./ecflow_tasks/"``)

        These keys are read by
        :class:`~mdt.ecflow_engine.EcFlowEngine` at construction time.

        Returns
        -------
        dict
            The execution configuration section, including any
            orchestrator-specific keys defined in the YAML.
        """
        exec_cfg: Dict[str, Any] = self.config.get("execution", {})
        if not exec_cfg:
            return {"default_cluster": "compute", "clusters": {"compute": {"mode": "local"}}}

        # Backwards compatibility: if they just provided 'mode', wrap it in a 'compute' cluster.
        # Preserve any non-cluster keys (e.g. orchestrator, ecflow_host, etc.)
        # at the top level so engine implementations can still access them.
        if "mode" in exec_cfg and "clusters" not in exec_cfg:
            cluster_cfg = {k: v for k, v in exec_cfg.items() if k == "mode" or k == "workers"}
            result = {k: v for k, v in exec_cfg.items() if k not in ("mode", "workers")}
            result["default_cluster"] = "compute"
            result["clusters"] = {"compute": cluster_cfg if cluster_cfg else exec_cfg}
            return result

        # Ensure default_cluster is set if clusters are defined
        if "clusters" in exec_cfg and "default_cluster" not in exec_cfg:
            exec_cfg["default_cluster"] = next(iter(exec_cfg["clusters"].keys()))

        # Ensure a service cluster is always defined for service-node tasks (like data download)
        if "service" not in exec_cfg.get("clusters", {}):
            # Attempt to derive the service mode from the default compute cluster's mode
            compute_mode = "local"
            if "clusters" in exec_cfg and exec_cfg["default_cluster"] in exec_cfg["clusters"]:
                compute_mode = exec_cfg["clusters"][exec_cfg["default_cluster"]].get("mode", "local")

            # If the primary execution is an HPC scheduler, we should map the service cluster to it
            if compute_mode != "local":
                service_mode = compute_mode
            else:
                service_mode = "local"

            exec_cfg.setdefault("clusters", {})["service"] = {"mode": service_mode, "workers": 1}

        return exec_cfg

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a specific key from the raw configuration.

        Parameters
        ----------
        key : str
            The configuration key to retrieve.
        default : any, optional
            The default value to return if the key is not found, by default None.

        Returns
        -------
        any
            The value associated with the key, or the default value.
        """
        return self.config.get(key, default)


def load_config(config_path: Union[str, pathlib.Path]) -> ConfigParser:
    """
    Helper function to quickly initialize a ConfigParser from a path.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The path to the YAML configuration file.

    Returns
    -------
    ConfigParser
        An instantiated and validated parser object.
    """
    return ConfigParser(config_path)
