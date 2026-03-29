"""Module for parsing and validating MDT YAML configuration files."""

import logging
import pathlib

import yaml

logger = logging.getLogger(__name__)


class ConfigParser:
    """
    Parses and validates MDT YAML configuration files.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The path to the YAML configuration file.
    """

    def __init__(self, config_path):
        self.config_path = pathlib.Path(config_path)
        self.config = self._load_yaml()
        self._validate_config()

    def _load_yaml(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            try:
                config = yaml.safe_load(f)
                return config if config is not None else {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration: {e}")

    def _validate_config(self):
        """Basic validation to ensure required top-level keys exist."""
        required_keys = ["data"]
        for key in required_keys:
            if key not in self.config:
                logger.warning(f"Configuration is missing top-level key: '{key}'.")

        # Ensure 'data' is a dictionary if it exists
        if "data" in self.config and not isinstance(self.config["data"], dict):
            raise ValueError("'data' section in configuration must be a dictionary.")

    @property
    def data(self):
        """dict: The 'data' section of the configuration."""
        return self.config.get("data", {})

    @property
    def pairing(self):
        """dict: The 'pairing' section of the configuration."""
        return self.config.get("pairing", {})

    @property
    def combine(self):
        """dict: The 'combine' section of the configuration."""
        return self.config.get("combine", {})

    @property
    def statistics(self):
        """dict: The 'statistics' section of the configuration."""
        return self.config.get("statistics", {})

    @property
    def plots(self):
        """dict: The 'plots' section of the configuration."""
        return self.config.get("plots", {})

    @property
    def execution(self):
        """
        Retrieves the execution configuration, which can define multiple clusters.

        If not specified, defaults to a single local cluster named 'compute'.

        Returns
        -------
        dict
            The execution configuration section.
        """
        exec_cfg = self.config.get("execution", {})
        if not exec_cfg:
            return {"default_cluster": "compute", "clusters": {"compute": {"mode": "local"}}}

        # Backwards compatibility: if they just provided 'mode', wrap it in a 'compute' cluster
        if "mode" in exec_cfg and "clusters" not in exec_cfg:
            return {"default_cluster": "compute", "clusters": {"compute": exec_cfg}}

        # Ensure default_cluster is set if clusters are defined
        if "clusters" in exec_cfg and "default_cluster" not in exec_cfg:
            exec_cfg["default_cluster"] = list(exec_cfg["clusters"].keys())[0]

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

    def get(self, key, default=None):
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


def load_config(config_path):
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
