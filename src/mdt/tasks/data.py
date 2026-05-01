import importlib
import logging
from typing import Union

import pandas as pd
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def load_data(name: str, dataset_type: str, kwargs: dict) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Dynamically loads data using monetio.

    Parameters
    ----------
    name : str
        The configuration identifier for this data.
    dataset_type : str
        The name of the monetio source (e.g., 'cmaq', 'aeronet', 'merra2').
    kwargs : dict
        Additional keyword arguments to pass to the monetio reader.
        When VirtualiZarr is enabled, kwargs may include:
        - use_virtualizarr (bool): Enable VirtualiZarr-based loading.
        - virtualizarr_backend (str): Backend format for virtual references.
        - store_path (str): Path to the virtual Zarr store.
        - icechunk_repo (str): Icechunk repository path (when backend is icechunk).

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        The loaded dataset object returned by monetio.

    Examples
    --------
    >>> ds = load_data("my_merra2", "merra2", {"dates": "2023-01-01"})
    """
    logger.info(f"Loading data '{name}' using monetio.load('{dataset_type}')")

    # Extract VirtualiZarr-specific keys for logging and fallback
    virtualizarr_keys = [
        "use_virtualizarr",
        "virtualizarr_backend",
        "store_path",
        "icechunk_repo",
    ]
    virtualizarr_params = {}
    for key in virtualizarr_keys:
        if key in kwargs:
            virtualizarr_params[key] = kwargs[key]

    use_virtualizarr = virtualizarr_params.get("use_virtualizarr", False)

    if use_virtualizarr:
        logger.info(
            f"Loading '{name}' with VirtualiZarr: "
            f"backend={virtualizarr_params.get('virtualizarr_backend')}, "
            f"store_path={virtualizarr_params.get('store_path')}, "
            f"icechunk_repo={virtualizarr_params.get('icechunk_repo', 'N/A')}"
        )

    try:
        import monetio

        # Use the universal load function if available
        if hasattr(monetio, "load"):
            try:
                dataset = monetio.load(dataset_type, **kwargs)
            except Exception as e:
                if use_virtualizarr:
                    logger.warning(f"VirtualiZarr load failed for '{name}': {e}. Retrying without VirtualiZarr parameters.")
                    fallback_kwargs = {k: v for k, v in kwargs.items() if k not in virtualizarr_keys}
                    dataset = monetio.load(dataset_type, **fallback_kwargs)
                else:
                    raise
        else:
            # Fallback for older versions
            # Dynamically import the specific monetio reader
            module_path = f"monetio.datasets.{dataset_type}"
            try:
                reader_module = importlib.import_module(module_path)
            except ImportError:
                module_path = f"monetio.readers.{dataset_type}"
                reader_module = importlib.import_module(module_path)

            if hasattr(reader_module, "open_dataset"):
                func = reader_module.open_dataset
            elif hasattr(reader_module, "open"):
                func = reader_module.open
            elif hasattr(reader_module, "open_mfdataset"):
                func = reader_module.open_mfdataset
            else:
                raise AttributeError(f"Could not find a standard open function in {module_path}")

            dataset = func(**kwargs)

        if use_virtualizarr:
            logger.info(f"Successfully loaded '{name}' with VirtualiZarr enabled.")

        # Provenance Tracking
        msg = f"Loaded dataset '{name}' of type '{dataset_type}' with params {kwargs}."
        dataset = update_history(dataset, msg)

        logger.info(f"Successfully loaded data '{name}'")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load data '{name}': {e}")
        raise
