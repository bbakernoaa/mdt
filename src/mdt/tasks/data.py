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

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        The loaded dataset object returned by monetio.

    Examples
    --------
    >>> ds = load_data("my_merra2", "merra2", {"dates": "2023-01-01"})
    """
    logger.info(f"Loading data '{name}' using monetio.load('{dataset_type}')")

    try:
        import monetio

        # Use the universal load function if available
        if hasattr(monetio, "load"):
            dataset = monetio.load(dataset_type, **kwargs)
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

        # Provenance Tracking
        msg = f"Loaded dataset '{name}' of type '{dataset_type}' with params {kwargs}."
        dataset = update_history(dataset, msg)

        logger.info(f"Successfully loaded data '{name}'")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load data '{name}': {e}")
        raise
