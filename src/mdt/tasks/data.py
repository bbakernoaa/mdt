import logging
from typing import Any, Dict, Union, cast

import pandas as pd
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def load_data(
    name: str,
    dataset_type: str,
    kwargs: Dict[str, Any],
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Unified entry point for loading data via monetio.load.

    Supports standard loading and VirtualiZarr-based ingestion (kerchunk/icechunk).
    If VirtualiZarr parameters cause a TypeError (e.g., old reader version),
    it automatically falls back to standard loading.

    Parameters
    ----------
    name : str
        A user-defined name for this dataset.
    dataset_type : str
        The dataset type/reader name recognized by monetio (e.g., 'cmaq', 'aeronet').
    kwargs : dict
        Keyword arguments passed to monetio.load.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        The loaded data object.

    Notes
    -----
    Aero Protocol: Ensures that data is loaded lazily where supported.
    """
    logger.info("Loading dataset '%s' of type '%s'", name, dataset_type)

    import monetio

    # 1. Attempt VirtualiZarr Loading if enabled
    use_virtualizarr = kwargs.get("use_virtualizarr", False)
    if use_virtualizarr:
        backend = kwargs.get("virtualizarr_backend", "kerchunk_json")
        store_path = kwargs.get("store_path", f"./zarr_stores/{name}/")
        icechunk_repo = kwargs.get("icechunk_repo", "")

        logger.info(
            "Attempting VirtualiZarr load for '%s' (backend=%s, store=%s, icechunk_repo=%s)",
            name,
            backend,
            store_path,
            icechunk_repo,
        )

        try:
            # Call monetio.load with all VirtualiZarr parameters
            ds = monetio.load(dataset_type, **kwargs)
            logger.info("Successfully loaded '%s' via VirtualiZarr pathway.", name)

            # Add MDT provenance tracking
            msg = f"Loaded dataset '{name}' via VirtualiZarr (backend={backend})."
            ds = update_history(ds, msg)
            return cast(Union[xr.Dataset, pd.DataFrame], ds)

        except TypeError as e:
            # Handle cases where reader doesn't support VirtualiZarr kwargs yet
            logger.warning(
                "VirtualiZarr load failed for '%s' due to unexpected arguments: %s. "
                "Retrying with standard monetio.load.",
                name,
                e,
            )
            # Fallback path: strip the VirtualiZarr-specific keys
            fallback_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["use_virtualizarr", "virtualizarr_backend", "store_path", "icechunk_repo"]
            }
            ds = monetio.load(dataset_type, **fallback_kwargs)
            msg = f"Loaded dataset '{name}' (VirtualiZarr fallback used)."
            ds = update_history(ds, msg)
            return cast(Union[xr.Dataset, pd.DataFrame], ds)

    # 2. Standard Loading Pathway
    try:
        ds = monetio.load(dataset_type, **kwargs)

        # Provenance Tracking
        msg = f"Loaded dataset '{name}' using type '{dataset_type}' with params {kwargs}."
        ds = update_history(ds, msg)

        logger.info("Successfully loaded dataset '%s'", name)
        return cast(Union[xr.Dataset, pd.DataFrame], ds)

    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", name, e)
        raise
