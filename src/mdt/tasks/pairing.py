import logging
from typing import Any, Dict, Union, cast

import pandas as pd
import xarray as xr

from mdt.utils import update_history

# Monet pairing is usually handled by `monet.models.*` or `monet.obs.*` depending on the object,
# or through `monet.util.interp_util` regridding.

logger = logging.getLogger(__name__)


def pair_data(
    name: str,
    method: str,
    source_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    target_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    kwargs: Dict[str, Any],
) -> Union[xr.Dataset, xr.DataArray, pd.DataFrame]:
    """
    Dynamically pair two datasets using monet.

    Parameters
    ----------
    name : str
        The identifier for this pairing task.
    method : str
        Spatial interpolation method (e.g., 'nearest', 'bilinear', 'conservative').
    source_data : xarray.Dataset or xarray.DataArray
        The source data object (typically a model).
    target_data : xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The target data object or grid (typically observations).
    kwargs : dict
        Additional keyword arguments for the pairing function (e.g., 'interp_time', 'suffix').

    Returns
    -------
    xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The resulting paired dataset.

    Raises
    ------
    ImportError
        If required pairing backend is not installed.

    Examples
    --------
    >>> paired = pair_data(
    ...     "my_pairing", "bilinear", model_ds, obs_df, {"interp_time": True}
    ... )
    """
    logger.info("Pairing data '%s' using method '%s'", name, method)

    import monet

    try:
        # Use the unified monet.pair interface
        # This handles both Xarray-to-Xarray and Xarray-to-DataFrame pairing,
        # maintaining Aero Protocol (laziness).
        paired_data = monet.util.combinetool.pair(source_data, target_data, method=method, **kwargs)

        # Provenance Tracking
        msg = f"Paired using method '{method}' with params {kwargs}."
        paired_data = update_history(paired_data, msg)

        logger.info("Successfully paired data '%s'", name)
        return paired_data

    except ImportError as e:
        logger.error("Required package for pairing not found: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to pair data '%s': %s", name, e)
        raise


def combine_paired_data(
    paired_data: Dict[str, Union[pd.DataFrame, xr.Dataset, xr.DataArray]],
    dim: str = "model",
) -> Union[pd.DataFrame, xr.Dataset, xr.DataArray]:
    """
    Combines multiple paired datasets into a single dataset.

    This is useful for creating a single dataset containing multiple model runs
    alongside observations, facilitating comparative analysis and plotting.

    Parameters
    ----------
    paired_data : dict
        A dictionary mapping model names/identifiers to their corresponding paired data objects.
    dim : str, optional
        The name of the dimension or column to use for the model identifier. Default is 'model'.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset or xarray.DataArray
        The combined dataset.

    Raises
    ------
    ValueError
        If no data is provided.
    TypeError
        If the data types are inconsistent or unsupported.
    """
    if not paired_data:
        raise ValueError("No paired data provided for combination.")

    first_key = next(iter(paired_data))
    first_item = paired_data[first_key]

    logger.info(f"Combining {len(paired_data)} paired datasets along dimension '{dim}'")

    if isinstance(first_item, pd.DataFrame):
        dfs = []
        for name, df in paired_data.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected DataFrame for '{name}', got {type(df)}")
            # Avoid modifying the original dataframe in place if checking equality elsewhere
            df_copy = df.copy()
            df_copy[dim] = name  # Add the identifier column
            dfs.append(df_copy)

        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    elif isinstance(first_item, (xr.Dataset, xr.DataArray)):
        datasets = []
        names = []
        for name, ds in paired_data.items():
            if not isinstance(ds, (xr.Dataset, xr.DataArray)):
                raise TypeError(f"All items must be xarray. Found {type(ds)}")

            datasets.append(ds)
            names.append(name)

        # Combine using xarray concatenation along a new dimension
        try:
            # Create the dimension coordinate
            combined_xr = xr.concat(datasets, dim=pd.Index(names, name=dim))  # type: ignore[arg-type]
            return cast(Union[xr.Dataset, xr.DataArray], combined_xr)
        except Exception as e:
            logger.error(f"Failed to combine xarray objects: {e}")
            raise

    else:
        raise TypeError(f"Unsupported data type for combination: {type(first_item)}")
