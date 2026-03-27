import logging
from typing import Union

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
    kwargs: dict,
) -> Union[xr.Dataset, xr.DataArray, pd.DataFrame]:
    """
    Dynamically pair two datasets using monet regridding or interpolation.

    Parameters
    ----------
    name : str
        The identifier for this pairing task.
    method : str
        The regridding method to use (e.g., 'interpolate', 'regrid', 'point_to_grid').
    source_data : xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The source data object (typically a model).
    target_data : xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The target data object or grid (typically observations or a reference grid).
    kwargs : dict
        Additional keyword arguments to pass to the underlying pairing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The resulting paired dataset.

    Raises
    ------
    ValueError
        If an unknown pairing method is specified.

    Examples
    --------
    >>> paired = pair_data(
    ...     "my_pairing", "interpolate", model_ds, obs_df, {"interp_kw": "val"}
    ... )
    """
    logger.info("Pairing data '%s' using method '%s'", name, method)

    import monet.util.interp_util as interp_util

    try:
        if method == "interpolate":
            # Check for a standard interpolation function in monet
            func = getattr(interp_util, "points_to_dataset", None)
            if func is None:
                # Fallback to nearest neighbor if generic not found
                func = getattr(interp_util, "nearest_point_swathdefinition", None)

            if func:
                paired_data = func(source_data, target_data, **kwargs)
            else:
                # If everything fails, this is likely a custom regridding needed
                raise AttributeError("Could not find suitable interpolation function in monet.")

        elif method == "regrid":
            import xregrid

            regridder = xregrid.Regridder(source_data, target_data, **kwargs)
            paired_data = regridder.regrid(source_data)

            # If user wants to compare source vs target (e.g. model-to-model),
            # merge the target dataset into the result.
            if kwargs.get("merge_target", False):
                # Consume these keys so they don't break regridding logic if passed there?
                # Actually regridder uses kwargs. But we checked above.
                # Assuming simple kwargs filtering or lenience in xregrid.
                # Let's pop them if possible, but kwargs is passed to Regridder above.
                pass

            # Safe access
            merge_flag = kwargs.get("merge_target", False)
            if merge_flag:
                suffix_source = kwargs.get("suffix_source", "_source")
                suffix_target = kwargs.get("suffix_target", "_target")

                # Rename source variables
                renamed_source = paired_data.rename({v: f"{v}{suffix_source}" for v in paired_data.data_vars})

                # Rename target variables (make a copy to not affect original loaded data)
                renamed_target = target_data.copy().rename({v: f"{v}{suffix_target}" for v in target_data.data_vars})

                paired_data = xr.merge([renamed_source, renamed_target])

        else:
            raise ValueError(f"Unknown pairing method '{method}'.")

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
    paired_data: dict,
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
        combined = pd.concat(dfs, ignore_index=True)
        return combined

    elif isinstance(first_item, (xr.Dataset, xr.DataArray)):
        datasets = []
        names = []
        for name, ds in paired_data.items():
            if not isinstance(ds, type(first_item)):
                raise TypeError(f"All items must be of the same type. Found {type(ds)} and {type(first_item)}")

            datasets.append(ds)
            names.append(name)

        # Combine using xarray concatenation along a new dimension
        try:
            # Create the dimension coordinate
            combined = xr.concat(datasets, dim=pd.Index(names, name=dim))
            return combined
        except Exception as e:
            logger.error(f"Failed to combine xarray objects: {e}")
            raise

    else:
        raise TypeError(f"Unsupported data type for combination: {type(first_item)}")
