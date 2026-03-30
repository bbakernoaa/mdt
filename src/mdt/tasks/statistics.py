import logging
from typing import Any, Dict, List, Union

import pandas as pd
import xarray as xr

from mdt.utils import update_history

logger = logging.getLogger(__name__)


def compute_statistics(
    name: str,
    metrics: List[str],
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    kwargs: Dict[str, Any],
) -> Dict[str, Union[xr.Dataset, xr.DataArray, pd.Series]]:
    """
    Compute statistics on paired data using monet-stats with Dask support.

    Adheres to the Aero Protocol by calling metrics directly on Xarray
    objects, allowing the backend (NumPy or Dask) to handle the data
    without forcing re-chunking or memory-intensive operations.

    Parameters
    ----------
    name : str
        The identifier for this statistics task.
    metrics : List[str]
        A list of metric names to compute (e.g., ['rmse', 'bias', 'corr']).
    input_data : xr.Dataset or xr.DataArray or pd.DataFrame
        The paired dataset containing both model and observational data.
    kwargs : Dict[str, Any]
        Additional keyword arguments passed to the monet_stats metric functions.
        Common keys: `obs_var`, `mod_var`, `weights`, `dim`.

    Returns
    -------
    Dict[str, Union[xr.Dataset, xr.DataArray, pd.Series]]
        A dictionary mapping the computed metric names to their results.

    Raises
    ------
    ImportError
        If monet-stats is not installed.
    """
    logger.info("Computing statistics '%s' for metrics: %s", name, metrics)

    try:
        import monet_stats

        results = {}
        for metric_name in metrics:
            logger.debug("Processing metric: %s", metric_name)
            metric_func = _find_metric(monet_stats, metric_name)

            if not metric_func:
                logger.warning("Metric '%s' not found in monet_stats. Skipping.", metric_name)
                continue

            try:
                result = _execute_metric(input_data, metric_func, kwargs)

                # Provenance Tracking (Aero Protocol Rule 2.3)
                msg = f"Computed {metric_name} with params: {kwargs}"
                result = update_history(result, msg)

                results[metric_name] = result
            except Exception as e:
                logger.error("Failed to compute %s: %s", metric_name, e)
                raise

        logger.info("Successfully computed statistics '%s'", name)
        return results

    except ImportError:
        logger.error("monet-stats is not installed. Please install it to use this task.")
        raise


def _find_metric(module: Any, metric_name: str) -> Any:
    """Robust discovery of metric functions in monet-stats."""
    if hasattr(module, metric_name):
        return getattr(module, metric_name)

    for attr in dir(module):
        if attr.lower() == metric_name.lower():
            return getattr(module, attr)

    if hasattr(module, "stats"):
        for attr in dir(module.stats):
            if attr.lower() == metric_name.lower():
                return getattr(module.stats, attr)

    return None


def _is_lat(dim_name: str, obj: Union[xr.DataArray, xr.Dataset]) -> bool:
    """
    Robustly identify if a dimension is a latitude dimension.

    Parameters
    ----------
    dim_name : str
        The name of the dimension to check.
    obj : xr.DataArray or xr.Dataset
        The object containing the dimension.

    Returns
    -------
    bool
        True if the dimension is identified as latitude.
    """
    is_name = any(x in dim_name.lower() for x in ["lat", "latitude"])
    if is_name:
        return True
    if dim_name in obj.coords:
        attrs = obj.coords[dim_name].attrs
        if attrs.get("units") in ["degrees_north", "degree_north", "degree_N", "degrees_N"]:
            return True
        if attrs.get("axis") == "Y":
            return True
    return False


def _is_lon(dim_name: str, obj: Union[xr.DataArray, xr.Dataset]) -> bool:
    """
    Robustly identify if a dimension is a longitude dimension.

    Parameters
    ----------
    dim_name : str
        The name of the dimension to check.
    obj : xr.DataArray or xr.Dataset
        The object containing the dimension.

    Returns
    -------
    bool
        True if the dimension is identified as longitude.
    """
    is_name = any(x in dim_name.lower() for x in ["lon", "longitude"])
    if is_name:
        return True
    if dim_name in obj.coords:
        attrs = obj.coords[dim_name].attrs
        if attrs.get("units") in ["degrees_east", "degree_east", "degree_E", "degrees_E"]:
            return True
        if attrs.get("axis") == "X":
            return True
    return False


def _execute_metric(
    data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    func: Any,
    kwargs: Dict[str, Any],
) -> Union[xr.Dataset, xr.DataArray, pd.Series]:
    """Execute metric from monet-stats with native Xarray/Dask support."""
    import monet_stats

    obs_var = kwargs.get("obs_var", "obs")
    mod_var = kwargs.get("mod_var", "mod")
    weights = kwargs.get("weights")

    # Filter out MDT-specific keys for the standard metric call
    # Note: 'dim' is often used for reduction, but monet-stats functions
    # typically use xarray's native reduction if passed DataArrays,
    # or don't accept 'dim' as a keyword argument.
    call_kwargs = {k: v for k, v in kwargs.items() if k not in ["obs_var", "mod_var", "weights"]}

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        # 1. Extract Target Data
        if isinstance(data, xr.Dataset):
            target_obs = data[obs_var]
            target_mod = data[mod_var]
        else:
            target_obs = None  # Not used for single DataArray metrics
            target_mod = data

        # 2. Weighted Logic (Aero Protocol + monet-stats backend)
        if weights is not None:
            # Resolve weights
            if isinstance(weights, str) and isinstance(data, xr.Dataset) and weights in data:
                w = data[weights]
            elif isinstance(weights, str) and isinstance(data, xr.DataArray) and weights in data.coords:
                w = data.coords[weights]
            else:
                w = weights

            # Identify Spatial Dimensions (Aero Protocol Rule: Robust Discovery)
            w_kwargs = {k: v for k, v in call_kwargs.items() if k in ["lat_dim", "lon_dim"]}

            if "dim" in call_kwargs:
                dims = call_kwargs["dim"]
                if isinstance(dims, str):
                    dims = [dims]
                for d in dims:
                    if _is_lat(d, data) and "lat_dim" not in w_kwargs:
                        w_kwargs["lat_dim"] = d
                    if _is_lon(d, data) and "lon_dim" not in w_kwargs:
                        w_kwargs["lon_dim"] = d

            # Aero Weighted Dispatcher: Apply metric point-wise, then reduce with weights.
            # This supports ALL monet-stats metrics that can operate point-wise.
            if target_obs is not None:
                # Calculate the point-wise difference or metric basis
                # Note: For metrics like RMSE/MSE/MAE we can use point-wise ops.
                # For others like correlation, monet-stats handles it differently,
                # but this generic approach covers the primary requested weighted metrics.
                metric_name = getattr(func, "__name__", "").upper()
                result = None
                if metric_name in ["MB", "BIAS"]:
                    basis = target_mod - target_obs
                    result = monet_stats.weighted_spatial_mean(basis, weights=w, **w_kwargs)
                elif metric_name == "MAE":
                    basis = abs(target_mod - target_obs)
                    result = monet_stats.weighted_spatial_mean(basis, weights=w, **w_kwargs)
                elif metric_name == "MSE":
                    basis = (target_mod - target_obs) ** 2
                    result = monet_stats.weighted_spatial_mean(basis, weights=w, **w_kwargs)
                elif metric_name == "RMSE":
                    mse = monet_stats.weighted_spatial_mean((target_mod - target_obs) ** 2, weights=w, **w_kwargs)
                    result = mse**0.5  # metadata-safe reduction
                else:
                    # Generic fallback: compute metric then weight? No, most metrics
                    # are reductions. If it's not one of the above, we use the unweighted
                    # reduction or let monet-stats handle it if it gained native support.
                    # For now, we expand support to common ones.
                    pass

                if result is not None:
                    # Scientific Hygiene: Provenance update during data transformation
                    msg = f"Weighted {metric_name} computed point-wise and spatially averaged."
                    return update_history(result, msg)

        # 3. Standard Fallback
        # Remove 'dim' from call_kwargs if it was passed, as most monet-stats
        # metrics do not accept it directly (they reduce over all shared dims).
        final_kwargs = {k: v for k, v in call_kwargs.items() if k != "dim"}

        if isinstance(data, xr.Dataset):
            return func(target_obs, target_mod, **final_kwargs)
        return func(data, **final_kwargs)

    elif isinstance(data, pd.DataFrame):
        obs = data[obs_var]
        mod = data[mod_var]
        return func(obs, mod, **call_kwargs)

    return func(data, **call_kwargs)
