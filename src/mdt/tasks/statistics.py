import logging
from typing import Any, Dict, List, Union

import numpy as np
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


def _execute_metric(
    data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    func: Any,
    kwargs: Dict[str, Any],
) -> Union[xr.Dataset, xr.DataArray, pd.Series]:
    """
    Execute metric from monet-stats with native Xarray/Dask support.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray or pd.DataFrame
        The input data for the metric.
    func : Any
        The metric function from monet_stats.
    kwargs : Dict[str, Any]
        Keyword arguments for the metric and orchestration.

    Returns
    -------
    xr.Dataset or xr.DataArray or pd.Series
        The computed metric result.
    """
    import monet_stats

    obs_var = kwargs.get("obs_var", "obs")
    mod_var = kwargs.get("mod_var", "mod")
    weights = kwargs.get("weights")

    # Filter out MDT-specific keys for the standard metric call
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

            # Map 'dim' keyword to 'lat_dim'/'lon_dim' for weighted_spatial_mean
            # Rule: Do not include 'weights' in w_kwargs as it is passed explicitly.
            w_kwargs = {k: v for k, v in call_kwargs.items() if k in ["lat_dim", "lon_dim"]}
            if "dim" in call_kwargs:
                dims = call_kwargs["dim"]
                if isinstance(dims, str):
                    dims = [dims]
                for d in dims:
                    d_lower = d.lower()
                    if "lat" in d_lower and "lat_dim" not in w_kwargs:
                        w_kwargs["lat_dim"] = d
                    if "lon" in d_lower and "lon_dim" not in w_kwargs:
                        w_kwargs["lon_dim"] = d

            # Orchestrator Rule: Push weighted reductions through monet-stats engine.
            # We delegate to weighted_spatial_mean as the primary calculation engine.
            metric_name = getattr(func, "__name__", "").upper()
            if target_obs is not None:
                if metric_name == "MB":
                    return monet_stats.weighted_spatial_mean(target_mod - target_obs, weights=w, **w_kwargs)
                if metric_name == "MAE":
                    return monet_stats.weighted_spatial_mean(abs(target_mod - target_obs), weights=w, **w_kwargs)
                if metric_name == "MSE":
                    return monet_stats.weighted_spatial_mean((target_mod - target_obs) ** 2, weights=w, **w_kwargs)
                if metric_name == "RMSE":
                    mse = monet_stats.weighted_spatial_mean((target_mod - target_obs) ** 2, weights=w, **w_kwargs)
                    return np.sqrt(mse)
                if metric_name in ["CORR", "PEARSONR", "CORRELATION"]:
                    mu_mod = monet_stats.weighted_spatial_mean(target_mod, weights=w, **w_kwargs)
                    mu_obs = monet_stats.weighted_spatial_mean(target_obs, weights=w, **w_kwargs)

                    dev_mod = target_mod - mu_mod
                    dev_obs = target_obs - mu_obs

                    cov = monet_stats.weighted_spatial_mean(dev_mod * dev_obs, weights=w, **w_kwargs)
                    var_mod = monet_stats.weighted_spatial_mean(dev_mod**2, weights=w, **w_kwargs)
                    var_obs = monet_stats.weighted_spatial_mean(dev_obs**2, weights=w, **w_kwargs)

                    return cov / np.sqrt(var_mod * var_obs)

        # 3. Standard Fallback
        if isinstance(data, xr.Dataset):
            return func(target_obs, target_mod, **call_kwargs)
        return func(data, **call_kwargs)

    elif isinstance(data, pd.DataFrame):
        obs = data[obs_var]
        mod = data[mod_var]
        return func(obs, mod, **call_kwargs)

    return func(data, **call_kwargs)
