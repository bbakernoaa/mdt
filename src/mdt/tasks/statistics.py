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

    Leverages xarray.apply_ufunc with dask='parallelized' to ensure
    computations remain lazy if inputs are Dask-backed.

    Parameters
    ----------
    name : str
        The identifier for this statistics task.
    metrics : List[str]
        A list of metric names to compute (e.g., ['rmse', 'bias', 'corr']).
    input_data : xr.Dataset or xr.DataArray or pd.DataFrame
        The paired dataset containing both model and observational data.
    kwargs : Dict[str, Any]
        Additional keyword arguments passed to the monet_stats metric functions
        (e.g., `obs_var`, `mod_var`, or group-by parameters).

    Returns
    -------
    Dict[str, Union[xr.Dataset, xr.DataArray, pd.Series]]
        A dictionary mapping the computed metric names to their results.

    Raises
    ------
    ImportError
        If monet-stats is not installed.
    AttributeError
        If a specified metric is not found in monet-stats.

    Examples
    --------
    >>> stats = compute_statistics(
    ...     "model_eval", ["rmse", "mae"], ds, {"obs_var": "obs", "mod_var": "mod"}
    ... )
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

                # Provenance Tracking
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
    except Exception as e:
        logger.error("An unexpected error occurred during statistics computation: %s", e)
        raise


def _find_metric(module: Any, metric_name: str) -> Any:
    """
    Robust discovery of metric functions in monet-stats.

    Parameters
    ----------
    module : Any
        The monet-stats module or submodule to search.
    metric_name : str
        The name of the metric to find (case-insensitive).

    Returns
    -------
    Any
        The callable metric function if found, otherwise None.

    Examples
    --------
    >>> import monet_stats
    >>> func = _find_metric(monet_stats, "RMSE")
    """
    # 1. Direct attribute access (case-sensitive)
    if hasattr(module, metric_name):
        return getattr(module, metric_name)

    # 2. Case-insensitive search in top-level
    for attr in dir(module):
        if attr.lower() == metric_name.lower():
            return getattr(module, attr)

    # 3. Search in .stats submodule if it exists
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
        The input data to compute metrics on.
    func : Any
        The callable metric function from monet-stats.
    kwargs : Dict[str, Any]
        Additional keyword arguments for the metric function.

    Returns
    -------
    xr.Dataset or xr.DataArray or pd.Series
        The computed metric result.

    Examples
    --------
    >>> res = _execute_metric(ds, monet_stats.rmse, {"obs_var": "obs", "mod_var": "mod"})
    """
    obs_var = kwargs.get("obs_var", "obs")
    mod_var = kwargs.get("mod_var", "mod")

    # Filter out MDT-specific keys before passing to monet-stats
    call_kwargs = {k: v for k, v in kwargs.items() if k not in ["obs_var", "mod_var"]}

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        if isinstance(data, xr.Dataset):
            obs = data[obs_var]
            mod = data[mod_var]

            # Aero Protocol: Use apply_ufunc to ensure backend-agnostic execution.
            # We treat all dimensions as core dimensions to support reduction metrics (like RMSE).
            return xr.apply_ufunc(
                func,
                obs,
                mod,
                input_core_dims=[obs.dims, mod.dims],
                kwargs=call_kwargs,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )
        else:
            # Handle DataArray: treat all dimensions as core dimensions.
            return xr.apply_ufunc(
                func,
                data,
                input_core_dims=[data.dims],
                kwargs=call_kwargs,
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )

    elif isinstance(data, pd.DataFrame):
        obs = data[obs_var]
        mod = data[mod_var]
        return func(obs, mod, **call_kwargs)

    else:
        return func(data, **call_kwargs)
