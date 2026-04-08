import inspect
import logging
from typing import Any, Dict, List, Union

import monet_stats
import pandas as pd
import xarray as xr

from mdt.utils import discover_spatial_dims, update_history

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

    Examples
    --------
    >>> metrics = ['rmse', 'mb']
    >>> kwargs = {'obs_var': 'obs', 'mod_var': 'mod', 'weights': 'w'}
    >>> results = compute_statistics('test_stats', metrics, ds, kwargs)
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
    """
    Robust discovery of metric functions in monet-stats.

    Parameters
    ----------
    module : Any
        The module to search for the metric (usually monet_stats).
    metric_name : str
        The name of the metric to find (case-insensitive).

    Returns
    -------
    Any or None
        The metric function if found, else None.

    Examples
    --------
    >>> import monet_stats
    >>> func = _find_metric(monet_stats, 'rmse')
    """
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

    This function orchestrates the execution by resolving weights and
    dimensions, and checking for native backend support via inspection.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray or pd.DataFrame
        The data object to process.
    func : Any
        The metric function to execute.
    kwargs : Dict[str, Any]
        Parameters for the metric execution, including obs/mod variable names
        and weighting options.

    Returns
    -------
    xr.Dataset or xr.DataArray or pd.Series
        The computed metric result.

    Examples
    --------
    >>> import monet_stats
    >>> res = _execute_metric(ds, monet_stats.rmse, {'obs_var': 'o', 'mod_var': 'm'})

    Notes
    -----
    Aero Protocol: Preserves Dask laziness and ensures backend-agnostic behavior.
    """
    obs_var = kwargs.get("obs_var", "obs")
    mod_var = kwargs.get("mod_var", "mod")
    weights = kwargs.get("weights")
    dim = kwargs.get("dim")

    # Filter out MDT-specific keys for the standard metric call
    call_kwargs = {k: v for k, v in kwargs.items() if k not in ["obs_var", "mod_var", "weights", "dim", "chunks"]}

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        # Optional Chunking Optimization (Aero Protocol Rule 1.3 - User-requested or Auto)
        chunks = kwargs.get("chunks")
        if chunks is not None:
            if chunks == "auto":
                chunks = monet_stats.get_chunk_recommendation(data)
                logger.info("Auto-chunking recommended: %s", chunks)
            data = data.chunk(chunks)

        # 1. Extract Target Data
        if isinstance(data, xr.Dataset):
            target_obs = data[obs_var]
            target_mod = data[mod_var]
        else:
            target_obs = None  # Not used for single DataArray metrics
            target_mod = data

        # 2. Resolve Weights and Dimensions
        w = None
        if weights is not None:
            if isinstance(weights, str):
                if isinstance(data, xr.Dataset) and weights in data:
                    w = data[weights]
                elif isinstance(data, xr.DataArray) and weights in data.coords:
                    w = data.coords[weights]
            else:
                w = weights

        # Resolve axis/dim for monet-stats (standardizing on 'axis' for metrics)
        axis = dim
        if axis is None:
            # Try to discover spatial dims if none provided
            lat_dim, lon_dim = discover_spatial_dims(data)
            axis = [d for d in [lat_dim, lon_dim] if d is not None]
            if not axis:
                axis = None

        # Check function signature for native support
        sig = inspect.signature(func)
        has_weights = "weights" in sig.parameters
        axis_param = "axis" if "axis" in sig.parameters else ("dim" if "dim" in sig.parameters else None)

        # 3. Execution (Orchestrator Rule - Delegate to monet-stats)
        result = None

        # Resolve axis/dim for monet-stats (standardizing on 'axis' for metrics)
        if axis_param:
            call_kwargs[axis_param] = axis

        # Handle Weighted Calculations
        if w is not None:
            if has_weights:
                # Direct call if metric supports weights natively
                if target_obs is not None:
                    result = func(target_obs, target_mod, weights=w, **call_kwargs)
                else:
                    result = func(target_mod, weights=w, **call_kwargs)
            else:
                # Search for weighted counterpart in monet-stats (e.g., MB -> WDMB)
                metric_name = getattr(func, "__name__", "").upper()
                wd_name = f"WD{metric_name}"
                wd_func = _find_metric(monet_stats, wd_name)

                if wd_func:
                    logger.debug("Delegating to weighted counterpart: %s", wd_name)
                    wd_sig = inspect.signature(wd_func)
                    if "weights" in wd_sig.parameters:
                        call_kwargs["weights"] = w

                    if target_obs is not None:
                        result = wd_func(target_obs, target_mod, **call_kwargs)
                    else:
                        result = wd_func(target_mod, **call_kwargs)
                else:
                    logger.warning(
                        "Metric '%s' does not support weights natively and no WD counterpart found in monet-stats.",
                        metric_name,
                    )

        # 4. Standard Call (Unweighted or Fallback)
        if result is None:
            if target_obs is not None:
                result = func(target_obs, target_mod, **call_kwargs)
            else:
                result = func(target_mod, **call_kwargs)

        # Apply provenance for optimization if it happened and survived
        if chunks is not None and isinstance(result, (xr.Dataset, xr.DataArray)):
            result = update_history(result, f"Optimized chunking with: {chunks}")
        return result

    elif isinstance(data, pd.DataFrame):
        obs = data[obs_var]
        mod = data[mod_var]
        return func(obs, mod, **call_kwargs)

    return func(data, **call_kwargs)
