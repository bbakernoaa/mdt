import logging
from typing import Any, Dict, List, Union

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def compute_statistics(
    name: str,
    metrics: List[str],
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    kwargs: Dict[str, Any],
) -> Dict[str, Union[xr.Dataset, xr.DataArray, pd.Series]]:
    """
    Compute statistics on paired data using monet-stats.

    Parameters
    ----------
    name : str
        The identifier for this statistics task.
    metrics : list of str
        A list of metric names to compute (e.g., ['rmse', 'bias', 'corr']).
    input_data : xarray.Dataset or xarray.DataArray or pandas.DataFrame
        The paired dataset containing both model and observational data.
    kwargs : dict
        Additional keyword arguments passed to the monet_stats metric functions
        (e.g., `obs_var`, `mod_var`, or group-by parameters).

    Returns
    -------
    dict
        A dictionary mapping the computed metric names to their results.

    Raises
    ------
    ImportError
        If monet_stats is not installed.
    TypeError
        If a metric function fails due to an invalid signature or invalid keyword arguments.

    Examples
    --------
    >>> stats = compute_statistics(
    ...     "my_stats", ["rmse", "bias"], paired_ds, {"obs_var": "obs", "mod_var": "mod"}
    ... )
    """
    logger.info("Computing statistics '%s' for metrics: %s", name, metrics)

    try:
        import monet_stats

        results = {}
        for metric in metrics:
            logger.debug("Computing metric: %s", metric)

            # Ensure the metric exists in the monet_stats module (it might be top-level or in .stats)
            found_metric = None
            for attr in dir(monet_stats):
                if attr.lower() == metric.lower():
                    found_metric = getattr(monet_stats, attr)
                    break

            if found_metric is None and hasattr(monet_stats, "stats"):
                for attr in dir(monet_stats.stats):
                    if attr.lower() == metric.lower():
                        found_metric = getattr(monet_stats.stats, attr)
                        break

            if found_metric and callable(found_metric):
                try:
                    # Attempt to handle both (data, **kwargs) and (obs, mod, **kwargs) styles
                    # Many monet-stats functions expect (obs, mod) arrays.
                    if isinstance(input_data, (xr.Dataset, pd.DataFrame)):
                        obs_var = kwargs.get("obs_var", "obs")
                        mod_var = kwargs.get("mod_var", "mod")
                        obs = input_data[obs_var]
                        mod = input_data[mod_var]
                        # Remove obs_var and mod_var from kwargs for the call
                        call_kwargs = {k: v for k, v in kwargs.items() if k not in ["obs_var", "mod_var"]}
                        result = found_metric(obs, mod, **call_kwargs)
                    else:
                        result = found_metric(input_data, **kwargs)

                    # Update history for provenance if the result is an xarray object
                    msg = f"Computed {metric} with metrics: {metrics} and params: {kwargs}"
                    if hasattr(result, "attrs"):  # Xarray
                        history = result.attrs.get("history", "")
                        result.attrs["history"] = f"{history}\n{msg}".strip()

                    results[metric] = result
                except TypeError as e:
                    logger.error("Failed to compute %s: %s", metric, e)
                    raise
            else:
                logger.warning("Metric '%s' not found in monet_stats. Skipping.", metric)

        logger.info("Successfully computed statistics '%s'", name)
        return results

    except ImportError:
        logger.error("monet-stats is not installed.")
        raise
    except Exception as e:
        logger.error("Failed to compute statistics '%s': %s", name, e)
        raise
