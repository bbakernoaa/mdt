import logging
from typing import Dict, List, Union

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def compute_statistics(
    name: str,
    metrics: List[str],
    input_data: Union[xr.Dataset, xr.DataArray, pd.DataFrame],
    kwargs: dict,
) -> Dict[str, Union[xr.Dataset, xr.DataArray, pd.Series]]:
    """
    Computes statistics on paired data using monet-stats.

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
    >>> stats = compute_statistics("my_stats", ["rmse", "bias"], paired_ds, {"obs_var": "obs", "mod_var": "mod"})
    """
    logger.info(f"Computing statistics '{name}' for metrics: {metrics}")

    try:
        import monet.util.stats as stats

        results = {}
        for metric in metrics:
            logger.debug(f"Computing metric: {metric}")

            # Ensure the metric exists in the monet_stats module
            if hasattr(stats, metric):
                metric_func = getattr(stats, metric)

                # Assume kwargs contains standard monet-stats args: df, model_var, obs_var
                # If input_data is a DataFrame with 'model' and 'obs', it fits perfectly
                try:
                    result = metric_func(input_data, **kwargs)

                    # Update history for provenance if the result is an xarray object
                    if hasattr(result, "attrs"):
                        history = result.attrs.get("history", "")
                        new_history = f"Computed {metric} with metrics: {metrics} and params: {kwargs}"
                        result.attrs["history"] = f"{history}\n{new_history}".strip()

                    results[metric] = result
                except TypeError as e:
                    logger.error(f"Failed to compute {metric}: {e}")
                    raise
            else:
                logger.warning(f"Metric '{metric}' not found in monet_stats.stats. Skipping.")

        logger.info(f"Successfully computed statistics '{name}'")
        return results

    except ImportError:
        logger.error("monet-stats is not installed.")
        raise
    except Exception as e:
        logger.error(f"Failed to compute statistics '{name}': {e}")
        raise
