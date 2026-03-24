import logging

logger = logging.getLogger(__name__)


def compute_statistics(name, metrics, input_data, kwargs):
    """
    Computes statistics on paired data using monet-stats.

    Parameters
    ----------
    name : str
        The identifier for this statistics task.
    metrics : list of str
        A list of metric names to compute (e.g., ['rmse', 'bias', 'corr']).
    input_data : pandas.DataFrame or xarray.Dataset
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
    """
    logger.info(f"Computing statistics '{name}' for metrics: {metrics}")

    try:
        import monet_stats.stats as stats

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
