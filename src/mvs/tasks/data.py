import importlib
import logging

logger = logging.getLogger(__name__)


def load_data(name, dataset_type, kwargs):
    """
    Dynamically loads data using monetio readers.

    Parameters
    ----------
    name : str
        The configuration identifier for this data.
    dataset_type : str
        The name of the monetio dataset (e.g., 'cmaq', 'aeronet', 'gcafs').
    kwargs : dict
        Additional keyword arguments to pass directly to the monetio dataset
        reader's open function.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame
        The loaded dataset object returned by monetio.

    Raises
    ------
    AttributeError
        If no standard open function (`open_dataset`, `open`, `open_mfdataset`)
        is found in the dynamically loaded module.
    """
    logger.info(f"Loading data '{name}' using monetio.readers.{dataset_type}")

    try:
        # Dynamically import the specific monetio reader
        module_path = f"monetio.datasets.{dataset_type}"  # Monetio changed from readers to datasets
        try:
            reader_module = importlib.import_module(module_path)
        except ImportError:
            # Try readers if datasets fails
            module_path = f"monetio.readers.{dataset_type}"
            reader_module = importlib.import_module(module_path)

        # Standard approach in monetio is usually an `open_dataset` or `open` function
        # Let's check for standard functions
        if hasattr(reader_module, "open_dataset"):
            func = reader_module.open_dataset
        elif hasattr(reader_module, "open"):
            func = reader_module.open
        elif hasattr(reader_module, "open_mfdataset"):
            func = reader_module.open_mfdataset
        else:
            # Maybe the user specified a specific function in kwargs, or it's a class
            raise AttributeError(f"Could not find a standard open function in {module_path}")

        dataset = func(**kwargs)
        logger.info(f"Successfully loaded data '{name}'")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load data '{name}': {e}")
        raise
