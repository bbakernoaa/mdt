# Model Development Tool (MDT)

Welcome to the documentation for the **Model Development Tool (MDT)**. MDT is a specialized verification system designed for Earth science model evaluation, leveraging the power of the **Pangeo** ecosystem and the **MONET** (Model Observation Evaluation Toolkit) suite of tools.

## What is MDT?

MDT is an orchestration layer built on **Prefect**, allowing you to define complex model-to-observation verification workflows using simple YAML configuration files. It automates the process of loading diverse datasets, pairing models with observations, computing performance statistics, and generating publication-quality visualizations.

### Core Philosophy

MDT is designed around several key principles:

*   **Backend-Agnostic Processing:** We prioritize `xarray.DataArray` and `xarray.Dataset` objects to ensure compatibility with both Eager (NumPy) and Lazy (Dask) computation backends.
*   **Aero Protocol:** MDT enforces "Double-Check" verification to ensure that lazy processing results match eager computations, preventing hidden compute overhead.
*   **Two-Track Visualization:**
    *   **Track A (Publication):** Static, high-fidelity plots using `matplotlib` and `cartopy`.
    *   **Track B (Exploration):** Interactive, scalable visualizations using `hvplot` and `geoviews`.
*   **HPC Compatibility:** Built-in support for Dask-Jobqueue allows MDT to scale seamlessly from a local workstation to large HPC clusters (e.g., SLURM, PBS).

## The MONET Ecosystem

MDT acts as the "glue" for several foundational libraries:

*   **monetio:** Handles the complex IO tasks of reading various atmospheric model and observation formats.
*   **monet:** Provides core utilities for spatial and temporal pairing of model data to observation points or grids.
*   **monet-stats:** A robust library for computing standardized atmospheric verification metrics.
*   **monet-plots:** Specialized plotting routines tailored for atmospheric science.
