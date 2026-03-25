# User Guide

This guide describes the core architecture and usage principles of MDT.

## Core Concepts

MDT is designed to handle the complexity of atmospheric model verification workflows. It uses a **Directed Acyclic Graph (DAG)** to model dependencies between tasks.

### Tasks and Workflows

MDT breaks down verification into several key task types:

1.  **Data Loading:** Using `monetio` to read model output (e.g., CMAQ, WRF-Chem, GEOS-Chem) and observational data (e.g., AERONET, IMPROVE, EPA AirNow).
2.  **Pairing:** Aligning model and observation data in time and space. This may involve:
    *   **Interpolation:** Extracting model values at specific observation point locations.
    *   **Regridding:** Mapping model output to a standard reference grid using `xregrid`.
3.  **Statistics:** Computing standard verification metrics (RMSE, Bias, Correlation, etc.) using `monet-stats`.
4.  **Plotting:** Visualizing the results using the **Two-Track Rule**.

### DAG Construction

When you run MDT with a configuration file, it automatically builds a dependency graph. For example, a `statistics` task that depends on a `pairing` task will wait for that pairing task to complete before starting. This allows MDT to optimize the execution order and run independent tasks in parallel.

## The Two-Track Visualization Rule

MDT handles visualization using two distinct tracks, allowing you to choose the best tool for your needs:

### Track A: Static (Publication-Quality)
*   **Engine:** Matplotlib + Cartopy.
*   **Use Case:** Final figures for papers, reports, or static dashboards.
*   **Key Feature:** High degree of control over map projections, transforms, and aesthetic details.

### Track B: Interactive (Exploratory)
*   **Engine:** HvPlot + GeoViews.
*   **Use Case:** Data exploration, zooming into specific regions, and interacting with large datasets.
*   **Key Feature:** Fast, interactive rendering of large grids using Datashader (mandatory `rasterize=True`).

## Scalability and HPC Support

MDT is built on **Prefect**, which provides a robust engine for task scheduling and monitoring. For large-scale verification tasks, MDT supports multiple execution clusters:

*   **Local:** Use local CPU cores for processing.
*   **Dask (SLURM/PBS/LSF):** Scale out to multiple nodes on an HPC cluster.

MDT's backend-agnostic design ensures that it can efficiently handle "big data" using **Dask** without requiring manual code changes.
