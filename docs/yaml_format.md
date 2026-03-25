# MDT YAML Configuration Format

MDT uses a structured YAML format to define verification workflows. Each section of the configuration defines a set of tasks that MDT will execute in order of their dependencies.

## Top-Level Sections

| Section | Description |
| :--- | :--- |
| `data` | Defines model and observation datasets to load. |
| `pairing` | Specifies how datasets are matched in time and space. |
| `statistics` | Lists the verification metrics to compute. |
| `plots` | Configures static or interactive visualizations. |
| `execution` | Sets up the compute environment (e.g., local or HPC). |

---

## 1. `data` Section

The `data` section defines the sources of information for your verification.

*   **`type`**: The `monetio` dataset name (e.g., `cmaq`, `wrfchem`, `aeronet`, `airnow`).
*   **`kwargs`**: A dictionary of arguments passed to the `monetio` reader's open function.

```yaml
data:
  cmaq_output:
    type: "cmaq"
    kwargs:
      fname: "/path/to/cmaq_file.nc"
```

---

## 2. `pairing` Section

Pairing tasks align two datasets together.

*   **`source`**: The name of the source dataset (usually a model).
*   **`target`**: The name of the target dataset (usually observations).
*   **`method`**: The pairing algorithm (`interpolate` or `regrid`).
*   **`kwargs`**: Additional arguments for the pairing function.

```yaml
pairing:
  cmaq_airnow:
    source: "cmaq_output"
    target: "airnow_obs"
    method: "interpolate"
```

---

## 3. `statistics` Section

Compute verification metrics on paired data.

*   **`input`**: The name of a pairing or data task.
*   **`metrics`**: A list of metric names to compute (e.g., `rmse`, `bias`, `corr`).
*   **`kwargs`**: Arguments passed to the metric functions (e.g., `obs_var`, `mod_var`).

```yaml
statistics:
  airnow_stats:
    input: "cmaq_airnow"
    metrics: ["rmse", "bias", "corr"]
    kwargs:
      obs_var: "OZONE"
      mod_var: "OZONE"
```

---

## 4. `plots` Section

Generate static or interactive visualizations.

*   **`input`**: The name of a pairing, statistics, or data task.
*   **`type`**: The type of plot (e.g., `spatial`, `scatter`, `timeseries`).
*   **`track`**: `A` for Static (Matplotlib) or `B` for Interactive (HvPlot).
*   **`kwargs`**: Arguments passed to the plotting function.

```yaml
plots:
  spatial_cmaq:
    input: "cmaq_airnow"
    type: "spatial"
    track: "A"
    kwargs:
      savename: "cmaq_map.png"
```

---

## 5. `execution` Section

Configure how and where tasks are executed.

*   **`default_cluster`**: The name of the cluster to use for tasks (defaults to `compute`).
*   **`clusters`**: A dictionary of cluster definitions.

```yaml
execution:
  default_cluster: "local_cpu"
  clusters:
    local_cpu:
      mode: "local"
      n_workers: 4
```

---

## Advanced Example: Multiple Models vs. Observations

This example shows how to compare two different models against a single set of observations.

```yaml
data:
  # Load two different model outputs
  cmaq_model:
    type: "cmaq"
    kwargs:
      fname: "cmaq_data.nc"
  wrfchem_model:
    type: "wrfchem"
    kwargs:
      fname: "wrfchem_data.nc"

  # Load observational data
  airnow_obs:
    type: "airnow"
    kwargs:
      fname: "airnow_data.nc"

pairing:
  # Pair each model to the same observations
  pair_cmaq:
    source: "cmaq_model"
    target: "airnow_obs"
    method: "interpolate"
  pair_wrfchem:
    source: "wrfchem_model"
    target: "airnow_obs"
    method: "interpolate"

statistics:
  # Compute stats for each model
  cmaq_stats:
    input: "pair_cmaq"
    metrics: ["rmse", "bias"]
    kwargs:
      obs_var: "OZONE"
      mod_var: "OZONE"
  wrfchem_stats:
    input: "pair_wrfchem"
    metrics: ["rmse", "bias"]
    kwargs:
      obs_var: "OZONE"
      mod_var: "OZONE"

plots:
  # Create comparative spatial plots
  plot_cmaq:
    input: "pair_cmaq"
    type: "spatial"
    track: "A"
    kwargs:
      savename: "cmaq_spatial.png"
  plot_wrfchem:
    input: "pair_wrfchem"
    type: "spatial"
    track: "A"
    kwargs:
      savename: "wrfchem_spatial.png"
```
