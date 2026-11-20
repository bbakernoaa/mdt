# HPC Usage Guide

MDT is designed to scale from local workstations to high-performance computing (HPC) clusters. It integrates with `dask-jobqueue` to automatically manage job submission and worker lifecycle on various NOAA platforms.

## Execution Configuration

The `execution` section of your YAML file controls where and how your tasks run.

```yaml
execution:
  default_cluster: "hera_batch"
  clusters:
    hera_batch:
      mode: "hera"
      account: "my_account"
      walltime: "02:00:00"
```

### Supported NOAA Platforms

MDT provides built-in profiles for major NOAA RDHPCS and production platforms. Specifying one of these in the `mode` field will automatically apply optimized defaults for that system.

| Platform | `mode` | Scheduler | Default Cores | Default Memory |
| :--- | :--- | :--- | :--- | :--- |
| **Hera** | `hera` | SLURM | 40 | 120GB |
| **Jet** | `jet` | SLURM | 24 | 60GB |
| **Orion** | `orion` | SLURM | 40 | 180GB |
| **Hercules** | `hercules` | SLURM | 80 | 250GB |
| **Gaea** | `gaea` | SLURM | 36 | 120GB |
| **Ursa** | `ursa` | SLURM | 36 | 120GB |
| **WCOSS2** | `wcoss2` | PBS | 128 | 256GB |

### Generic Schedulers

If you are on a platform not listed above, you can use generic scheduler modes:
*   `slurm`
*   `pbs`
*   `lsf`

## Customizing Cluster Parameters

You can override any platform default by providing additional keys in the cluster configuration. These are passed directly to the underlying `dask-jobqueue` cluster class (e.g., `SLURMCluster` or `PBSCluster`).

```yaml
execution:
  clusters:
    custom_jet:
      mode: "jet"
      cores: 12
      memory: "30GB"
      job_extra_directives: ["--qos=windfall"]
```

## Assigning Tasks to Specific Clusters

MDT allows you to run different tasks on different clusters within the same workflow. For example, you might want to load data on a local cluster but perform heavy pairing and statistics on an HPC cluster.

```yaml
data:
  obs_data:
    type: "airnow"
    cluster: "local"  # Run locally
    kwargs:
      fname: "obs.nc"

pairing:
  heavy_pairing:
    source: "model_data"
    target: "obs_data"
    cluster: "hera_batch"  # Run on HPC
    method: "regrid"
```

If no `cluster` is specified for a task, it uses the `default_cluster` defined in the `execution` section.

## Gaea C6 Data Transfer Nodes (DTNs)

See `docs/examples/gefs_aeronet_gaea_dtn.yaml` for a complete working example. Key platform specifics:

- DTNs are reached through the SLURM federation cluster `es`, partition `dtn_f5_f6`, QoS `dtn`. Use `job_extra_directives: ["-M es", "--qos=dtn", "--constraint=f6"]` to land on the DTNs with `/gpfs/f6` mounted.
- The `dtn` QoS allows at most **4 CPUs per job** (`MaxTRESPerJob`) and 50 CPUs per user (`MaxTRESPU`). Larger jobs pend forever with `QOSMaxCpuPerJobLimit` and deadlock the workflow. Scale by submitting more small worker jobs (`workers: N` with `cores: 4`), not bigger ones.
- Set `execution.dask_interface: "hsn0"` so the scheduler advertises the Slingshot HSN address. The default front-end network (`fen`) does not carry arbitrary TCP ports between login nodes and DTNs, so workers time out connecting to the scheduler without this.

## Two-Stage Workflow: Prep + Fixed Allocation

For NOAA-style fixed compute allocations (a set block of nodes/CPUs, often network-isolated), MDT supports a two-stage path. Complete example: `docs/examples/gefs_aeronet_gaea_prep.yaml`.

### Stage 1 — Prepare data (`mdt prep`)

Run where external data access exists (a DTN or service node), either interactively:

```bash
mdt prep docs/examples/gefs_aeronet_gaea_prep.yaml
```

or as a batch job on a DTN (sized from `execution.prep_allocation`; the Gaea
example defaults to the dtn QoS 4-CPU/job cap):

```bash
mdt script docs/examples/gefs_aeronet_gaea_prep.yaml --stage prep -o run_prep.slurm
sbatch run_prep.slurm
```

- Executes only the data-retrieval/preparation stages; pairing, statistics and plots never run here.
- **Storage policy**: remote sources (S3/web) are downloaded and materialized as real local zarr data — the compute stage never touches the remote source. Local sources (files already on shared storage) use icechunk *virtual references* (byte-range pointers, no copy) when a `zarr_store` with `backend: icechunk` is configured.
- Persists every dataset with a completeness manifest (`<store>.mdt-prep.json`).
- Local-file icechunk references require the monetio fix from branch `fix/local-file-icechunk-containers` (bbakernoaa/monetio, PR pending); until it lands upstream, `scripts/patch_monetio_local_icechunk.py` applies the same change to an installed monetio.
- Idempotent: complete stores covering the requested dates are reused (`cached`); `--force` re-retrieves everything.
- Interrupt-safe: an interrupted store stays `incomplete` and is re-prepared on the next run, never served as a complete cache.
- Prep runs on local Dask workers at the launch point regardless of the configured compute clusters. On Gaea DTNs, remember the `dtn` QoS 4-CPU-per-job cap when launching prep itself as a batch job.

### Stage 2 — Compute inside a fixed allocation

Generate and submit the allocation script:

```bash
mdt script docs/examples/gefs_aeronet_gaea_prep.yaml --stage compute -o run_analysis.slurm
sbatch run_analysis.slurm
```

Sizing: the Gaea example requests one full C6 node — `ntasks_per_node: 48`,
`cpus_per_task: 4` → 46 Dask worker processes (tasks 0/1 are scheduler/client).
Prefer many small workers over few large ones (one GIL per process). Scale out
with `nodes: N`; only constraint is total tasks ≥ 3.

The script requests the resources in `execution.allocation` and runs `srun -n <ntasks> python -m mdt.allocation <config>`. Inside the allocation, dask-jobqueue's **SLURMRunner** bootstraps the whole workload: task 0 = Dask scheduler, task 1 = client (executes the DAG), remaining tasks = workers — coordinated through a `scheduler-<jobid>.json` file on shared storage. **Zero additional batch jobs are submitted**, and all loads are served from the prepared stores; a missing/incomplete store fails fast with a message naming the dataset and pointing back to `mdt prep`.

Interactive smoke test on Gaea C6:

```bash
salloc -M c6 -p batch -A bil-fire3 -N 1 --ntasks=4 --cpus-per-task=8 -t 00:30:00
srun -n 4 /gpfs/f6/bil-fire3/world-shared/python/envs/mdt/bin/python \
     -m mdt.allocation docs/examples/gefs_aeronet_gaea_prep.yaml
```

Notes:
- `nodes * ntasks_per_node` must be >= 3 (scheduler + client + at least one worker).
- Keep `dask_interface: hsn0` on Gaea; the scheduler now runs *inside* the allocation, so login-node reachability is no longer involved.
- You can also use your own allocation script — `mdt.allocation` only needs the SLURM environment and the config path.
- The elastic mode (`SLURMCluster` worker jobs, previous section) remains available unchanged for sites that allow per-worker submissions.

## Prefect State on Network File Systems

Prefect's ephemeral server keeps a SQLite database in `PREFECT_HOME` (default `~/.prefect`). On NFS home directories, SQLite WAL locking is unreliable and the database can corrupt (symptoms: `sqlite3.DatabaseError: file is not a database`, repeated `prefect.server.events.triggers` errors, or "Timed out while attempting to connect to ephemeral Prefect API server"). Keep Prefect state on node-local disk:

```bash
export PREFECT_HOME=/tmp/prefect-mdt-$USER
mdt run workflow.yaml
```

If the default database is already corrupted, move `~/.prefect/prefect.db*` aside; Prefect recreates it on the next run.
