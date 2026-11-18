"""Direct repro of the gcafs local-file icechunk load to expose the full error chain."""

import glob
import traceback

import monetio

files = sorted(glob.glob("/gpfs/f6/bil-fire3/scratch/Barry.Baker/tmp/gefs*/12/chem/pgrb2ap25/*.grib2"))
print(f"{len(files)} files", flush=True)
try:
    ds = monetio.load(
        "gefs",
        files=files,
        filters={"typeOfFirstFixedSurface": 10, "shortName": "totAOD550"},
        sel={"lead_time": [0, 3, 6, 9, 12, 15, 18, 21]},
        use_icechunk=True,
        use_virtualizarr=True,
        icechunk_url="./zarr_stores/gcafs_debug_icechunk",
    )
    print("OK:", ds)
except Exception:
    traceback.print_exc()
    exc = traceback.format_exc()