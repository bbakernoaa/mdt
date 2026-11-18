"""Probe: per-worker resource specs injected into LocalCluster.worker_spec."""

import time

from dask.distributed import Client, LocalCluster

c = LocalCluster(n_workers=0, host="127.0.0.1")
base = dict(c.new_spec)
print("base spec:", base)
specs = {}
for i, res in enumerate(["COMPUTE", "SERVICE"]):
    s = dict(base)
    s["options"] = {**base.get("options", {}), "nthreads": 1, "resources": {res: 1}}
    specs[f"{res.lower()}-{i}"] = s
c.worker_spec.update(specs)
c.scale(len(specs))
time.sleep(8)
cl = Client(c)
print("workers:", {x["name"]: x.get("resources") for x in cl.scheduler_info()["workers"].values()})
cl.close()
c.close()