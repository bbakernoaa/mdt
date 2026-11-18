"""One-shot inspection of a running MDT Dask scheduler (diagnostic helper)."""

import sys

from distributed import Client

addr = sys.argv[1]
c = Client(addr, timeout="10s")
info = c.scheduler_info()
for w in info["workers"].values():
    print(
        "worker",
        w["name"],
        "host",
        w["host"],
        "resources",
        w.get("resources"),
        "executing",
        w.get("executing"),
        "in_memory",
        w.get("in_memory"),
    )
print("processing:", c.processing())
print("nthreads:", c.nthreads())
c.close()