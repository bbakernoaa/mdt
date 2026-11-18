"""Inspect the manifest paths produced for one local GRIB file (monetio pipeline)."""

import glob
import os
import pathlib
import tempfile

import ujson
from grib2io.kerchunk import ReferenceGenerator
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.manifests.array import ManifestArray
from virtualizarr.parsers import KerchunkJSONParser

f = sorted(glob.glob("/gpfs/f6/bil-fire3/scratch/Barry.Baker/tmp/gefs*/12/chem/pgrb2ap25/*.grib2"))[0]
print("file:", f)
gen = ReferenceGenerator(file_paths=[f], filters={"typeOfFirstFixedSurface": 10, "shortName": "totAOD550"})
manifest = gen.generate()
fd, tmp = tempfile.mkstemp(suffix=".json")
os.close(fd)
with open(tmp, "w") as fh:
    ujson.dump(manifest, fh)

from monetio.readers.drivers import _select_store

registry, _ = _select_store([f], {})
registry.register("file:///", LocalStore(prefix="/"))
vds = open_virtual_dataset(url=pathlib.Path(tmp).resolve().as_uri(), registry=registry, parser=KerchunkJSONParser(), loadable_variables=[])
paths = set()
for var in vds.variables.values():
    if isinstance(var.data, ManifestArray):
        paths.update(var.data.manifest.iter_nonempty_paths())
print("manifest paths:", sorted(paths)[:3])
os.unlink(tmp)