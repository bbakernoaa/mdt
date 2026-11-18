"""Probe icechunk container API + patched monetio prefix collection."""

import icechunk

c = icechunk.RepositoryConfig.default()
c.set_virtual_chunk_container(icechunk.VirtualChunkContainer(url_prefix="/gpfs/", store=icechunk.local_filesystem_store()))
print("containers after set:", c.virtual_chunk_containers)

import monetio.readers.drivers as d
import inspect

src = inspect.getsource(d._open_via_icechunk)
print("patch present:", "MDT-PATCH" in src)
print("local_prefixes in src:", "local_prefixes" in src)