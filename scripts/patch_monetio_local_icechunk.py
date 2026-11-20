"""Patch monetio's icechunk writer to support LOCAL-file virtual references.

Problem: ``_open_via_icechunk`` in monetio/readers/drivers.py only collects
``s3://`` and ``http(s)://`` URL prefixes when registering icechunk Virtual
Chunk Containers. Local GRIB files produce plain filesystem paths in the
virtual manifest, so no container is registered and virtualizarr raises
"No Virtual Chunk Containers set" when writing the store.

Fix: register a local-filesystem container covering every non-URL manifest
path (one container per top-level directory), so local-file icechunk
references can be written and later read on any node that mounts the files.

Idempotent: re-running detects the marker and does nothing.
Usage: python scripts/patch_monetio_local_icechunk.py [path/to/monetio/readers/drivers.py]
"""

import sys
from pathlib import Path

MARKER = "# MDT-PATCH: local-file virtual chunk containers"

OLD = """                elif path.startswith("http://") or path.startswith("https://"):
                    parts = path.split("/", 4)
                    prefix = "/".join(parts[:3]) + "/"
                    unique_prefixes.add(prefix)

    config = icechunk.RepositoryConfig.default()"""

NEW = """                elif path.startswith("http://") or path.startswith("https://"):
                    parts = path.split("/", 4)
                    prefix = "/".join(parts[:3]) + "/"
                    unique_prefixes.add(prefix)
                elif path.startswith("file://"):
                    # MDT-PATCH: local-file virtual chunk containers
                    # Local GRIB manifests use file:///abs/path URLs; register a
                    # local-filesystem container rooted at the top-level dir.
                    fs_path = path[len("file://"):]
                    prefix = "file:///" + fs_path.lstrip("/").split("/", 1)[0] + "/"
                    local_prefixes.add(prefix)
                else:
                    # MDT-PATCH: local-file virtual chunk containers
                    import os as _os

                    if _os.path.isabs(path):
                        prefix = "file:///" + path.lstrip("/").split("/", 1)[0] + "/"
                        local_prefixes.add(prefix)

    config = icechunk.RepositoryConfig.default()
    for prefix in sorted(local_prefixes):
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                url_prefix=prefix,
                store=icechunk.local_filesystem_store(prefix[len("file://"):]),
            )
        )"""

OLD_INIT = """    unique_prefixes = set()
    for var in vds.variables.values():"""

NEW_INIT = """    unique_prefixes = set()
    local_prefixes = set()  # MDT-PATCH: local-file virtual chunk containers
    for var in vds.variables.values():"""

OLD_AUTH = """    authorize = {prefix: None for prefix in unique_prefixes}"""
NEW_AUTH = """    authorize = {prefix: None for prefix in unique_prefixes | local_prefixes}"""


def apply_patches(drivers_path: Path) -> None:
    backup = drivers_path.with_suffix(".py.bak-local-icechunk")
    if not backup.exists():
        backup.write_text(drivers_path.read_text())
    # Always patch from the pristine backup so re-runs pick up script updates.
    text = backup.read_text()
    for old, new, label in (
        (OLD_INIT, NEW_INIT, "local_prefixes init"),
        (OLD, NEW, "local prefix collection"),
        (OLD_AUTH, NEW_AUTH, "authorize local prefixes"),
    ):
        if old not in text:
            raise SystemExit(f"ERROR: expected block for '{label}' not found in {drivers_path}; monetio version may differ.")
        text = text.replace(old, new, 1)
    drivers_path.write_text(text)
    print(f"Patched {drivers_path} (pristine backup: {backup})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        import monetio

        target = Path(monetio.__file__).parent / "readers" / "drivers.py"
    apply_patches(target)
