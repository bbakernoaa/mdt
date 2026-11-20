"""Allow ``python -m mdt`` as an alias for the ``mdt`` console entry point."""

from mdt.cli import main

if __name__ == "__main__":
    main()