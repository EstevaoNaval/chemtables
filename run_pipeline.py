#!/usr/bin/env python3
"""Dev convenience entry point; equivalent to `python -m chemtables`.

Requires an editable install of this package (`pip install -e .`) so that
`chemtables` is importable. See README.md for setup.
"""

from chemtables.cli import main

if __name__ == "__main__":
    main()
