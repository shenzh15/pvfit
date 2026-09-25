"""Support ``python -m pvfit`` in addition to the installed command."""

from .cli import main

raise SystemExit(main())
