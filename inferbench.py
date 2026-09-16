#!/usr/bin/env python3
"""Single public entry point for inference performance and accuracy work."""

import sys

from inference_suite.cli import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; managed searches clean up their owned container.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
