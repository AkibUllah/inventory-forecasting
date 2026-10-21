"""Write the FastAPI app's OpenAPI document to a file (default: openapi.json).

Used by the GitHub Pages workflow to publish a static API reference. Importing
``invforecast.api`` needs no model bundle or environment: the bundle is only
loaded on the first request.

    python scripts/export_openapi.py [OUTPUT_PATH]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from invforecast.api import app


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("openapi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(app.openapi(), indent=2) + "\n")
    print(f"wrote {out} ({len(app.openapi()['paths'])} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
