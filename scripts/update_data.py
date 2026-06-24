"""Update live FIFA schedule/results data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.auto_updater import run_auto_update


def main() -> None:
    """Run the FIFA updater and print a compact JSON summary."""
    result = run_auto_update()
    serializable = {
        key: str(value) if key == "last_update" else value
        for key, value in result.items()
    }
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
