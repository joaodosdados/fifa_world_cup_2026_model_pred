#!/usr/bin/env python3
"""Download external datasets used to expand model training data.

Fetches the full international football results dataset (martj42 /
international_results on GitHub), which includes World Cup, World Cup
qualifiers, continental tournaments and friendlies — far more rows than
the original WorldCupMatches.csv (Copa do Mundo only).

Usage:
    python scripts/fetch_external_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"

SOURCES = {
    "international_results.csv": f"{BASE_URL}/results.csv",
    "goalscorers.csv": f"{BASE_URL}/goalscorers.csv",
    "shootouts.csv": f"{BASE_URL}/shootouts.csv",
    "former_names.csv": f"{BASE_URL}/former_names.csv",
}


def download(url: str, destination: Path) -> None:
    print(f"Baixando {url} -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    size_kb = destination.stat().st_size / 1024
    print(f"  ok ({size_kb:.0f} KB)")


def main() -> int:
    for filename, url in SOURCES.items():
        destination = RAW_DIR / filename
        try:
            download(url, destination)
        except Exception as exc:  # noqa: BLE001
            print(f"  falhou: {exc}", file=sys.stderr)
            return 1
    print("\nDataset(s) atualizados em data/raw/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
