#!/usr/bin/env bash
# Download the ULB Credit Card Fraud dataset into data/raw/.
# Requires the Kaggle CLI and an API token at ~/.kaggle/kaggle.json
# (Kaggle -> Account -> Create New API Token). See README.md.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found." >&2
  echo "  pip install kaggle" >&2
  echo "  then place your token at ~/.kaggle/kaggle.json (chmod 600)" >&2
  exit 1
fi

kaggle datasets download -d mlg-ulb/creditcardfraud -p "$DEST" --unzip
echo "Done -> $DEST/creditcard.csv"
