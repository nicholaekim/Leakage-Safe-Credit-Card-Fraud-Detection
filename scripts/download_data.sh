#!/usr/bin/env bash
# download the ulb credit card fraud dataset into data/raw/
# needs the kaggle cli and an api token at ~/.kaggle/kaggle.json
# (kaggle -> account -> create new api token). see readme
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
