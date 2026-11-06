#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script, regardless of the current directory.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Permanently remove generated experiment data, models, and reports.
for directory in mlflow mlruns model reports; do
    rm -rf -- "$project_root/$directory"
    printf 'Removed %s\n' "$directory"
done
