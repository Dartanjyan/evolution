#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
archive="$repo_root/archive.zip"
temporary_archive="$(mktemp "${TMPDIR:-/tmp}/evolution.XXXXXX.zip")"
rm "$temporary_archive"
trap 'rm -f "$temporary_archive"' EXIT

cd "$repo_root"
git ls-files --cached --others --exclude-standard -z |
  xargs -0 -r zip -q "$temporary_archive" --
mv "$temporary_archive" "$archive"