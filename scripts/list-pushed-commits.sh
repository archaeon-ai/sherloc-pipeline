#!/usr/bin/env bash
# List every commit a push to main brings, one id per line, for
# scripts/check-public-wording.py --commits-file.
#
# Usage: scripts/list-pushed-commits.sh BEFORE AFTER OUT_FILE
#   a normal push     BEFORE..AFTER;
#   a new branch      (BEFORE is all zeros) the commits of AFTER on no other
#                     branch of origin;
#   a forced push     merge-base(BEFORE, AFTER)..AFTER, after fetching the old
#                     tip when it is not in the clone.
# Exits 1 when the old tip cannot be fetched: those messages were not read.
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "usage: $0 BEFORE AFTER OUT_FILE" >&2
  exit 2
fi
before=$1
after=$2
out=$3

if [ "$before" = "0000000000000000000000000000000000000000" ]; then
  # With --remotes, --exclude matches the name below refs/remotes/. origin/HEAD,
  # when the clone has it, names main too.
  git rev-list "$after" --not --exclude=origin/main --exclude=origin/HEAD \
    --remotes=origin > "$out"
  exit 0
fi
if ! git cat-file -e "$before^{commit}" 2>/dev/null; then
  if ! git fetch --no-tags --quiet origin "$before" 2>/dev/null \
      || ! git cat-file -e "$before^{commit}" 2>/dev/null; then
    echo "::error::old tip $before could not be fetched; check the pushed commit messages by hand" >&2
    exit 1
  fi
fi
git rev-list "$(git merge-base "$before" "$after")..$after" > "$out"
