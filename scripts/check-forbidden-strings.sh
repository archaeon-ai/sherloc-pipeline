#!/usr/bin/env bash
# Block operator-local infrastructure strings from entering tracked content.
#
# Usage:
#   scripts/check-forbidden-strings.sh             # default: --staged
#   scripts/check-forbidden-strings.sh --staged    # check git-staged files (pre-commit hook)
#   scripts/check-forbidden-strings.sh --tree      # check every tracked file (CI / audit)
#
# Exits 0 if clean, 1 on hit. See CONTRIBUTING.md "Public-repo discipline".
#
# Pattern literals are written as adjacent shell-concatenated string parts
# (e.g. '/data''/sherloc/') so that history-rewrite tools applying literal
# byte substitutions (such as `git filter-repo --replace-text`) cannot
# silently rewrite the patterns inside this script. The runtime values
# match the originals — bash concatenates adjacent quoted strings.

set -euo pipefail

MODE="${1:---staged}"

# Patterns that must not appear in tracked content. Keep in lockstep with
# the substitution table in CONTRIBUTING.md.
PATTERNS=(
  '/data''/sherloc/'
  '/home''/kenwilliford'
  '/nas''/000_sherloc'
  'evom''1ni'
  'kenwill''iford\.net'
  'Magenta''Anchor'
  'Jade''Bay'
  'Silent''Raven'
  'Frosty''Beaver'
  'Silent''Castle'
  'Copper''Brook'
  'Purple''Compass'
)

# Files allowed to mention these patterns:
# - the script itself (where the patterns are defined)
# - CONTRIBUTING.md (documents the substitution table)
EXCLUDE_PATTERNS=(
  '^scripts/check-forbidden-strings\.sh$'
  '^CONTRIBUTING\.md$'
)

case "$MODE" in
  --staged)
    files=$(git diff --cached --name-only --diff-filter=ACM)
    ;;
  --tree)
    files=$(git ls-files)
    ;;
  *)
    echo "Usage: $0 [--staged|--tree]" >&2
    exit 2
    ;;
esac

if [ -z "$files" ]; then
  exit 0
fi

filtered=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  skip=0
  for ex in "${EXCLUDE_PATTERNS[@]}"; do
    if printf '%s' "$f" | grep -qE "$ex"; then
      skip=1
      break
    fi
  done
  [ "$skip" = 0 ] && filtered+="$f"$'\n'
done <<< "$files"

if [ -z "$filtered" ]; then
  exit 0
fi

hit=0
for pattern in "${PATTERNS[@]}"; do
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    [ -f "$f" ] || continue
    if grep -nE -- "$pattern" "$f" > /dev/null 2>&1; then
      hit=1
      echo "FORBIDDEN STRING: pattern '$pattern' in $f"
      grep -nE -- "$pattern" "$f" | head -5 | sed 's/^/    /'
    fi
  done <<< "$filtered"
done

if [ "$hit" -ne 0 ]; then
  cat <<'EOF'

Forbidden operator-local infrastructure strings detected. Substitutions:
  Local data path     ->  use SHERLOC_DATA_ROOT env var or relative paths
  Operator home dir   ->  use $HOME or ~
  Operator NAS path   ->  use SHERLOC_NAS_ROOT env var or relative paths
  Operator hostname   ->  generic "devhost" or env-driven
  Operator domain     ->  example.com or env-driven hostnames
  Internal codenames  ->  do not appear in tracked content

See CONTRIBUTING.md "Public-repo discipline" for the substitution table
and rationale.
EOF
  exit 1
fi

exit 0
