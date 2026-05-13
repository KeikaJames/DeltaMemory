#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

messages_dir="tests/.hook-test-messages"
mkdir -p "$messages_dir"
trap 'rm -rf "$messages_dir"' EXIT

bad_message="$messages_dir/bad-message.txt"
clean_message="$messages_dir/clean-message.txt"

cat > "$bad_message" <<'MSG'
test commit

Co-authored-by: Copilot <fake@x>
MSG

cat > "$clean_message" <<'MSG'
test commit

Plain body without forbidden trailers.
MSG

if ( source .githooks/commit-msg "$bad_message" ); then
  echo "BUG: commit-msg hook accepted a Copilot co-author trailer" >&2
  exit 1
fi

if ! ( source .githooks/commit-msg "$clean_message" ); then
  echo "BUG: commit-msg hook rejected a clean commit message" >&2
  exit 1
fi

echo "git hook tests passed"
