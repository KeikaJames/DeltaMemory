#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="$REPO_ROOT/scripts/branch_pr_helper.sh"
SCRATCH="$REPO_ROOT/tests/.branch_pr_helper_tmp"

cleanup() {
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

assert_contains() {
    local haystack=$1 needle=$2
    if [[ "$haystack" != *"$needle"* ]]; then
        echo "expected output to contain: $needle" >&2
        echo "actual output:" >&2
        echo "$haystack" >&2
        exit 1
    fi
}

test_help_prints_usage() {
    local output
    output="$($HELPER --help)"
    assert_contains "$output" "Usage: scripts/branch_pr_helper.sh"
    assert_contains "$output" "--strict-no-coauthor"
}

test_strict_no_coauthor_rejects_recent_head_commit() {
    rm -rf "$SCRATCH"
    mkdir -p "$SCRATCH"
    (
        cd "$SCRATCH"
        git init -q
        git config user.name "KeikaJames"
        git config user.email "gabira@bayagud.com"
        printf 'content\n' > file.txt
        git add file.txt
        git commit -q -m $'test: forbidden trailer\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>'

        set +e
        output="$($HELPER --strict-no-coauthor 2>&1)"
        status=$?
        set -e

        if [ "$status" -eq 0 ]; then
            echo "expected --strict-no-coauthor to fail" >&2
            exit 1
        fi
        assert_contains "$output" "Co-authored-by: Copilot"
    )
}

test_help_prints_usage
test_strict_no_coauthor_rejects_recent_head_commit

echo "branch_pr_helper tests passed"
