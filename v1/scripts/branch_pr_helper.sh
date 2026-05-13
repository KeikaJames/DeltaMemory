#!/usr/bin/env bash
# Usage: scripts/branch_pr_helper.sh <branch-suffix> "<commit subject>" "<pr title>" "<pr body>"
# Example: scripts/branch_pr_helper.sh fix/ci "fix(ci): macos pytest hang" "Fix CI" "Resolves run 25376294415"
#
# Steps performed:
#   1. assert clean working tree
#   2. assert author config = KeikaJames <gabira@bayagud.com> (set if missing)
#   3. assert no "Co-authored-by: Copilot" trailers in pending commit (commit-msg hook also enforces)
#   4. git fetch origin
#   5. git checkout -b "$branch" origin/main
#   6. (caller is expected to have already staged/committed; if there are
#       staged changes pending, this script can also perform the commit
#       with the author/no-trailer guardrails)
#   7. git push -u origin "$branch"
#   8. gh pr create --base main --head "$branch" --title ... --body ...
#   9. gh pr checks --watch (block until CI green or fail)
#  10. on green: print "READY TO MERGE" + PR URL; on red: print PR URL + failed jobs

set -euo pipefail

AUTHOR_NAME="KeikaJames"
AUTHOR_EMAIL="gabira@bayagud.com"
AUTHOR_IDENT="$AUTHOR_NAME <$AUTHOR_EMAIL>"
BASE_BRANCH="main"
REMOTE="origin"
STRICT_NO_COAUTHOR=0
DRY_RUN=0

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --help                 Show this help text.
  --dry-run              Validate only; print actions without mutating git/GitHub.
  --strict-no-coauthor   Also reject if HEAD's last 5 commits contain
                         "Co-authored-by: Copilot". With no positional
                         arguments, only this check is run.
USAGE
}

die() {
    echo "error: $*" >&2
    exit 1
}

info() {
    echo "==> $*"
}

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'DRY-RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

contains_copilot_trailer() {
    grep -Fqi "Co-authored-by: Copilot"
}

assert_git_repo() {
    git rev-parse --show-toplevel >/dev/null 2>&1 || die "not inside a git repository"
}

assert_tools_and_auth() {
    command -v gh >/dev/null 2>&1 || die "missing required tool: gh"
    if [ "$DRY_RUN" -eq 0 ]; then
        gh auth status >/dev/null 2>&1 || die "gh is not authenticated; run 'gh auth login' and retry"
    fi
}

assert_worktree_committable() {
    if ! git diff --quiet --; then
        die "working tree has unstaged changes; stage or commit them before running"
    fi

    local untracked
    untracked="$(git ls-files --others --exclude-standard)"
    if [ -n "$untracked" ]; then
        die "working tree has untracked files; stage or remove them before running"
    fi
}

assert_author_config() {
    local name email
    name="$(git config --get user.name || true)"
    email="$(git config --get user.email || true)"

    if [ -z "$name" ]; then
        run git config user.name "$AUTHOR_NAME"
        name="$AUTHOR_NAME"
    fi
    if [ -z "$email" ]; then
        run git config user.email "$AUTHOR_EMAIL"
        email="$AUTHOR_EMAIL"
    fi

    [ "$name" = "$AUTHOR_NAME" ] || die "git user.name must be '$AUTHOR_NAME' (found '$name')"
    [ "$email" = "$AUTHOR_EMAIL" ] || die "git user.email must be '$AUTHOR_EMAIL' (found '$email')"
}

assert_no_recent_copilot_coauthor() {
    if git log -5 --format=%B | contains_copilot_trailer; then
        die "HEAD's last 5 commits contain a forbidden 'Co-authored-by: Copilot' trailer"
    fi
}

assert_commit_message_clean() {
    local message=$1
    if printf '%s\n' "$message" | contains_copilot_trailer; then
        die "commit message contains forbidden 'Co-authored-by: Copilot' trailer"
    fi
}

checkout_branch() {
    local branch=$1

    git check-ref-format --branch "$branch" >/dev/null 2>&1 || die "invalid branch name: $branch"
    case "$branch" in
        "$BASE_BRANCH"|"$REMOTE/$BASE_BRANCH"|main|master)
            die "refusing to operate directly on protected branch '$branch'"
            ;;
    esac

    info "Fetching $REMOTE"
    run git fetch "$REMOTE"

    local current_branch
    current_branch="$(git branch --show-current)"
    if [ "$current_branch" = "$branch" ]; then
        info "Already on $branch; skipping checkout"
        return
    fi

    if git show-ref --verify --quiet "refs/heads/$branch"; then
        info "Local branch $branch exists; checking it out"
        run git checkout "$branch"
    elif git show-ref --verify --quiet "refs/remotes/$REMOTE/$branch"; then
        info "Remote branch $REMOTE/$branch exists; creating local tracking branch"
        run git checkout -b "$branch" "$REMOTE/$branch"
    else
        info "Creating $branch from $REMOTE/$BASE_BRANCH"
        run git checkout -b "$branch" "$REMOTE/$BASE_BRANCH"
    fi
}

commit_staged_if_any() {
    local subject=$1

    if git diff --cached --quiet --; then
        info "No staged changes pending; skipping commit"
        return
    fi

    assert_commit_message_clean "$subject"
    info "Committing staged changes as $AUTHOR_IDENT"
    run git commit --author "$AUTHOR_IDENT" -m "$subject"
}

push_branch() {
    local branch=$1
    local local_sha remote_sha

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN: git push -u $REMOTE $branch"
        return
    fi

    local_sha="$(git rev-parse "$branch")"
    remote_sha="$(git rev-parse --verify --quiet "$REMOTE/$branch" || true)"
    if [ -n "$remote_sha" ] && [ "$local_sha" = "$remote_sha" ]; then
        info "Remote $REMOTE/$branch already matches local branch; skipping push"
        return
    fi

    info "Pushing $branch"
    run git push -u "$REMOTE" "$branch"
}

ensure_pr() {
    local branch=$1 title=$2 body=$3 pr_url

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN: gh pr create --base $BASE_BRANCH --head $branch --title '$title' --body '$body'"
        echo "https://github.com/DRY-RUN/PR"
        return
    fi

    pr_url="$(gh pr list --base "$BASE_BRANCH" --head "$branch" --state open --json url --jq '.[0].url // empty')"
    if [ -n "$pr_url" ]; then
        info "Open PR already exists: $pr_url"
        echo "$pr_url"
        return
    fi

    info "Creating PR"
    gh pr create --base "$BASE_BRANCH" --head "$branch" --title "$title" --body "$body"
}

watch_checks() {
    local pr_url=$1 failed

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN: gh pr checks --watch $pr_url"
        echo "READY TO MERGE"
        echo "$pr_url"
        return
    fi

    info "Watching PR checks"
    if gh pr checks "$pr_url" --watch; then
        echo "READY TO MERGE"
        echo "$pr_url"
        return
    fi

    echo "PR checks failed or did not complete successfully: $pr_url" >&2
    failed="$(gh pr checks "$pr_url" --json name,bucket,state,link --jq '.[] | select(.bucket == "fail" or .bucket == "cancel") | "- \(.name): \(.state) \(.link)"' || true)"
    if [ -n "$failed" ]; then
        echo "Failed jobs:" >&2
        echo "$failed" >&2
    else
        echo "Failed jobs could not be determined; run: gh pr checks $pr_url" >&2
    fi
    exit 1
}

main() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --help|-h)
                usage
                exit 0
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --strict-no-coauthor)
                STRICT_NO_COAUTHOR=1
                shift
                ;;
            --)
                shift
                break
                ;;
            -*)
                die "unknown option: $1"
                ;;
            *)
                break
                ;;
        esac
    done

    assert_git_repo

    if [ "$STRICT_NO_COAUTHOR" -eq 1 ]; then
        assert_no_recent_copilot_coauthor
        if [ "$#" -eq 0 ]; then
            info "No Copilot co-author trailers found in HEAD's last 5 commits"
            exit 0
        fi
    fi

    [ "$#" -eq 4 ] || { usage >&2; die "expected 4 arguments, got $#"; }

    local branch=$1 commit_subject=$2 pr_title=$3 pr_body=$4 pr_url

    assert_tools_and_auth
    assert_worktree_committable
    assert_author_config
    assert_commit_message_clean "$commit_subject"
    checkout_branch "$branch"
    commit_staged_if_any "$commit_subject"
    push_branch "$branch"
    pr_url="$(ensure_pr "$branch" "$pr_title" "$pr_body" | tail -n 1)"
    watch_checks "$pr_url"
}

main "$@"
