# Branch audit — 2026-05-05

Read-only inventory of remote branches on `origin`. No branches were deleted.

## Summary

- Remote branches audited (excluding `origin/main` and `origin/HEAD`): 4
- Merged into `origin/main`: 0
- Stale branches (>14 days without push and no open PR): 0
- Branches with open PRs: 4

## Open PR branches — keep

- `fix/ci-green-pyarrow` (#14) → `main` — fix(ci): add pyarrow to test extras
- `chore/rename-sweep` (#15) → `main` — chore: rename residual RCV-HC/DeltaMemory references to MnEmE/Mneme
- `feat/pr-helper` (#16) → `main` — feat(scripts): branch_pr_helper.sh for PR-only workflow
- `chore/githooks` (#17) → `main` — chore(hooks): forbid Copilot co-author trailer; lock author=KeikaJames

## Deletion plan

- No remote branches are recommended for deletion in this audit.

## Branch inventory

| branch | last_commit | age_days | merged_into_main | open_PR | recommendation |
| --- | --- | ---: | --- | --- | --- |
| `chore/githooks` | 2026-05-05T20:50:52+08:00 | 0 | no | #17 | keep |
| `chore/rename-sweep` | 2026-05-05T20:56:07+08:00 | 0 | no | #15 | keep |
| `feat/pr-helper` | 2026-05-05T20:53:55+08:00 | 0 | no | #16 | keep |
| `fix/ci-green-pyarrow` | 2026-05-05T21:02:37+08:00 | 0 | no | #14 | keep |
