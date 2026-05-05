# Changelog

Keep-a-Changelog format. Dates are omitted for historical backfill; commit SHAs identify source changes.

## [0.4.0]
### Added
- W.0-W.14 preregistration/scaffold work (`936b0458`, `5d044870`, `a6a6a7b7`, `6f5bd180`, `ad579c4d`, `831983d2`, `70bdb173`, `65769acf`).
- CAA, SCAR, diagnostics, Gemma2/Gemma3 adapters, shared layer locator (`4a287904`, `a6ae69fa`, `bf0eb65d`, `e1869163`, `c6d592ca`, `4f9cca6d`).
- API docs, FastAPI scaffold, vLLM plan, migration guide, versioning policy (this PR).
### Changed
- Project label changed to Mneme while preserving `deltamemory` imports (`84cc340d`, `09ed55d0`, `55d46473`).
### Deprecated
- Legacy modules and mHC/LOPI production-default usage (`b15969e5`, `ad579c4d`).
### Removed
- Infra files deleted in PR #27 (`71125860`).
### Fixed
- CI/audit/diagnostics issues (`d51ddfef`, `265e1303`, `d02664a8`, `bf0eb65d`).
### Security
- FastAPI security notes; PR-only workflow guardrails (`05c5a30d`, `3a0571a6`).

## [3.6]
### Added
- V-scale config and `ulopi_v36`.
### Changed
- `auto_rms_cap` for no-`v_norm` families.
### Deprecated
- Older schemas compatibility-only.
### Removed
- Full-matrix Sinkhorn path.
### Fixed
- V-scale mismatch.
### Security
- No security changes.

## [3.2]
### Added
- mHC shield.
### Changed
- Optional bank-column cap.
### Deprecated
- None.
### Removed
- Full-matrix Sinkhorn path.
### Fixed
- Bank-column amplification bound.
### Security
- No security changes.

## [3.1]
### Added
- Attn-native bank and adapters.
### Changed
- README vocabulary.
### Deprecated
- Earlier paths historical.
### Removed
- None.
### Fixed
- Architecture alpha defaults.
### Security
- No security changes.
