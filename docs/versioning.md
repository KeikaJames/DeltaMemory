# Versioning policy

Mneme follows SemVer 2.0.0: major for methodological breakage, minor for additive features, patch for bugfix/test/doc-only changes. Public API is everything in `deltamemory.__all__` plus the documented `DiagnosticRecorder` schema (`step`, `layer`, `token`, `signal_name`, `value`). Unstable surfaces include `deltamemory.experiments.*`, `deltamemory.runtime.*` if added, runners, aggregation scripts, and private names except documented extraction points such as `_layer_locator`. Releases are PR-driven, not calendar-driven.
