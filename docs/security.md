# Security scaffold

Mneme v0.4 adds operator-facing primitives for auditing injections, encrypting
small tensor banks at rest, and stubbing role checks around sensitive bank
operations. This is a scaffold, not a complete production security boundary.

## Threat model

Covered:

- Accidental or unaudited mutation of hidden states by CAA, SCAR, or LOPI.
- Plaintext tensor-bank files copied from disk.
- Missing role checks before bank load/store or key rotation workflows.

Not covered:

- Compromised Python process memory, model weights, tokenizer, or host kernel.
- Malicious code with access to the Fernet key.
- Identity-provider token validation. Operators must supply that integration.

## Audit log format

`deltamemory.security.AuditLogger` writes JSON lines to a file path or callback
sink. Each event has:

```json
{
  "ts_ns": 0,
  "event_type": "inject|bank_load|bank_store|access_denied",
  "injector": "caa|scar|lopi|null",
  "layer": 0,
  "alpha": 1.0,
  "signal_summary": {
    "steer_norm": 0.0,
    "drift_ratio": 0.0,
    "gate_mean": 1.0
  },
  "vector_hash": "sha256:<64 hex chars>",
  "actor": null,
  "request_id": null
}
```

For injection events, `vector_hash` is computed over the raw bytes of the CAA
steering vector, SCAR projection tensor, or LOPI gate tensor after contiguous CPU
materialization. `alpha=0` paths short-circuit before audit emission to preserve
the bit-equal no-op contract.

## Encrypted bank storage

Use the optional security extra:

```bash
pip install 'deltamemory[security]'
```

Then:

```python
from cryptography.fernet import Fernet
from deltamemory.security import load_encrypted, save_encrypted

key = Fernet.generate_key()
save_encrypted({"layer0": tensor}, "bank.enc", key)
bank = load_encrypted("bank.enc", key)
```

`torch.load` is called with `weights_only=True`. Keys are never logged. Wrong or
invalid keys raise `BankAuthError`.

## Key management recommendations

- Generate and rotate keys outside the repo.
- Store keys in an HSM, cloud KMS, or sealed-secret mechanism.
- Pass keys to the runtime via short-lived environment injection or a secrets
  manager client.
- Never commit keys, encrypted test fixtures with real secrets, or decrypted
  bank dumps.

## RBAC integration guide

`AccessGuard.check(operation, role, actor=...)` implements a local role hierarchy:

- `bank_load` and `inject`: `READER`
- `bank_store`: `WRITER`
- `rotate_key`: `ADMIN`

This guard does not verify tokens. A production operator should:

1. Authenticate a request with an external IdP (OIDC/SAML/JWT).
2. Map IdP claims or groups to `Role.READER`, `Role.WRITER`, or `Role.ADMIN`.
3. Call `AccessGuard.check(...)` before load, store, inject, or key rotation.
4. Attach an `AuditLogger` so denials produce `access_denied` events.
