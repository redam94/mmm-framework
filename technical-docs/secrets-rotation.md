# Secrets Rotation Runbook

How to rotate the platform's secrets. Pair with
[operations-runbook.md](operations-runbook.md).

## Secrets inventory

| Secret | Env | Used for | Rotation impact |
|--------|-----|----------|-----------------|
| JWT signing key | `MMM_AUTH_SECRET` | signs/verifies access + refresh tokens (HS256) | rotating invalidates live tokens unless overlap is supported (see below) |
| Encryption-at-rest key | `MMM_ENCRYPTION_KEY` | Fernet encryption of stored data | **zero-downtime** — dual-key supported |
| LLM provider key | `MMM_LLM_API_KEY` / provider creds | agent LLM calls | transparent; next call uses the new key |
| Legacy REST API key(s) | `VALID_API_KEYS` | the deprecated `api/` stack | rotate by updating the allowlist |

Store all of these in a secret manager (Vault / cloud KMS / sealed secrets), never
in the repo or a plaintext `.env` in production.

## Encryption-at-rest key (`MMM_ENCRYPTION_KEY`) — zero downtime

`DatasetEncryptor` uses `MultiFernet`, so it accepts a **comma-separated** key
list: the FIRST key encrypts new data; ALL keys are tried for decryption. Rotate
without downtime:

1. Generate a new key:
   `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
2. Set `MMM_ENCRYPTION_KEY="<new>,<old>"` and restart. New writes use `<new>`;
   data encrypted with `<old>` still decrypts.
3. (Optional) Re-encrypt at rest by reading + rewriting artifacts.
4. Once all data is re-encrypted, drop `<old>`: `MMM_ENCRYPTION_KEY="<new>"`.

## JWT signing key (`MMM_AUTH_SECRET`)

**Current behavior:** a single signing key signs and verifies all tokens.
Rotating it invalidates every live access/refresh token, so all users must
re-authenticate. Procedure:

1. Pick a low-traffic window; notify users (brief re-login).
2. Set the new `MMM_AUTH_SECRET`; restart all API replicas together (so they
   share one key — a mixed fleet would reject each other's tokens).
3. Users log in again; refresh tokens are reissued under the new key.

Access tokens are short-lived, so the disruption is bounded.

**Recommended enhancement (zero-downtime, follow-up):** mirror the encryption
key's dual-key pattern — accept a comma-separated `MMM_AUTH_SECRET` where the
first key signs and ALL are accepted for verification. Then a rotation is: add
the new key first, deploy, wait past the refresh-token TTL, drop the old key — no
forced logout. This requires verifying tokens against the key list in the auth
verifier (`auth/tokens.py`); it is not yet implemented.

## Emergency revocation

Independent of key rotation, the auth layer supports instant revocation via
token-version generations (deactivate user / password reset / sign-out-everywhere)
and a refresh-token blocklist — use these for a single compromised account rather
than rotating the global signing key.

## Verification after rotation

- `GET /health` on every replica.
- A fresh login + an authenticated request succeed.
- For encryption: read back a previously-stored encrypted artifact (decrypts via
  the retained old key).
