/**
 * Shared API types.
 *
 * The modern app talks to the agent API (src/mmm_framework/api) through
 * feature-specific service modules (see ./services). The legacy REST-API
 * request/response types that used to live here were removed with the
 * services that consumed them; only the transport-level error shape the
 * axios client rejects with remains.
 */

export interface ApiError {
  status: number;
  message: string;
  details?: unknown;
}
