/**
 * Decide whether the login screen must ask for an LLM API key (U3).
 *
 * The "exec can't self-serve" problem: the Login page demanded an Anthropic-style
 * key. It shouldn't when the deployment provides the model:
 *  - password auth on  -> the user logs in with email/password, no LLM key.
 *  - server-managed     -> the backend supplies the model (Vertex / local), so the
 *    user picks a model but supplies no key.
 * Only a key-required server with auth off should surface the key field.
 */
export interface ServerLoginConfig {
  requires_api_key?: boolean;
}

export function requiresApiKey(
  serverConfig: ServerLoginConfig | null | undefined,
  authEnabled: boolean,
): boolean {
  if (authEnabled) return false; // password auth: no LLM key at the login step
  if (serverConfig == null) return true; // unknown server posture -> ask, to be safe
  return serverConfig.requires_api_key !== false; // server-managed model -> no key
}
