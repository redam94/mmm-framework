import { describe, it, expect } from 'vitest';
import { requiresApiKey } from './loginMode';

describe('requiresApiKey (U3)', () => {
  it('never asks for a key when password auth is enabled', () => {
    expect(requiresApiKey(null, true)).toBe(false);
    expect(requiresApiKey({ requires_api_key: true }, true)).toBe(false);
  });

  it('does not ask when the server is managed (provides the model)', () => {
    expect(requiresApiKey({ requires_api_key: false }, false)).toBe(false);
  });

  it('asks when the server requires a key and auth is off', () => {
    expect(requiresApiKey({ requires_api_key: true }, false)).toBe(true);
  });

  it('asks (safe default) when the server posture is unknown', () => {
    expect(requiresApiKey(null, false)).toBe(true);
    expect(requiresApiKey(undefined, false)).toBe(true);
    expect(requiresApiKey({}, false)).toBe(true);
  });
});
