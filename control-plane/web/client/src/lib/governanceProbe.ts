import { getGlobalApiKey } from "@/services/api";

const API_BASE = "/api/v1";

/**
 * What the server's governance admin routes require from this browser.
 *
 * - "open": routes exist and no admin token is enforced (AdminTokenAuth is a
 *   no-op server-side) — admin calls work with the API key alone.
 * - "token-required": routes exist and a non-empty admin token is enforced.
 * - "unavailable": authorization feature is disabled — routes not registered.
 * - "unauthorized": the API key is missing/invalid; says nothing about the
 *   admin token.
 */
export type GovernanceAdminAccess =
  | "open"
  | "token-required"
  | "unavailable"
  | "unauthorized";

/**
 * Probe `/api/v1/admin/policies` to learn what admin access requires.
 *
 * The request deliberately omits X-Admin-Token even when one is stored: the
 * tokenless status is what discriminates. AdminTokenAuth wraps the whole
 * admin group, so a 200 without a token proves no token is enforced for any
 * admin route, while a 403 proves one is.
 */
export async function probeGovernanceAdminAccess(): Promise<GovernanceAdminAccess> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const apiKey = getGlobalApiKey();
  if (apiKey) headers["X-Api-Key"] = apiKey;

  const res = await fetch(`${API_BASE}/admin/policies`, {
    method: "GET",
    headers,
  });

  if (res.status === 404) return "unavailable";
  if (res.status === 401) return "unauthorized";
  if (res.status === 403) return "token-required";
  if (res.ok) return "open";
  throw new Error(`Governance probe failed with status ${res.status}`);
}
