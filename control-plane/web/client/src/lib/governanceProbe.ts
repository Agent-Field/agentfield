import { getGlobalAdminToken, getGlobalApiKey } from "@/services/api";

const API_BASE = "/api/v1";

function buildHeaders(): HeadersInit {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const apiKey = getGlobalApiKey();
  if (apiKey) headers["X-Api-Key"] = apiKey;
  const admin = getGlobalAdminToken();
  if (admin) headers["X-Admin-Token"] = admin;
  return headers;
}

/** True when `/api/v1/admin/policies` exists (authorization feature enabled on server). */
export async function areGovernanceAdminRoutesAvailable(): Promise<boolean> {
  const res = await fetch(`${API_BASE}/admin/policies`, {
    method: "GET",
    headers: buildHeaders(),
  });
  return res.status !== 404;
}
