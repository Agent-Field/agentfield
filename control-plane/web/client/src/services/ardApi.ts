import { getGlobalApiKey } from "@/services/api";
import type {
  ARDCatalogEntry,
  ARDDashboard,
  ARDExternalBinding,
  ARDPublication,
  ARDRegistryRecord,
  ARDRuntimeSettings,
  ARDSearchRequest,
  ARDSearchResponse,
} from "@/types/ard";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/ui/v1";

async function ardFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  const apiKey = getGlobalApiKey();
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }

  const response = await fetch(`${API_BASE_URL}/ard${path}`, {
    ...options,
    headers,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || payload.message || `ARD request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getARDDashboard(): Promise<ARDDashboard> {
  return ardFetch<ARDDashboard>("");
}

export function updateARDSettings(settings: ARDRuntimeSettings): Promise<ARDDashboard> {
  return ardFetch<ARDDashboard>("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export function saveARDPublication(publication: ARDPublication): Promise<ARDDashboard> {
  return ardFetch<ARDDashboard>("/publications", {
    method: "PUT",
    body: JSON.stringify(publication),
  });
}

export function searchExternalARD(request: ARDSearchRequest): Promise<ARDSearchResponse> {
  return ardFetch<ARDSearchResponse>("/external/search", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function importARDEntry(entry: ARDCatalogEntry, sourceRegistry?: string): Promise<ARDDashboard> {
  return ardFetch<ARDDashboard>("/imports", {
    method: "POST",
    body: JSON.stringify({
      source_registry: sourceRegistry,
      entry,
    }),
  });
}

export function saveARDBinding(entryId: string, binding: ARDExternalBinding): Promise<ARDDashboard> {
  return ardFetch<ARDDashboard>(`/imports/${encodeURIComponent(entryId)}/binding`, {
    method: "PUT",
    body: JSON.stringify(binding),
  });
}

export function saveARDRegistries(registries: ARDRegistryRecord[]): Promise<ARDDashboard> {
  return ardFetch<ARDDashboard>("/registries", {
    method: "PUT",
    body: JSON.stringify({ registries }),
  });
}
