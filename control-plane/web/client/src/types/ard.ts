export type ARDTargetKind = "reasoner" | "skill";

export interface ARDRuntimeSettings {
  enabled?: boolean;
  publish_enabled?: boolean;
  registry_public?: boolean;
  public_base_url?: string;
  publisher_domain?: string;
  display_name?: string;
  documentation_url?: string;
  logo_url?: string;
}

export interface ARDEffectiveConfig {
  enabled: boolean;
  publish_enabled: boolean;
  registry_enabled: boolean;
  registry_public: boolean;
  external_search_enabled: boolean;
  external_invocation_enabled: boolean;
  public_base_url: string;
  publisher_domain: string;
  display_name: string;
  documentation_url: string;
  logo_url: string;
  identifier?: string;
  allowed_registries: string[];
  locked: Record<string, boolean>;
}

export interface ARDSummary {
  ard_enabled: boolean;
  catalog_published: boolean;
  public_url_reachable: boolean;
  did_available: boolean;
  published_reasoners: number;
  published_skills: number;
  imported_resources: number;
  callable_external_resources: number;
  catalog_url: string;
}

export interface ARDTrustManifest {
  identity?: string;
  identityType?: "spiffe" | "did" | "https" | "other" | string;
}

export interface ARDAgentFieldMeta {
  targetKind: ARDTargetKind;
  nodeId: string;
  targetId: string;
  invocationTarget: string;
  healthStatus?: string;
  version?: string;
}

export interface ARDCatalogEntry {
  identifier: string;
  displayName: string;
  description?: string;
  type: string;
  url?: string;
  data?: unknown;
  publisher?: string;
  tags?: string[];
  capabilities?: string[];
  representativeQueries?: string[];
  trustManifest?: ARDTrustManifest;
}

export interface ARDCatalogManifest {
  specVersion: string;
  host: {
    displayName: string;
    identifier?: string;
    publisherDomain?: string;
    documentationUrl?: string;
    logoUrl?: string;
  };
  entries: ARDCatalogEntry[];
}

export interface ARDPublication {
  target_kind: ARDTargetKind;
  node_id: string;
  target_id: string;
  published: boolean;
  display_name: string;
  description: string;
  tags: string[];
  capabilities: string[];
  representative_queries: string[];
  artifact_type: string;
  artifact_url_override?: string;
  validation_status: string;
  validation_errors?: string[];
  last_validated_at?: string;
  updated_at?: string;
}

export interface ARDPublicationView extends ARDPublication {
  key: string;
  status: string;
  entry: ARDCatalogEntry;
  agentfield: ARDAgentFieldMeta;
  artifact_url?: string;
  input_schema?: unknown;
}

export interface ARDExternalEntry {
  id: string;
  source_registry?: string;
  identifier: string;
  type: string;
  display_name: string;
  description?: string;
  url?: string;
  data?: unknown;
  publisher?: string;
  trust_summary?: string;
  representative_queries?: string[];
  imported_at: string;
}

export interface ARDExternalBinding {
  external_entry_id: string;
  callable: boolean;
  local_target: string;
  adapter: "openapi" | "mcp" | "a2a" | string;
  auth_ref?: string;
  timeout_ms: number;
  allowed_operations?: string[];
  policy?: string;
  updated_at?: string;
}

export interface ARDExternalImportView {
  entry: ARDExternalEntry;
  binding?: ARDExternalBinding;
  status: string;
}

export interface ARDRegistryRecord {
  url: string;
  name?: string;
  submission_state?: string;
  last_checked_at?: string;
}

export interface ARDState {
  settings: ARDRuntimeSettings;
  publications: Record<string, ARDPublication>;
  imports: ARDExternalEntry[];
  bindings: Record<string, ARDExternalBinding>;
  registries: ARDRegistryRecord[];
  updated_at?: string;
  last_updated_by?: string;
}

export interface ARDDashboard {
  config: ARDEffectiveConfig;
  state: ARDState;
  summary: ARDSummary;
  publications: ARDPublicationView[];
  catalog: ARDCatalogManifest;
  imports: ARDExternalImportView[];
  registries: ARDRegistryRecord[];
}

export interface ARDSearchRequest {
  query: {
    text?: string;
    filter?: Record<string, string[]>;
  };
  federation?: "auto" | "referrals" | "none";
  pageSize?: number;
  pageToken?: string;
  registries?: string[];
}

export interface ARDSearchResult extends ARDCatalogEntry {
  score: number;
  source: string;
}

export interface ARDSearchResponse {
  results: ARDSearchResult[];
  sources: Array<{ url: string; status: string; error?: string }>;
}
