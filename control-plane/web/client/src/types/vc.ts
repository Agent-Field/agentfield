import type { ExecutionVC } from "./did";

export interface VCStatusData
  extends Partial<Pick<ExecutionVC, "storage_uri" | "document_size_bytes">> {
  has_vc: boolean;
  vc_id?: string;
  status: string;
  created_at?: string;
  vc_document?: Record<string, unknown>;
  original_status?: string;
}
