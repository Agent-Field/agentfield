import { useQuery } from "@tanstack/react-query";
import { areGovernanceAdminRoutesAvailable } from "@/lib/governanceProbe";
import { listPolicies } from "@/services/accessPoliciesApi";
import { listAllAgentsWithTags } from "@/services/tagApprovalApi";

/** Shared TanStack Query prefix for access policy + agent tag admin data. */
export const ACCESS_MANAGEMENT_QUERY_KEY = "access-management";

export function useAccessAdminRoutesProbe(adminToken: string | null) {
  return useQuery({
    queryKey: [ACCESS_MANAGEMENT_QUERY_KEY, "probe", adminToken ?? ""],
    queryFn: areGovernanceAdminRoutesAvailable,
    staleTime: 60_000,
    retry: 1,
  });
}

export function useAccessPolicies(enabled: boolean) {
  return useQuery({
    queryKey: [ACCESS_MANAGEMENT_QUERY_KEY, "policies"],
    queryFn: async () => {
      const res = await listPolicies();
      return res.policies ?? [];
    },
    enabled,
    staleTime: 30_000,
  });
}

export function useAgentTagSummaries() {
  return useQuery({
    queryKey: [ACCESS_MANAGEMENT_QUERY_KEY, "agent-tags"],
    queryFn: async () => {
      const res = await listAllAgentsWithTags();
      return res.agents ?? [];
    },
    staleTime: 15_000,
  });
}
