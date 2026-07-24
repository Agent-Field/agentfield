import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    exclude: [
      "tests/harness_functional.test.ts",
      "tests/mcp.test.ts",
      "tests/mcp_client.test.ts",
      "tests/mcp_registry.test.ts",
    ],
    coverage: {
      all: true,
      provider: "v8",
      include: ["src/**/*.ts"],
      exclude: [
        "dist/**",
        "src/mcp/**",
        "src/types/mcp.ts",
        "src/**/*.d.ts",
        "src/**/__tests__/**",
      ],
      reporter: ["text-summary", "json-summary", "cobertura"],
      reportsDirectory: "coverage",
    },
  },
});
