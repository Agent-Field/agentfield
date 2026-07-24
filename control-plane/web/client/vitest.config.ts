import path from "path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    test: {
        environment: "jsdom",
        setupFiles: ["./src/test/setup.ts"],
        globals: true,
        coverage: {
            all: true,
            provider: "v8",
            include: ["src/**/*.{ts,tsx}"],
            exclude: [
                "dist/**",
                "node_modules/**",
                "src/test/**",
                "src/**/*.d.ts",
            ],
            reporter: ["text-summary", "json-summary", "cobertura"],
            reportsDirectory: "coverage",
        },
    },
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
});
