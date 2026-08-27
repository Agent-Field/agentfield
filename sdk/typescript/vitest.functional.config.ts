import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/harness_functional.test.ts'],
    testTimeout: 300_000,
    hookTimeout: 30_000,
  },
});
