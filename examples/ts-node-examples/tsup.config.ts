import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['discovery-memory/main.ts', 'simulation/main.ts'],
  format: ['esm'],
  splitting: false,
  dts: false,
  sourcemap: true,
  clean: true
});
