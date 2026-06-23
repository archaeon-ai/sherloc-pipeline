import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// Minimal vitest config for Svelte component tests (issue #21 Round 2 F3).
// `hot: false` keeps the Svelte plugin from injecting HMR runtime code
// that jsdom can't execute. `resolve.conditions: ['browser']` matches
// the default @testing-library/svelte v4 setup so the svelte client
// runtime is used rather than the server one.
export default defineConfig({
  plugins: [svelte({ hot: false })],
  resolve: {
    conditions: ['browser'],
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.ts'],
  },
});
