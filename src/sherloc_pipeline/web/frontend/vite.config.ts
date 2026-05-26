import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api/ws': { target: 'ws://localhost:8002', ws: true },
      '/api': 'http://localhost:8002',
    },
  },
});
