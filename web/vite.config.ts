import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  build: {
    // FastAPI serves this directory as static files, so students never need
    // Node installed. See #25.
    outDir: 'dist',
  },
  server: {
    port: 5173,
    proxy: {
      // The local API from #19. Its routes are at the root of the server
      // (/ask, /chunks, ...), so the /api prefix the dev client adds is
      // stripped here. Proxied so the browser sees one origin and no CORS
      // configuration is needed. In production the bundle is served by that
      // same server and the client uses no prefix.
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
