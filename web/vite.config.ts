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
      // The local API from #19. Proxied in development so the browser sees
      // one origin and no CORS configuration is needed.
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
    },
  },
});
