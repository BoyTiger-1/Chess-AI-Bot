import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg'],
      manifest: {
        name: 'AI Business Assistant',
        short_name: 'AI Assistant',
        description: 'Premium dashboards and workflows for business insights.',
        theme_color: '#0b1220',
        background_color: '#0b1220',
        display: 'standalone',
        scope: '/',
        start_url: '/'
      }
    })
  ],
  resolve: {
    alias: {
      '@': new URL('./src', import.meta.url).pathname
    }
  },
  server: {
    port: 5173
  }
});
