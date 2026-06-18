import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend the dev server proxies API traffic to. Override with MMM_API_PROXY_TARGET
// (e.g. when the FastAPI backend runs on another host/port).
const API_PROXY_TARGET = process.env.MMM_API_PROXY_TARGET || 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Listen on all interfaces so a tunnel (cloudflared / ngrok / VS Code port
    // forwarding) can reach the dev server.
    host: true,
    // Allow requests from any tunnel hostname. Vite otherwise rejects unknown
    // Host headers with "Blocked request. This host is not allowed." Dev-only.
    allowedHosts: true,
    proxy: {
      // Route all backend calls through the dev server so only ONE port needs to
      // be forwarded/tunneled. The frontend talks to /api/* in dev; we strip the
      // prefix before forwarding to the backend, whose routes live at the root
      // (/chat, /sessions, /health, ...). Streaming responses (e.g. /chat) are
      // piped through unbuffered by Vite's proxy.
      '/api': {
        target: API_PROXY_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
