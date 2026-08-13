import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Dev-only: proxies to the FastAPI backend so the app is same-origin
    // from the browser's perspective (matches production, where the
    // backend serves this app's built static files itself) -- keeps the
    // session cookie working without any CORS configuration anywhere.
    proxy: {
      "/api": "http://localhost:8000",
      "/auth": "http://localhost:8000",
    },
  },
});
