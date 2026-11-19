import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    host: "127.0.0.1",
    port: 1420,
    strictPort: true,
    // Large evaluation outputs, the Rust crate and reference art are not app sources.
    watch: { ignored: ["**/artifacts/**", "**/src-tauri/**", "**/ref/**", "**/tools/**"] },
  },
  optimizeDeps: { entries: ["index.html", "brand.html"] },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 900,
    rollupOptions: { input: { main: "index.html", brand: "brand.html" } },
  },
});
