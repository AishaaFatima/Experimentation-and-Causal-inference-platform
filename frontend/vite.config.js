import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: { chunkSizeWarningLimit: 900 },
  test: { environment: "jsdom", globals: false, setupFiles: ["./src/test/setup.js"] },
});
