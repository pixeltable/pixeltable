import { defineConfig } from 'vitest/config'
import path from 'node:path'

export default defineConfig({
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  esbuild: { jsx: 'automatic' },
  define: { __DASHBOARD_LOGO_URL__: JSON.stringify('/logo.png?v=3') },
  test: { environment: 'happy-dom', include: ['src/**/*.test.{ts,tsx}'] },
})
