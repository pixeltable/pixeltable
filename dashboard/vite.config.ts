import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { readFileSync } from 'node:fs'
import { viteSingleFile } from 'vite-plugin-singlefile'

export default defineConfig(({ mode }) => {
  const embedded = mode === 'embedded'
  return {
    plugins: [react(), ...(embedded ? [viteSingleFile()] : [])],
    publicDir: embedded ? false : 'public',
    define: {
      __DASHBOARD_LOGO_URL__: JSON.stringify(embedded
        ? 'data:image/png;base64,' + readFileSync(path.resolve(__dirname, 'public/logo.png')).toString('base64')
        : '/logo.png?v=3'),
    },
    resolve: { alias: { '@': path.resolve(__dirname, './src') } },
    server: {
      proxy: {
        '/api': { target: `http://localhost:${process.env.PXT_PORT || 22089}`, changeOrigin: true },
      },
    },
    build: {
      outDir: embedded ? '../pixeltable_cli/server/embedded' : '../pixeltable_cli/server/static',
      ...(embedded ? { rollupOptions: { input: path.resolve(__dirname, 'embedded.html') } } : {}),
      emptyOutDir: true,
      sourcemap: false,
    },
  }
})
