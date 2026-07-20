import react from '@vitejs/plugin-react'
import { defineConfig } from 'electron-vite'

// electron-vite defaults (kept deliberately):
//   main:     src/main/index.ts     -> out/main/index.js   (CJS, package has no "type": "module")
//   preload:  src/preload/index.ts  -> out/preload/index.js (CJS — required for sandbox: true)
//   renderer: src/renderer/index.html + src/renderer/src   -> out/renderer
export default defineConfig({
  main: {},
  preload: {},
  renderer: {
    plugins: [react()]
  }
})
