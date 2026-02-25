import { defineConfig } from 'vite'
import basicSsl from '@vitejs/plugin-basic-ssl'

export default defineConfig({
  plugins: [basicSsl()],
  server: {
    host: true,  // 让局域网设备能访问
    https: true, // 开启 https
  },
})
