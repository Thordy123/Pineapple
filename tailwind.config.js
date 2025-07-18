/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        border: 'hsl(var(--border))',
        ring: "hsl(var(--ring))",
        pineapple: {
          500: "#f59e0b",
          600: "#d97706",
        },
      }
    },
  },
  plugins: [],
}