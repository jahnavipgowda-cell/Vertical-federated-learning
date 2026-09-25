/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0B1220",
        panel: "#121B2E",
        panelAlt: "#182640",
        line: "#24314D",
        signal: "#3FD9C7",
        signalDim: "#1F5F58",
        mask: "#F0B429",
        maskDim: "#5A4419",
        danger: "#F0654F",
        textPrimary: "#E7EEF7",
        textMuted: "#8CA0C2",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'IBM Plex Sans'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};
