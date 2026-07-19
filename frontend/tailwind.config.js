/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        mesh: {
          navyDeep: "#071522",
          navy: "#0d2a40",
          navyGlow: "#1c5578",
          redDeep: "#2a0808",
          red: "#6e1414",
          redBright: "#8f1c1c",
        },
        panel: {
          DEFAULT: "rgba(10, 19, 28, 0.52)",
          strong: "rgba(8, 15, 23, 0.68)",
          border: "rgba(255, 255, 255, 0.08)",
          borderHover: "rgba(255, 255, 255, 0.16)",
        },
        ink: {
          primary: "#fdf0d5",
          secondary: "#aebbc7",
          muted: "#728296",
          faint: "#4d5c6d",
        },
        accent: {
          blue: "#4fb2ff",
          blueSoft: "rgba(79, 178, 255, 0.16)",
          red: "#805c57",
          redSoft: "rgba(255, 90, 90, 0.16)",
          green: "#3ddc84",
        },
        platform: {
          github: "#3fd67a",
          leetcode: "#ffc94d",
          codeforces: "#4aa3ff",
          codechef: "#c96a35",
        },
      },
      fontFamily: {
        aclonica: ["Aclonica", "sans-serif"],
        display: ["Fraunces", "Iowan Old Style", "Georgia", "serif"],
        body: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "SFMono-Regular", "Menlo", "monospace"],
      },
      borderRadius: {
        panel: "18px",
      },
      backdropBlur: {
        panel: "22px",
      },
      boxShadow: {
        panel:
          "inset 0 1px 0 rgba(255,255,255,0.04), 0 20px 40px -20px rgba(0,0,0,0.6)",
      },
      keyframes: {
        breathe: {
          "0%": { transform: "scale(1)", opacity: "0.92" },
          "100%": { transform: "scale(1.04)", opacity: "1" },
        },
      },
      animation: {
        breathe: "breathe 9s ease-in-out infinite alternate",
      },
    },
  },
  plugins: [],
};
