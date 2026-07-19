import { useState } from "react";
import MeshGradientBackground from "./components/MeshGradientBackground.jsx";
import TitleBar from "./components/TitleBar.jsx";
import Dashboard from "./components/Dashboard.jsx";
import "./styles/mesh.css";
import "./styles/button_glow.css";

export default function App() {
  const [safeMode, setSafeMode] = useState(false);

  return (
    <div className={`relative h-screen w-screen z-10 flex flex-col ${safeMode ? "temp-safe" : ""}`}>
      {/* mode="image" uses public/mesh-gradient.png (the reference asset).
          Flip to mode="generated" to render the mesh from the control
          points instead, e.g. for a Safe Mode re-theme later. */}
      <MeshGradientBackground mode="image" />
      <TitleBar />
      <main className="flex-1 overflow-y-auto overflow-x-hidden">
        <Dashboard safeMode={safeMode} onToggleSafeMode={() => setSafeMode((v) => !v)} />
      </main>
    </div>
  );
}
