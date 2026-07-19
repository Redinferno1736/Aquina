import { useEffect, useRef } from "react";
import anime from "animejs";

/**
 * MeshGradientBackground
 * ------------------------------------------------------------------
 * Two ways to render Aquina's signature navy → deep-red mesh:
 *
 *  mode="image"     (default) — uses the actual reference PNG
 *                    (public/mesh-gradient.png) as a full-bleed
 *                    background, with a very slow animated
 *                    "breathing" scale/opacity so the app doesn't
 *                    feel like a static wallpaper.
 *
 *  mode="generated" — recreates the gradient from scratch on a
 *                    <canvas>, using control points modeled on the
 *                    mesh-gradient reference points you shared
 *                    (navy cluster top-left/left, red cluster
 *                    right, a soft lighter-blue glow bottom-left).
 *                    Use this if you ever want to drop the PNG
 *                    asset and keep everything code-generated, or
 *                    want to re-theme the mesh (e.g. per Safe Mode).
 *
 * Both modes sit in a fixed, full-viewport layer behind the app
 * shell (z-index: 0) — every screen/panel renders on top of it.
 */

const CONTROL_POINTS = [
  // Modeled on the two reference panels: navy dominates the
  // top-left/left edge, red dominates the right, with a soft
  // brighter-blue glow low on the left side.
  { x: 0.06, y: 0.08, color: "#0d2a40", radius: 0.55 },
  { x: 0.18, y: 0.42, color: "#0c2233", radius: 0.6 },
  { x: 0.14, y: 0.78, color: "#1c5578", radius: 0.42 }, // bright glow
  { x: 0.02, y: 0.95, color: "#071522", radius: 0.5 },
  { x: 0.55, y: 0.12, color: "#3a0d0d", radius: 0.55 },
  { x: 0.82, y: 0.3, color: "#6e1414", radius: 0.6 },
  { x: 0.95, y: 0.65, color: "#8f1c1c", radius: 0.55 },
  { x: 0.7, y: 0.9, color: "#4a0f0f", radius: 0.5 },
];

function drawMesh(ctx, w, h, points, t) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a1c2b";
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = "screen";

  points.forEach((p, i) => {
    // Tiny orbital drift per point so the mesh feels alive, not static.
    const drift = 14 * Math.sin(t * 0.0002 + i * 1.7);
    const cx = p.x * w + drift;
    const cy = p.y * h + drift * 0.6;
    const r = p.radius * Math.max(w, h);

    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    g.addColorStop(0, p.color);
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  });

  ctx.globalCompositeOperation = "source-over";
}

function GeneratedMesh() {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      canvas.width = window.innerWidth * devicePixelRatio;
      canvas.height = window.innerHeight * devicePixelRatio;
      canvas.style.width = "100vw";
      canvas.style.height = "100vh";
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const loop = (t) => {
      drawMesh(ctx, window.innerWidth, window.innerHeight, CONTROL_POINTS, t);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return <canvas ref={canvasRef} className="mesh-canvas absolute inset-0 block" />;
}

function ImageMesh() {
  const layerRef = useRef(null);

  useEffect(() => {
    // Slow ambient breathing — subtle scale + opacity drift.
    // This is the "normal AI-active state: warm/blue, alive" cue
    // from the brief, applied at the background layer.
    anime({
      targets: layerRef.current,
      scale: [1, 1.04],
      opacity: [0.92, 1],
      duration: 9000,
      easing: "easeInOutSine",
      direction: "alternate",
      loop: true,
    });
  }, []);

  return (
    <div
      ref={layerRef}
      className="absolute -inset-[4%] bg-cover bg-center bg-no-repeat will-change-transform"
      style={{ backgroundImage: `url(${import.meta.env.BASE_URL}mesh-gradient.png)` }}
    />
  );
}

export default function MeshGradientBackground({ mode = "image" }) {
  return (
    <div className="fixed inset-0 z-0 overflow-hidden bg-[#0a1c2b]" aria-hidden="true">
      {mode === "generated" ? <GeneratedMesh /> : <ImageMesh />}
      {/* Fine grain keeps the smooth gradient from banding on large
          panels, and reads as an intentional "material". */}
      <div className="mesh-grain absolute inset-0 opacity-[0.05] mix-blend-overlay" />
      {/* Darkens the outer edges so foreground panels stay legible
          regardless of what's happening in the mesh underneath them. */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 45%, rgba(0,0,0,0) 40%, rgba(3,8,14,0.55) 100%)",
        }}
      />
    </div>
  );
}
