import { Minus, Square, X } from "lucide-react";

// Tauri's window is created with decorations:false (see
// src-tauri/tauri.conf.json), so this bar is the *only* chrome —
// it owns dragging and the min/max/close controls.
//
// The window API is imported lazily on click so this file still
// renders fine in a plain browser (e.g. `npm run dev` without
// `tauri dev`), where "@tauri-apps/api" has nothing to talk to.
async function withAppWindow(fn) {
  try {
    const { appWindow } = await import("@tauri-apps/api/window");
    fn(appWindow);
  } catch {
    // Not running inside Tauri (e.g. plain browser dev) — no-op.
  }
}

export default function TitleBar() {
  return (
    <div
      className="h-9 min-h-[36px] flex items-center justify-between pl-3.5 pr-2.5 z-50"
      data-tauri-drag-region
    >
      <div
        className="flex items-center gap-2 font-display text-[13px] text-ink-secondary tracking-wide"
        data-tauri-drag-region
      >
        <span className="w-1.5 h-1.5 rounded-full bg-accent-blue shadow-[0_0_8px_theme(colors.accent.blue)]" />
        <span>Aquina</span>
      </div>
      <div className="flex gap-1">
        <button
          className="w-[30px] h-6 flex items-center justify-center rounded-md text-ink-muted
            hover:bg-white/[0.08] hover:text-ink-primary transition-colors"
          aria-label="Minimize"
          onClick={() => withAppWindow((w) => w.minimize())}
        >
          <Minus size={13} />
        </button>
        <button
          className="w-[30px] h-6 flex items-center justify-center rounded-md text-ink-muted
            hover:bg-white/[0.08] hover:text-ink-primary transition-colors"
          aria-label="Maximize"
          onClick={() => withAppWindow((w) => w.toggleMaximize())}
        >
          <Square size={11} />
        </button>
        <button
          className="w-[30px] h-6 flex items-center justify-center rounded-md text-ink-muted
            hover:bg-accent-red hover:text-white transition-colors"
          aria-label="Close"
          onClick={() => withAppWindow((w) => w.close())}
        >
          <X size={13} />
        </button>
      </div>
    </div>
  );
}
