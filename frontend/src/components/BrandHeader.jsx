import { ShieldCheck } from "lucide-react";
import "../styles/button_glow.css";

export default function BrandHeader({ safeMode, onToggleSafeMode }) {
  return (
    <div className="brand-header px-4 pt-1 pb-1.5 flex justify-between">
      <div className="flex flex-col">
        <div className="text-[66px] font-aclonica -tracking-tight">Aquina</div>
        <div className="flex gap-20">
          <div>
            <div className="text-base font-semibold mb-1">Welcome back, Pranavvvvv 👋</div>
            <div className="text-[12.5px] text-ink-muted">Let's build something incredible today.</div>
          </div>
          <div>
            <button
              onClick={onToggleSafeMode}
              className={`glow-btn glow-safe inline-flex items-center justify-center gap-2 text-xs font-medium rounded-[18px] h-10 w-[120px] py-[7px] transition-colors
          ${safeMode
                  ? "text-ink-secondary bg-white/[0.06]"
                  : "text-accent-blue bg-accent-blueSoft"
                }`}
            >
              <ShieldCheck size={14} />
              {safeMode ? "Safe Mode: On" : "Safe Mode"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}