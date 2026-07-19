import { useState } from "react";
import { Pause, Play } from "lucide-react";
import { NOW_PLAYING } from "../data/mockData.js";

export default function NowPlaying() {
  const [playing, setPlaying] = useState(true);

  return (
    <section className="panel flex-1 relative flex items-center gap-3 px-4 py-3.5">
      <div
        className="w-[38px] h-[38px] rounded-[10px] flex-shrink-0"
        style={{ background: "conic-gradient(from 200deg, #4fb2ff, #ff5a5a, #4fb2ff)" }}
        aria-hidden="true"
      />
      <div className="flex-1 min-w-0 flex flex-col gap-1">
        <span className="text-xs font-semibold whitespace-nowrap overflow-hidden text-ellipsis">
          {NOW_PLAYING.title}
        </span>
        <span className="text-[10.5px] text-ink-muted">{NOW_PLAYING.subtitle}</span>
        <div className="h-[3px] bg-white/[0.08] rounded-full overflow-hidden">
          <span className="block h-full bg-accent-red" style={{ width: `${NOW_PLAYING.progress * 100}%` }} />
        </div>
      </div>
      <button
        onClick={() => setPlaying((p) => !p)}
        className="w-[30px] h-[30px] rounded-full bg-ink-primary text-[#0a0a0a] flex items-center justify-center flex-shrink-0"
      >
        {playing ? <Pause size={14} /> : <Play size={14} />}
      </button>
      <span className="absolute top-2.5 right-3 text-[9px] text-accent-green">Spotify</span>
    </section>
  );
}
