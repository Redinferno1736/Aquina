import { Search, ArrowRight, Clock, Code2, FolderOpen, FileStack, Globe } from "lucide-react";

const CHIPS = [
  { label: "Recents", icon: Clock },
  { label: "Code Snippets", icon: Code2 },
  { label: "Projects", icon: FolderOpen },
  { label: "Documentation", icon: FileStack },
  { label: "Web", icon: Globe },
];

export default function AISearch() {
  return (
    <section className="panel px-[18px] py-4">
      <header className="panel-header">
        <div className="panel-title">
          <Search size={15} /> AI-Powered Search
        </div>
      </header>

      <div className="flex items-center gap-2 bg-white/[0.04] border border-panel-border rounded-xl px-3 py-2.5 mb-3">
        <Search size={16} className="text-ink-muted flex-shrink-0" />
        <input
          type="text"
          placeholder="Search your code, projects, docs..."
          className="flex-1 bg-transparent border-none outline-none text-ink-primary text-[12.5px] placeholder:text-ink-faint"
        />
        <kbd className="text-[10px] text-ink-muted bg-white/[0.06] rounded px-1.5 py-0.5 font-mono">Ctrl K</kbd>
        <button
          aria-label="Search"
          className="w-[26px] h-[26px] rounded-lg bg-accent-blue text-[#06131f] flex items-center justify-center flex-shrink-0"
        >
          <ArrowRight size={15} />
        </button>
      </div>

      <div className="flex flex-wrap gap-2">
        {CHIPS.map(({ label, icon: Icon }) => (
          <button
            key={label}
            className="flex items-center gap-1.5 text-[11px] text-ink-secondary bg-white/[0.04] border border-panel-border
              rounded-full px-2.5 py-1.5 hover:border-panel-borderHover hover:text-ink-primary transition-colors"
          >
            <Icon size={12} /> {label}
          </button>
        ))}
      </div>
    </section>
  );
}
