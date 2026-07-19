import { Code2, FolderOpen, Bot, FileText, TerminalSquare } from "lucide-react";
import "../styles/button_glow.css";

const ICONS = {
  code: Code2,
  projects: FolderOpen,
  assistant: Bot,
  notes: FileText,
  terminal: TerminalSquare,
};

const ACTIONS = [
  { key: "code", label: "Code", glow: "glow-blue" },
  { key: "projects", label: "Projects", glow: "glow-blue" },
  { key: "assistant", label: "AI Assistant", glow: "glow-blue" },
  { key: "notes", label: "Notes", glow: "glow-red" },
  { key: "terminal", label: "Terminal", glow: "glow-red" },
];

export default function QuickActions() {
  return (
    <div className="grid grid-cols-5 gap-2">
      {ACTIONS.map(({ key, label, glow }) => {
        const Icon = ICONS[key];
        return (
          <button
            key={key}
            className={`glow-btn ${glow} panel flex flex-col items-center justify-center gap-2 px-1.5 py-4 text-[11px]
              text-ink-secondary hover:text-accent-blue
              transition-colors`}
          >
            <Icon size={20} />
            <span>{label}</span>
          </button>
        );
      })}
    </div>
  );
}