import { FileText, Plus } from "lucide-react";

export default function QuickNote() {
  return (
    <section className="panel flex-1 flex flex-col px-[18px] py-4">
      <header className="panel-header">
        <div className="panel-title">
          <FileText size={15} /> Quick Note
        </div>
        <button className="icon-btn" aria-label="Add note">
          <Plus size={14} />
        </button>
      </header>
      <p className="flex-1 text-[12.5px] text-ink-secondary leading-[1.45] m-0 mb-2.5">
        Finish the AI model integration by this weekend.
      </p>
      <span className="text-[10.5px] text-ink-faint">Today, 11:45 AM</span>
    </section>
  );
}
