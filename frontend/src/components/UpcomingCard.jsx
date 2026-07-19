import { CalendarDays, ArrowRight } from "lucide-react";
import { UPCOMING } from "../data/mockData.js";

const TAG_CLASS = {
  Contest: "tag-red",
  Study: "tag-blue",
  Deadline: "tag-orange",
};

export default function UpcomingCard() {
  return (
    <section className="panel flex-1 flex flex-col px-[18px] py-4">
      <header className="panel-header">
        <div className="panel-title">
          <CalendarDays size={15} /> Upcoming
        </div>
      </header>

      <ul className="flex-1 flex flex-col gap-2.5 list-none m-0 mb-2.5 p-0">
        {UPCOMING.map((item) => (
          <li key={item.title} className="flex items-center gap-2.5">
            <div className="flex-none w-10 flex flex-col items-center py-1 rounded-[9px] bg-white/5 border border-panel-border">
              <span className="text-[8px] font-bold tracking-wide text-accent-red">{item.date.month}</span>
              <span className="text-[13px] font-bold">{item.date.day}</span>
            </div>
            <div className="flex-1 min-w-0 flex flex-col">
              <span className="text-xs font-semibold whitespace-nowrap overflow-hidden text-ellipsis">
                {item.title}
              </span>
              <span className="text-[10.5px] text-ink-muted">{item.meta}</span>
            </div>
            <span className={`tag-pill ${TAG_CLASS[item.tag]}`}>{item.tag}</span>
          </li>
        ))}
      </ul>

      <button className="self-start flex items-center gap-1.5 text-[11px] text-accent-blue">
        View Full Calendar <ArrowRight size={13} />
      </button>
    </section>
  );
}
