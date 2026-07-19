import { useEffect, useMemo, useRef, useState } from "react";
import anime from "animejs";
import { ChevronDown, Code2 } from "lucide-react";
import { generateHeatmapData, PLATFORM_COLORS, PLATFORM_LABELS } from "../data/mockData.js";

const DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// Mixes N platform hex colors, weighted by minutes spent on each —
// this is the "real color-mixing, not a split-cell hack" the brief
// asks for. Simple additive RGB blend, weighted by share of the day.
function blendColors(activity) {
  const entries = Object.entries(activity);
  if (entries.length === 0) return null;
  const total = entries.reduce((sum, [, mins]) => sum + mins, 0);

  let r = 0, g = 0, b = 0;
  entries.forEach(([platform, mins]) => {
    const hex = PLATFORM_COLORS[platform];
    const weight = mins / total;
    r += parseInt(hex.slice(1, 3), 16) * weight;
    g += parseInt(hex.slice(3, 5), 16) * weight;
    b += parseInt(hex.slice(5, 7), 16) * weight;
  });
  return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
}

function intensityFor(total) {
  if (total === 0) return 0;
  if (total < 30) return 0.35;
  if (total < 70) return 0.6;
  if (total < 130) return 0.8;
  return 1;
}

// Groups days by real calendar month, then into Mon-Sun week columns
// inside that month (padded with nulls so columns stay 7 cells tall).
// Returns months in chronological order (oldest -> newest).
function groupIntoMonths(days) {
  if (days.length === 0) return [];

  const byMonthKey = new Map();
  days.forEach((day) => {
    const d = new Date(day.date);
    const key = `${d.getFullYear()}-${d.getMonth()}`;
    if (!byMonthKey.has(key)) {
      byMonthKey.set(key, { year: d.getFullYear(), month: d.getMonth(), days: [] });
    }
    byMonthKey.get(key).days.push(day);
  });

  const months = Array.from(byMonthKey.values()).sort((a, b) =>
    a.year !== b.year ? a.year - b.year : a.month - b.month
  );

  return months.map(({ year, month, days: monthDays }) => {
    const byDate = new Map(monthDays.map((d) => [d.date, d]));
    const daysInMonth = new Date(year, month + 1, 0).getDate();

    // Mon = 0 ... Sun = 6
    const dow = (dateObj) => (dateObj.getDay() + 6) % 7;

    const firstDow = dow(new Date(year, month, 1));
    const totalCells = firstDow + daysInMonth;
    const weekCount = Math.ceil(totalCells / 7);

    const weeks = Array.from({ length: weekCount }, () => new Array(7).fill(null));

    for (let dayNum = 1; dayNum <= daysInMonth; dayNum++) {
      const cellIndex = firstDow + (dayNum - 1);
      const weekIndex = Math.floor(cellIndex / 7);
      const weekday = cellIndex % 7;
      const dateObj = new Date(year, month, dayNum);
      const iso = dateObj.toISOString().slice(0, 10);
      weeks[weekIndex][weekday] = byDate.get(iso) ?? { date: iso, activity: {}, total: 0 };
    }

    return {
      key: `${year}-${month}`,
      label: MONTH_NAMES[month],
      year,
      weeks,
    };
  });
}

export default function CodingHeatmap() {
  const days = useMemo(() => generateHeatmapData(53), []);
  const months = useMemo(() => groupIntoMonths(days), [days]);
  const gridRef = useRef(null);
  const [active, setActive] = useState(null);

  useEffect(() => {
    const cells = gridRef.current?.querySelectorAll(".heat-cell") ?? [];
    anime({
      targets: cells,
      opacity: [0, 1],
      scale: [0.4, 1],
      delay: anime.stagger(2.2, { from: "first" }),
      duration: 500,
      easing: "easeOutQuad",
    });
  }, []);

  return (
    <section className="panel relative flex flex-col px-5 pt-4 pb-3">
      <div className="flex justify-between gap-4">
        <div className="flex-1 min-w-0 overflow-x-auto">
          <div className="flex gap-1.5">
            <div className="flex flex-col justify-between text-[8px] text-ink-faint w-6 py-px flex-none pt-[15.7px]">
              {DAY_LABELS.map((d) => (
                <span key={d}>{d}</span>
              ))}
            </div>
            <div className="flex gap-2 flex-1" ref={gridRef}>
              {months.map((month) => (
                <div className="flex flex-col items-center gap-1" key={month.key}>
                  <span className="text-[10px] text-ink-muted whitespace-nowrap">{month.label}</span>
                  <div className="flex gap-[3px]">
                    {month.weeks.map((week, wi) => (
                      <div className="flex flex-col gap-[3px]" key={wi}>
                        {week.map((day, di) => {
                          if (!day) {
                            return <div className="heat-cell w-[8.5px] h-[8.5px] rounded-[2.5px] bg-transparent" key={di} />;
                          }
                          const color = blendColors(day.activity);
                          const intensity = intensityFor(day.total);
                          return (
                            <div
                              key={day.date}
                              className="heat-cell w-[8.5px] h-[8.5px] rounded-[2.5px] transition-transform duration-150
                                hover:scale-[1.35] hover:outline hover:outline-1 hover:outline-white/50"
                              style={{
                                background: color ?? "rgba(255,255,255,0.05)",
                                opacity: color ? 0.25 + intensity * 0.75 : 1,
                              }}
                              onMouseEnter={() => setActive(day)}
                              onMouseLeave={() => setActive(null)}
                            />
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="flex gap-4 pt-1.5 px-3 text-[11px] text-ink-secondary">
        <span className="flex items-center gap-1.5">
          <span className="live-dot" /> Currently: <strong>Coding (VS Code)</strong>
        </span>
        <span className="flex items-center gap-1.5 text-ink-muted">
          <Code2 size={12} /> Session: 02h 34m
        </span>
      </div>
        </div>

        <div className="flex-none w-[108px] flex flex-col gap-[7px] justify-center border-l border-panel-border pl-4">
          <button className="dropdown-pill">
          This Year <ChevronDown size={13} />
        </button>
          {Object.entries(PLATFORM_LABELS).map(([key, label]) => (
            <div className="flex items-center px-2.5 gap-1.5 text-[11px] text-ink-secondary" key={key}>
              <span className="legend-dot" style={{ background: PLATFORM_COLORS[key] }} />
              {label}
            </div>
          ))}
        </div>
      </div>

      {active && (
        <div className="panel absolute bottom-[46px] left-5 z-10 px-3 py-2 text-[11px]">
          <strong>
            {new Date(active.date).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
          </strong>
          {Object.keys(active.activity).length === 0 && <span className="text-ink-muted"> · No activity</span>}
          <div className="flex gap-2.5 mt-1 text-ink-secondary">
            {Object.entries(active.activity).map(([platform, mins]) => (
              <span key={platform} className="flex items-center gap-1.5">
                <span className="legend-dot" style={{ background: PLATFORM_COLORS[platform] }} />
                {PLATFORM_LABELS[platform]}: {mins}m
              </span>
            ))}
          </div>
        </div>
      )}

      
    </section>
  );
}