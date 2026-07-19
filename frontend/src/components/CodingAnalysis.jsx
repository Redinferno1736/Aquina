import { useEffect, useRef } from "react";
import anime from "animejs";
import { ChevronDown, Code2, ListChecks, Timer, Target, Flame, TrendingUp, TrendingDown } from "lucide-react";
import { CODING_SCORE } from "../data/mockData.js";

const METRIC_ICONS = {
  solved: ListChecks,
  avgTime: Timer,
  accuracy: Target,
  streak: Flame,
};

const DELTA_COLOR = {
  up: "text-accent-green",
  down: "text-accent-blue",
  flat: "text-ink-muted",
};

function ScoreRing({ score = 87 }) {
  const circleRef = useRef(null);
  const radius = 52;
  const circumference = 2 * Math.PI * radius;

  useEffect(() => {
    const offset = circumference - (score / 100) * circumference;
    anime({
      targets: circleRef.current,
      strokeDashoffset: [circumference, offset],
      duration: 1200,
      easing: "easeOutCubic",
    });
  }, [score, circumference]);

  return (
    <div className="relative w-[140px] h-[140px]">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <defs>
          <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#4fb2ff" />
            <stop offset="100%" stopColor="#ff5a5a" />
          </linearGradient>
        </defs>
        <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="10" />
        <circle
          ref={circleRef}
          cx="70"
          cy="70"
          r={radius}
          fill="none"
          stroke="url(#scoreGradient)"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference}
          transform="rotate(-90 70 70)"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-[34px] font-semibold leading-none">{score}</span>
        <span className="text-[11px] text-ink-muted mt-0.5">/100</span>
      </div>
    </div>
  );
}

function RadarChart({ data }) {
  const size = 220;
  const center = size / 2;
  const maxRadius = 78;
  const angleStep = (Math.PI * 2) / data.length;

  const points = data.map((d, i) => {
    const angle = -Math.PI / 2 + i * angleStep;
    const r = d.value * maxRadius;
    return [center + r * Math.cos(angle), center + r * Math.sin(angle)];
  });

  const polygonPoints = points.map((p) => p.join(",")).join(" ");
  const rings = [0.25, 0.5, 0.75, 1];

  return (
    <svg width={size+50} height={size} viewBox={`0 0 ${size} ${size}`} className="flex-none">
      {rings.map((ringR) => {
        const ringPoints = data
          .map((_, i) => {
            const angle = -Math.PI / 2 + i * angleStep;
            const r = ringR * maxRadius;
            return `${center + r * Math.cos(angle)},${center + r * Math.sin(angle)}`;
          })
          .join(" ");
        return (
          <polygon key={ringR} points={ringPoints} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
        );
      })}
      <polygon points={polygonPoints} fill="rgba(255,90,90,0.22)" stroke="#ff5a5a" strokeWidth="1.5" />
      {data.map((d, i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        const lx = center + (maxRadius + 24) * Math.cos(angle);
        const ly = center + (maxRadius + 24) * Math.sin(angle);
        return (
          <text
            key={d.axis}
            x={lx}
            y={ly}
            fill="#728296"
            fontSize="10"
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {d.axis}
          </text>
        );
      })}
    </svg>
  );
}

export default function CodingAnalysis() {
  const { score, label, metrics, radar } = CODING_SCORE;

  return (
    <section className="panel flex flex-col px-5 py-4">
      <header className="panel-header">
        <div className="panel-title">
          <Code2 size={15} /> Coding Analysis
        </div>
        <button className="dropdown-pill">
          This Week <ChevronDown size={13} />
        </button>
      </header>

      <div className="flex-1 flex items-center gap-3 ">
        {/* Score Ring */}
        <div className="flex flex-col items-center gap-2 flex-none">
          <ScoreRing score={score} />
          <span className="text-xs text-ink-secondary">Coding Score</span>
          <span className="text-[11px] font-semibold text-accent-green bg-accent-green/[0.14] rounded-full px-2.5 py-[3px]">
            {label}
          </span>
        </div>

        {/* Metrics Card */}
        <div className="flex-1 max-w-[220px] rounded-xl border border-panel-border bg-white/[0.02] px-3.5 py-1">
          <ul className="flex flex-col list-none m-0 p-0">
            {metrics.map((m, i) => {
              const Icon = METRIC_ICONS[m.key];
              const TrendIcon = m.trend === "up" ? TrendingUp : m.trend === "down" ? TrendingDown : null;
              return (
                <li
                  key={m.key}
                  className={`flex items-center justify-between py-2.5 ${
                    i !== metrics.length - 1 ? "border-b border-panel-border" : ""
                  }`}
                >
                  <div className="flex items-center gap-3.5">
                    <span className="w-[26px] h-[26px] rounded-lg flex items-center justify-center bg-accent-blueSoft text-accent-blue flex-none">
                      <Icon size={14} />
                    </span>

                    <div className="flex flex-col leading-tight">
                      <span className="text-[11px] text-ink-secondary">
                        {m.label}
                      </span>
                      <span className="font-semibold font-mono text-[14px]">
                        {m.value}
                      </span>
                    </div>
                  </div>

                  <span
                    className={`flex items-center gap-1 text-[11px] font-semibold flex-none ${DELTA_COLOR[m.trend]}`}
                  >
                    {TrendIcon && <TrendIcon size={12} />}
                    {m.delta}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Radar Chart */}
        <div className="flex-none flex items-center justify-center">
          <RadarChart data={radar} />
        </div>
      </div>

    </section>
  );
}
