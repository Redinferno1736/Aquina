import { useEffect, useState } from "react";

function formatTime(date) {
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  }).toUpperCase();
}

export default function ClockPanel() {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  console.log("ClockPanel rendered");
  return (
    <section
  className="flex-none w-[197px] flex flex-col items-end justify-center py-10 relative z-50"
>
      <div className="font-mono text-[25px] font-semibold tracking-wide text-ink-primary">
        {formatTime(now)}
      </div>
      <div className="font-mono text-xs text-[#805c57] mt-1">
        {`${now.toLocaleDateString(undefined, { weekday: "long" })}, ${now.toLocaleDateString(undefined, {
  month: "short",
  day: "numeric",
})}`}
      </div>
    </section>
  );
}
