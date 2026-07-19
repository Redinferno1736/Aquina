import { useEffect, useRef } from "react";
import anime from "animejs";

import CodingHeatmap from "./CodingHeatmap.jsx";
import ClockPanel from "./ClockPanel.jsx";
import BrandHeader from "./BrandHeader.jsx";
import QuickActions from "./QuickActions.jsx";
import CodingAnalysis from "./CodingAnalysis.jsx";
import UpcomingCard from "./UpcomingCard.jsx";
import QuickNote from "./QuickNote.jsx";
import NowPlaying from "./NowPlaying.jsx";
import AISearch from "./AISearch.jsx";

export default function Dashboard({ safeMode, onToggleSafeMode }) {
  const rootRef = useRef(null);

  useEffect(() => {
    // One orchestrated page-load sequence rather than scattered
    // per-widget effects: everything rises + fades in, staggered
    // top-to-bottom / left-to-right, per the frontend-design brief.
    anime({
      targets: rootRef.current.querySelectorAll(".panel, .brand-header"),
      opacity: [0, 1],
      translateY: [16, 0],
      delay: anime.stagger(70, { start: 80 }),
      duration: 650,
      easing: "easeOutQuint",
    });
  }, []);

  return (
    <div ref={rootRef} className="flex flex-col gap-[18px] max-w-[1500px] mx-auto px-7 pt-1 pb-8">
      {/* Row 1 — heatmap + live clock */}
      <div className="flex h-[150px]">
  <div className="w-[84%]">
    <CodingHeatmap />
  </div>

  <div className="w-[16%]">
    <ClockPanel />
  </div>
</div>

      {/* Row 2 — brand/quick-actions column + Coding Analysis */}
      <div className="flex gap-[18px] h-[50px] justify-between items-stretch min-h-[300px] max-[1180px]:flex-col">
        <div className="flex flex-col gap-3.5 flex-none w-[40%] max-[1180px]:w-full">
          <BrandHeader safeMode={safeMode} onToggleSafeMode={onToggleSafeMode} />
          <QuickActions />
        </div>
        <CodingAnalysis />
      </div>

      {/* Row 3 — upcoming/quote column + quick-note/now-playing + search */}
      <div className="flex gap-[18px] items-stretch max-[1180px]:flex-col">
        <div className="flex flex-row gap-3.5 flex-none w-[42%] max-[1180px]:w-full max-[1180px]:flex-col">
          <UpcomingCard />
          
        </div>
        <div className="flex flex-col gap-3.5 flex-1">
          <div className="flex gap-3.5">
            <QuickNote />
            <NowPlaying />
          </div>
          <AISearch />
        </div>
      </div>
    </div>
  );
}
