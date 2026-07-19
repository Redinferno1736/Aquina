// Static mock data. There is no backend wired up yet — swap these
// for real Tauri commands / activity-tracker events later.

export const PLATFORM_COLORS = {
  github: "#3fd67a",
  leetcode: "#ffc94d",
  codeforces: "#4aa3ff",
  codechef: "#c96a35",
};

export const PLATFORM_LABELS = {
  github: "GitHub",
  leetcode: "LeetCode",
  codeforces: "Codeforces",
  codechef: "CodeChef",
};

// Deterministic pseudo-random generator so the heatmap looks the
// same on every reload instead of reshuffling.
function seeded(seed) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

export function generateHeatmapData(weeks = 53) {
  const rand = seeded(42);
  const days = [];
  const today = new Date();

  for (let i = weeks * 7 - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);

    const activity = {};
    let total = 0;
    Object.keys(PLATFORM_COLORS).forEach((platform) => {
      const chance = rand();
      if (chance > (platform === "github" ? 0.55 : 0.78)) {
        const minutes = Math.round(10 + rand() * 90);
        activity[platform] = minutes;
        total += minutes;
      }
    });

    days.push({ date: date.toISOString().slice(0, 10), activity, total });
  }
  return days;
}

export const CODING_SCORE = {
  score: 87,
  label: "Excellent",
  trend: "up",
  metrics: [
    { key: "solved", label: "Problems Solved", value: "48", delta: "+20%", trend: "up" },
    { key: "avgTime", label: "Avg. Time / Problem", value: "24m 18s", delta: "-12%", trend: "down" },
    { key: "accuracy", label: "Accuracy", value: "92%", delta: "+8%", trend: "up" },
    { key: "streak", label: "Consistency", value: "7 Days", delta: "streak", trend: "flat" },
  ],
  radar: [
    { axis: "Logic", value: 0.86 },
    { axis: "Problem Solving", value: 0.72 },
    { axis: "Speed", value: 0.64 },
    { axis: "Accuracy", value: 0.9 },
    { axis: "Consistency", value: 0.8 },
  ],
};

export const UPCOMING = [
  { date: { month: "JUL", day: "15" }, title: "Codeforces Round #962", meta: "In 2 hours", tag: "Contest" },
  { date: { month: "JUL", day: "16" }, title: "System Design Study", meta: "8:00 PM – 9:30 PM", tag: "Study" },
  { date: { month: "JUL", day: "18" }, title: "Project Deadline", meta: "2 days left", tag: "Deadline" },
];

export const QUICK_ACTIONS = [
  { key: "code", label: "Code" },
  { key: "projects", label: "Projects" },
  { key: "assistant", label: "AI Assistant" },
  { key: "notes", label: "Notes" },
  { key: "terminal", label: "Terminal" },
];

export const NOW_PLAYING = {
  title: "Nightshift focus...",
  subtitle: "Aquina Focus Radio",
  progress: 0.35,
};
