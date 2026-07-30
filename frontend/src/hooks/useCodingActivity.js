import { useEffect, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { CREDENTIALS } from "../config/credentials.js";

// Maps backend platform strings -> the shape CodingHeatmap expects:
// { date, activity: { platform: minutesOrCount }, total }
function toHeatmapDays(daySummaries) {
  return daySummaries.map((day) => {
    const activity = {};
    let total = 0;
    for (const p of day.platforms) {
      activity[p.platform] = p.total_value;
      total += p.total_value;
    }
    return { date: day.date, activity, total };
  });
}

export function useCodingActivity() {
  const [days, setDays] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [syncing, setSyncing] = useState(false);

  const loadHeatmap = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const today = new Date();
      const start = new Date(today);
      start.setDate(start.getDate() - 371); // ~53 weeks back, matches old mock range

      const startDate = start.toISOString().slice(0, 10);
      const endDate = today.toISOString().slice(0, 10);

      const result = await invoke("get_heatmap", { startDate, endDate });
      setDays(toHeatmapDays(result));
    } catch (err) {
      console.error("get_heatmap failed:", err);
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const syncAll = useCallback(async () => {
    setSyncing(true);
    setError(null);
    const results = { github: null, codeforces: null, leetcode: null };

    try {
      await invoke("sync_github", {
        token: CREDENTIALS.github.token,
        username: CREDENTIALS.github.username,
      });
      results.github = "ok";
    } catch (err) {
      results.github = String(err);
      console.error("sync_github failed:", err);
    }

    try {
      await invoke("sync_codeforces", { handle: CREDENTIALS.codeforces.handle });
      results.codeforces = "ok";
    } catch (err) {
      results.codeforces = String(err);
      console.error("sync_codeforces failed:", err);
    }

    try {
      await invoke("sync_leetcode", { username: CREDENTIALS.leetcode.username });
      results.leetcode = "ok";
    } catch (err) {
      results.leetcode = String(err);
      console.error("sync_leetcode failed:", err);
    }

    setSyncing(false);
    await loadHeatmap(); // refresh after sync
    return results;
  }, [loadHeatmap]);

  useEffect(() => {
    loadHeatmap();
  }, [loadHeatmap]);

  return { days, loading, error, syncing, syncAll, reload: loadHeatmap };
}