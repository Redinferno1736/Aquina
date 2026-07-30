use anyhow::Result;
use std::collections::HashMap;

use super::models::{DaySummary, PlatformDayStat};
use super::storage::Storage;

/// Aggregation only ever reads normalized data. No platform-specific logic here.
pub fn get_heatmap_data(storage: &Storage, start_date: &str, end_date: &str) -> Result<Vec<DaySummary>> {
    let records = storage.query_range(start_date, end_date)?;

    // date -> platform -> Vec<(metric_type, value)>
    let mut by_day: HashMap<String, HashMap<String, Vec<(String, f64)>>> = HashMap::new();

    for r in records {
        by_day
            .entry(r.date.clone())
            .or_default()
            .entry(r.platform.clone())
            .or_default()
            .push((r.metric_type.clone(), r.value));
    }

    let mut result: Vec<DaySummary> = by_day
        .into_iter()
        .map(|(date, platforms)| {
            let platform_stats = platforms
                .into_iter()
                .map(|(platform, metrics)| {
                    let total_value: f64 = metrics.iter().map(|(_, v)| v).sum();
                    PlatformDayStat {
                        platform,
                        total_value,
                        metrics,
                    }
                })
                .collect();
            DaySummary { date, platforms: platform_stats }
        })
        .collect();

    result.sort_by(|a, b| a.date.cmp(&b.date));
    Ok(result)
}