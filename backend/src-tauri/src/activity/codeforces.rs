use anyhow::Result;
use chrono::{TimeZone, Utc};
use serde::Deserialize;

use super::models::{ActivityRecord, Platform};
use super::storage::Storage;

const CF_API: &str = "https://codeforces.com/api";

#[derive(Debug, Deserialize)]
struct CfResponse {
    result: Vec<Submission>,
}

#[derive(Debug, Deserialize)]
pub struct Submission {
    #[serde(rename = "creationTimeSeconds")]
    creation_time_seconds: i64,
    verdict: Option<String>,
}

/// Public API, no auth required.
pub async fn fetch(handle: &str) -> Result<Vec<Submission>> {
    let url = format!("{CF_API}/user.status?handle={handle}&from=1&count=10000");
    let resp: CfResponse = reqwest::get(url).await?.json().await?;
    Ok(resp.result)
}

pub fn normalize(submissions: &[Submission]) -> Vec<ActivityRecord> {
    let mut counts: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for s in submissions {
        let is_accepted = s.verdict.as_deref() == Some("OK");
        if !is_accepted {
            continue;
        }
        if let Some(dt) = Utc.timestamp_opt(s.creation_time_seconds, 0).single() {
            let date = dt.format("%Y-%m-%d").to_string();
            *counts.entry(date).or_insert(0.0) += 1.0;
        }
    }

    counts
        .into_iter()
        .map(|(date, value)| ActivityRecord {
            date,
            platform: Platform::Codeforces.as_str().to_string(),
            metric_type: "submission".to_string(),
            value,
        })
        .collect()
}

pub async fn store(storage: &Storage, handle: &str) -> Result<()> {
    let submissions = fetch(handle).await?;
    let records = normalize(&submissions);
    storage.store(&records)?;
    Ok(())
}