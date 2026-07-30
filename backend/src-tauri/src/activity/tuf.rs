use anyhow::{anyhow, Result};

use super::auth::SessionManager;
use super::models::{ActivityRecord, Platform};
use super::storage::Storage;

const TUF_LOGIN_URL: &str = "https://takeuforward.org/login";
const TUF_PROFILE_ENDPOINT: &str = "https://takeuforward.org/api/v1/profile/activity"; // placeholder — adjust to real internal endpoint once inspected

/// TUF is a fully independent platform — its calendar is NOT derived from
/// LeetCode, and there's no public API, so this goes through authenticated
/// scraping using the shared session/auth infrastructure.
pub async fn fetch(sessions: &SessionManager) -> Result<serde_json::Value> {
    let cookies = sessions
        .get(Platform::Tuf.as_str())
        .ok_or_else(|| anyhow!("no TUF session — run authenticate_via_webview first"))?;

    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .get(TUF_PROFILE_ENDPOINT)
        .header("Cookie", cookies)
        .send()
        .await?
        .json()
        .await?;

    Ok(resp)
}

/// Parser is isolated here — swapping in the real response shape only
/// touches this function, per the "new parser, new fetcher only" rule.
pub fn normalize(raw: &serde_json::Value) -> Vec<ActivityRecord> {
    let mut records = Vec::new();

    if let Some(entries) = raw.get("activity").and_then(|v| v.as_array()) {
        for entry in entries {
            let date = entry.get("date").and_then(|v| v.as_str());
            let count = entry.get("count").and_then(|v| v.as_f64());
            if let (Some(date), Some(count)) = (date, count) {
                records.push(ActivityRecord {
                    date: date.to_string(),
                    platform: Platform::Tuf.as_str().to_string(),
                    metric_type: "submission".to_string(),
                    value: count,
                });
            }
        }
    }

    records
}

pub async fn store(storage: &Storage, sessions: &SessionManager) -> Result<()> {
    let raw = fetch(sessions).await?;
    let records = normalize(&raw);
    storage.store(&records)?;
    Ok(())
}

pub fn login_url() -> &'static str {
    TUF_LOGIN_URL
}