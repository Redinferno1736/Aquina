use anyhow::Result;
use chrono::TimeZone;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::models::{ActivityRecord, Platform};
use super::storage::Storage;

const LC_GRAPHQL: &str = "https://leetcode.com/graphql";

#[derive(Debug, Serialize)]
struct GraphQLQuery {
    query: String,
    variables: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct LcResponse {
    data: LcData,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct LcData {
    #[serde(rename = "matchedUser")]
    matched_user: MatchedUser,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MatchedUser {
    #[serde(rename = "submitStatsGlobal")]
    submit_stats_global: SubmitStats,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SubmitStats {
    #[serde(rename = "acSubmissionNum")]
    ac_submission_num: Vec<AcCount>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AcCount {
    difficulty: String,
    count: i64,
}

/// Public GraphQL endpoint, no auth required.
/// Note: LeetCode's public API gives aggregate counts, not per-day timestamps
/// for submissions in one simple call — the submission calendar query is used
/// separately for date-level granularity.
pub async fn fetch_submission_calendar(username: &str) -> Result<std::collections::HashMap<String, f64>> {
    let query = GraphQLQuery {
        query: r#"
            query userProfileCalendar($username: String!) {
                matchedUser(username: $username) {
                    userCalendar {
                        submissionCalendar
                    }
                }
            }
        "#
        .to_string(),
        variables: json!({ "username": username }),
    };

    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .post(LC_GRAPHQL)
        .json(&query)
        .send()
        .await?
        .json()
        .await?;

    let calendar_str = resp["data"]["matchedUser"]["userCalendar"]["submissionCalendar"]
        .as_str()
        .unwrap_or("{}");

    // submissionCalendar is a JSON string: { "<unix_ts>": count, ... }
    let raw: std::collections::HashMap<String, i64> = serde_json::from_str(calendar_str)?;

    let mut by_date: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (ts_str, count) in raw {
        if let Ok(ts) = ts_str.parse::<i64>() {
            if let Some(dt) = chrono::Utc.timestamp_opt(ts, 0).single() {
                let date = dt.format("%Y-%m-%d").to_string();
                *by_date.entry(date).or_insert(0.0) += count as f64;
            }
        }
    }

    Ok(by_date)
}

pub fn normalize(by_date: std::collections::HashMap<String, f64>) -> Vec<ActivityRecord> {
    by_date
        .into_iter()
        .map(|(date, value)| ActivityRecord {
            date,
            platform: Platform::LeetCode.as_str().to_string(),
            metric_type: "submission".to_string(),
            value,
        })
        .collect()
}

pub async fn store(storage: &Storage, username: &str) -> Result<()> {
    let by_date = fetch_submission_calendar(username).await?;
    let records = normalize(by_date);
    storage.store(&records)?;
    Ok(())
}