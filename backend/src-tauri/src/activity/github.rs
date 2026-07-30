use anyhow::Result;
use chrono::Utc;
use serde::Deserialize;

use super::models::{ActivityRecord, Platform};
use super::storage::Storage;

const GITHUB_API: &str = "https://api.github.com";

#[derive(Debug, Deserialize)]
pub struct Repo {
    name: String,
    #[allow(dead_code)]
    pushed_at: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CommitEntry {
    commit: CommitDetail,
}

#[derive(Debug, Deserialize)]
pub struct CommitDetail {
    author: CommitAuthor,
}

#[derive(Debug, Deserialize)]
pub struct CommitAuthor {
    date: String, // ISO8601
}

/// NeetCode is not a separate fetcher. Repos matching NeetCode naming
/// conventions get filtered off into NeetCode activity; everything else
/// stays GitHub activity. This filtering logic lives entirely here.
fn is_neetcode_repo(repo_name: &str) -> bool {
    let lower = repo_name.to_lowercase();
    lower.contains("neetcode") || lower.contains("neet-code") || lower.contains("neet_code")
}

pub async fn fetch(token: &str, username: &str) -> Result<Vec<Repo>> {
    let client = reqwest::Client::new();
    let url = format!("{GITHUB_API}/users/{username}/repos?per_page=100");
    let repos: Vec<Repo> = client
        .get(url)
        .header("User-Agent", "aquina")
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .await?
        .json()
        .await?;
    Ok(repos)
}

pub async fn fetch_commits_for_repo(
    token: &str,
    owner: &str,
    repo_name: &str,
) -> Result<Vec<CommitEntry>> {
    let client = reqwest::Client::new();
    let url = format!("{GITHUB_API}/repos/{owner}/{repo_name}/commits?per_page=100");
    let commits: Vec<CommitEntry> = client
        .get(url)
        .header("User-Agent", "aquina")
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .await?
        .json()
        .await
        .unwrap_or_default();
    Ok(commits)
}

/// Converts raw commit data into the common ActivityRecord schema,
/// splitting into GitHub vs NeetCode based on repo name only.
pub fn normalize(repo_name: &str, commits: &[CommitEntry]) -> Vec<ActivityRecord> {
    let platform = if is_neetcode_repo(repo_name) {
        Platform::NeetCode
    } else {
        Platform::GitHub
    };

    let mut counts: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for c in commits {
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&c.commit.author.date) {
            let date = dt.format("%Y-%m-%d").to_string();
            *counts.entry(date).or_insert(0.0) += 1.0;
        }
    }

    counts
        .into_iter()
        .map(|(date, value)| ActivityRecord {
            date,
            platform: platform.as_str().to_string(),
            metric_type: "commit".to_string(),
            value,
        })
        .collect()
}

pub async fn store(storage: &Storage, token: &str, username: &str) -> Result<()> {
    let repos = fetch(token, username).await?;
    let mut all_records = Vec::new();

    for repo in repos {
        let commits = fetch_commits_for_repo(token, username, &repo.name).await?;
        let records = normalize(&repo.name, &commits);
        all_records.extend(records);
    }

    storage.store(&all_records)?;
    Ok(())
}

#[allow(dead_code)]
fn _unused_pushed_at_marker() {
    // pushed_at kept on Repo for future incremental-sync use; not wired yet.
    let _ = Utc::now();
}