use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Platform {
    GitHub,
    NeetCode,
    LeetCode,
    Codeforces,
    Tuf,
    CodeChef,   // future
    HackerRank, // future
}

impl Platform {
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::GitHub => "github",
            Platform::NeetCode => "neetcode",
            Platform::LeetCode => "leetcode",
            Platform::Codeforces => "codeforces",
            Platform::Tuf => "tuf",
            Platform::CodeChef => "codechef",
            Platform::HackerRank => "hackerrank",
        }
    }

    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "github" => Some(Platform::GitHub),
            "neetcode" => Some(Platform::NeetCode),
            "leetcode" => Some(Platform::LeetCode),
            "codeforces" => Some(Platform::Codeforces),
            "tuf" => Some(Platform::Tuf),
            "codechef" => Some(Platform::CodeChef),
            "hackerrank" => Some(Platform::HackerRank),
            _ => None,
        }
    }
}

/// The single normalized shape every platform writes into.
/// date + platform + metric_type + value — nothing platform-specific here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityRecord {
    pub date: String,        // YYYY-MM-DD
    pub platform: String,    // Platform::as_str()
    pub metric_type: String, // e.g. "commit", "submission", "minutes_active"
    pub value: f64,
}

/// One day's blended-across-platforms summary, used by the dashboard heatmap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaySummary {
    pub date: String,
    pub platforms: Vec<PlatformDayStat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformDayStat {
    pub platform: String,
    pub total_value: f64,
    pub metrics: Vec<(String, f64)>,
}