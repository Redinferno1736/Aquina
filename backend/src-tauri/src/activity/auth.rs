use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, WebviewUrl, WebviewWindowBuilder};

/// One reusable session store for every login-gated platform.
/// TUF, and future CodeChef/HackerRank, all go through this — never
/// build a platform-specific login system.
#[derive(Default)]
pub struct SessionManager {
    // platform -> cookie string
    sessions: Mutex<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct StoredSession {
    pub platform: String,
    pub cookies: String,
}

impl SessionManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, platform: &str) -> Option<String> {
        self.sessions.lock().unwrap().get(platform).cloned()
    }

    pub fn set(&self, platform: &str, cookies: String) {
        self.sessions
            .lock()
            .unwrap()
            .insert(platform.to_string(), cookies);
    }
}

/// Opens a Tauri WebView for the given platform's login page, waits for the
/// user to authenticate normally, then captures the session cookies.
/// Never prompts for a password directly — reuses the platform's real login UI.
pub async fn authenticate_via_webview(
    app: &AppHandle,
    platform: &str,
    login_url: &str,
) -> Result<String> {
    let label = format!("auth-{}", platform);

    let window = WebviewWindowBuilder::new(app, &label, WebviewUrl::External(login_url.parse()?))
        .title(format!("Sign in to {}", platform))
        .inner_size(480.0, 720.0)
        .build()
        .map_err(|e| anyhow!("failed to open auth webview: {e}"))?;

    // In practice: listen for navigation to a post-login URL, then pull
    // cookies via the webview's cookie store (platform-specific detection
    // of "login succeeded" lives in each platform's own auth config, not here).
    // This is the shared skeleton every login-gated platform reuses.
    let cookies = window
        .cookies()
        .map_err(|e| anyhow!("failed to read cookies: {e}"))?
        .iter()
        .map(|c| format!("{}={}", c.name(), c.value()))
        .collect::<Vec<_>>()
        .join("; ");

    window.close().ok();

    if cookies.is_empty() {
        return Err(anyhow!("no session cookies captured for {platform}"));
    }

    Ok(cookies)
}