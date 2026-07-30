use activity::auth::SessionManager;
use activity::storage::Storage;
use std::sync::Arc;
use tauri::Manager;

mod activity;

#[tauri::command]
async fn sync_github(
    state: tauri::State<'_, Arc<Storage>>,
    token: String,
    username: String,
) -> Result<(), String> {
    activity::github::store(&state, &token, &username)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn sync_codeforces(
    state: tauri::State<'_, Arc<Storage>>,
    handle: String,
) -> Result<(), String> {
    activity::codeforces::store(&state, &handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn sync_leetcode(
    state: tauri::State<'_, Arc<Storage>>,
    username: String,
) -> Result<(), String> {
    activity::leetcode::store(&state, &username)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn authenticate_tuf(
    app: tauri::AppHandle,
    sessions: tauri::State<'_, Arc<SessionManager>>,
) -> Result<(), String> {
    let cookies = activity::auth::authenticate_via_webview(&app, "tuf", activity::tuf::login_url())
        .await
        .map_err(|e| e.to_string())?;
    sessions.set("tuf", cookies);
    Ok(())
}

#[tauri::command]
async fn sync_tuf(
    state: tauri::State<'_, Arc<Storage>>,
    sessions: tauri::State<'_, Arc<SessionManager>>,
) -> Result<(), String> {
    activity::tuf::store(&state, &sessions)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn get_heatmap(
    state: tauri::State<'_, Arc<Storage>>,
    start_date: String,
    end_date: String,
) -> Result<Vec<activity::models::DaySummary>, String> {
    activity::aggregate::get_heatmap_data(&state, &start_date, &end_date)
        .map_err(|e| e.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let db_path = app
                .path()
                .app_data_dir()
                .expect("no app data dir")
                .join("aquina.db");
            std::fs::create_dir_all(db_path.parent().unwrap()).ok();

            let storage = Arc::new(Storage::new(db_path.to_str().unwrap()).expect("db init failed"));
            let sessions = Arc::new(SessionManager::new());

            app.manage(storage);
            app.manage(sessions);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            sync_github,
            sync_codeforces,
            sync_leetcode,
            authenticate_tuf,
            sync_tuf,
            get_heatmap
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}