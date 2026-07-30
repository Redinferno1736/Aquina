use anyhow::Result;
use rusqlite::{params, Connection};
use std::sync::Mutex;

use super::models::ActivityRecord;

pub struct Storage {
    conn: Mutex<Connection>,
}

impl Storage {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let storage = Storage {
            conn: Mutex::new(conn),
        };
        storage.init_schema()?;
        Ok(storage)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        // Single normalized table. No platform-specific tables, ever.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                platform TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                UNIQUE(date, platform, metric_type)
            )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_date ON activity(date)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_platform ON activity(platform)",
            [],
        )?;
        Ok(())
    }

    /// Every fetcher calls this same method. No fetcher writes raw SQL.
    pub fn store(&self, records: &[ActivityRecord]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        for r in records {
            conn.execute(
                "INSERT INTO activity (date, platform, metric_type, value)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(date, platform, metric_type)
                 DO UPDATE SET value = excluded.value",
                params![r.date, r.platform, r.metric_type, r.value],
            )?;
        }
        Ok(())
    }

    pub fn query_range(&self, start_date: &str, end_date: &str) -> Result<Vec<ActivityRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT date, platform, metric_type, value FROM activity
             WHERE date BETWEEN ?1 AND ?2
             ORDER BY date ASC",
        )?;
        let rows = stmt.query_map(params![start_date, end_date], |row| {
            Ok(ActivityRecord {
                date: row.get(0)?,
                platform: row.get(1)?,
                metric_type: row.get(2)?,
                value: row.get(3)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }
}