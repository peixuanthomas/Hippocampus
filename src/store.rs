use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use uuid::Uuid;

use crate::model::{BudgetConfig, DEFAULT_SYSTEM_PROMPT, Session, SessionStatus};

#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let expanded = expand_tilde(root.as_ref());
        let path = expanded.as_path();
        let root = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn create(
        &self,
        model: &str,
        ollama_host: &str,
        system_prompt: Option<&str>,
        budget: BudgetConfig,
        think: bool,
    ) -> Result<Session> {
        let id = format!(
            "{}-{}",
            Utc::now().format("%Y%m%d-%H%M%S"),
            &Uuid::new_v4().simple().to_string()[..8]
        );
        let mut session = Session::new(
            id,
            model.to_owned(),
            ollama_host.trim_end_matches('/').to_owned(),
            system_prompt.unwrap_or(DEFAULT_SYSTEM_PROMPT).to_owned(),
            budget,
            think,
        )?;
        self.save(&mut session)?;
        Ok(session)
    }

    pub fn save(&self, session: &mut Session) -> Result<PathBuf> {
        validate_identifier(&session.id)?;
        session.validate()?;
        session.touch();
        session.refresh_cumulative_usage();
        fs::create_dir_all(&self.root)
            .with_context(|| format!("无法创建会话目录 {}", self.root.display()))?;
        let target = self.root.join(format!("{}.json", session.id));
        let temporary = self
            .root
            .join(format!(".{}.{}.tmp", session.id, Uuid::new_v4().simple()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            serde_json::to_writer_pretty(&mut file, session)?;
            file.write_all(b"\n")?;
            file.flush()?;
            file.sync_all()?;
            fs::rename(&temporary, &target)?;
            sync_directory(&self.root)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result.with_context(|| format!("无法原子保存会话 {}", target.display()))?;
        Ok(target)
    }

    pub fn load(&self, identifier: &str) -> Result<Session> {
        let path = self.resolve(identifier)?;
        let raw =
            fs::read(&path).with_context(|| format!("无法读取会话文件 {}", path.display()))?;
        let mut session: Session = serde_json::from_slice(&raw)
            .map_err(|error| anyhow!("会话文件损坏或格式不受支持: {}: {error}", path.display()))?;
        session
            .validate()
            .map_err(|error| anyhow!("会话文件损坏或格式不受支持: {}: {error}", path.display()))?;
        session.refresh_cumulative_usage();
        Ok(session)
    }

    pub fn resolve(&self, identifier: &str) -> Result<PathBuf> {
        validate_identifier(identifier).map_err(|_| anyhow!("无效会话标识: {identifier:?}"))?;
        let exact = self.root.join(format!("{identifier}.json"));
        if exact.is_file() {
            return Ok(exact);
        }
        let mut matches = Vec::new();
        if self.root.is_dir() {
            for entry in fs::read_dir(&self.root)? {
                let path = entry?.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if name.starts_with(identifier) && name.ends_with(".json") {
                    matches.push(path);
                }
            }
        }
        matches.sort();
        match matches.len() {
            0 => bail!("找不到会话: {identifier}"),
            1 => Ok(matches.remove(0)),
            _ => {
                let names = matches
                    .iter()
                    .take(5)
                    .filter_map(|path| path.file_stem()?.to_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!("会话前缀不唯一: {identifier}（匹配 {names}）")
            }
        }
    }

    pub fn list_sessions(&self) -> Result<Vec<Session>> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) == Some("json")
                && let Some(stem) = path.file_stem().and_then(|value| value.to_str())
            {
                sessions.push(self.load(stem)?);
            }
        }
        sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        Ok(sessions)
    }

    pub fn reopen(&self, session: &mut Session) -> Result<()> {
        session.status = SessionStatus::Active;
        self.save(session)?;
        Ok(())
    }
}

fn expand_tilde(path: &Path) -> PathBuf {
    let Some(value) = path.to_str() else {
        return path.to_path_buf();
    };
    if value == "~" {
        return std::env::var_os("HOME").map_or_else(|| path.to_path_buf(), PathBuf::from);
    }
    if let Some(remainder) = value.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(remainder);
    }
    path.to_path_buf()
}

fn validate_identifier(identifier: &str) -> Result<()> {
    let mut chars = identifier.chars();
    let Some(first) = chars.next() else {
        bail!("unsafe session id");
    };
    if !first.is_ascii_alphanumeric()
        || !chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
    {
        bail!("unsafe session id: {identifier:?}");
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ContextTrace, TokenUsage, Turn, TurnStatus, utc_now};

    #[test]
    fn round_trip_legacy_schema_and_atomic_file() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create(
                "model",
                "http://localhost:11434",
                None,
                BudgetConfig::default(),
                true,
            )
            .unwrap();
        let now = utc_now();
        session.turns.push(Turn {
            id: "turn".into(),
            created_at: now.clone(),
            updated_at: now,
            status: TurnStatus::Complete,
            user_content: "hello".into(),
            assistant_content: "world".into(),
            thinking: "private".into(),
            usage: TokenUsage::new(Some(12), Some(4)),
            probe_usage: TokenUsage::new(Some(12), Some(1)),
            context_trace: ContextTrace::default(),
            done_reason: Some("stop".into()),
            error: None,
        });
        session.active_context_start_index = 1;
        let path = store.save(&mut session).unwrap();
        let restored = store.load(&session.id[..12]).unwrap();
        assert_eq!(restored.turns[0].thinking, "private");
        assert_eq!(restored.cumulative_usage.total_tokens, Some(16));
        assert!(fs::read_to_string(path).unwrap().ends_with('\n'));
        assert!(fs::read_dir(root.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
    }

    #[test]
    fn corrupt_file_is_not_overwritten() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let path = root.path().join("broken.json");
        fs::write(&path, "{broken").unwrap();
        assert!(store.load("broken").is_err());
        assert_eq!(fs::read_to_string(path).unwrap(), "{broken");
    }

    #[test]
    fn loads_schema_v1_json_written_by_python_version() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let legacy = r#"{
  "schema_version": 1,
  "id": "20260811-abcdef12",
  "title": "旧会话",
  "created_at": "2026-08-11T01:02:03.123456+00:00",
  "updated_at": "2026-08-11T01:03:04.123456+00:00",
  "status": "active",
  "model": "qwen3.5:9b",
  "ollama_host": "http://127.0.0.1:11434",
  "system_prompt": "系统",
  "think": true,
  "budget": {
    "context_window": 32768,
    "max_output_tokens": 4096,
    "safety_margin_tokens": 512,
    "probe_ratio": 0.8,
    "warning_ratio": 0.9,
    "trim_target_ratio": 0.8
  },
  "active_context_start_index": 0,
  "turns": [{
    "id": "turn-1",
    "created_at": "2026-08-11T01:02:03.123456+00:00",
    "updated_at": "2026-08-11T01:02:04.123456+00:00",
    "status": "complete",
    "user_content": "你好",
    "assistant_content": "世界",
    "thinking": "不会回注",
    "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
    "probe_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    "context_trace": {
      "included_turn_ids": [], "omitted_turn_ids": [],
      "estimated_upper_tokens": 100, "exact_input_tokens": 12,
      "input_budget": 28160, "decision": "ready",
      "active_context_start_before": 0, "active_context_start_after": 0
    },
    "done_reason": "stop", "error": null
  }],
  "cumulative_usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
  "cumulative_probe_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
}"#;
        fs::write(root.path().join("20260811-abcdef12.json"), legacy).unwrap();
        let mut session = store.load("20260811-abc").unwrap();
        assert_eq!(session.title, "旧会话");
        assert_eq!(session.turns[0].thinking, "不会回注");
        assert_eq!(session.cumulative_usage.total_tokens, Some(16));
        store.save(&mut session).unwrap();
        assert!(store.load(&session.id).is_ok());
    }

    #[test]
    fn identifier_validation_blocks_path_traversal() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        assert!(store.resolve("../secret").is_err());
        assert!(store.resolve("/absolute").is_err());
    }
}
