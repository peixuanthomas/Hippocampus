use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use thiserror::Error;
use uuid::Uuid;

use crate::knowledge::KnowledgeStore;
use crate::model::{
    BudgetConfig, DEFAULT_SYSTEM_PROMPT, SCHEMA_VERSION, Session, SessionStatus, TurnStatus,
};
use crate::retrieval::{RetrievalError, RetrievalStore};

#[derive(Debug, Error)]
#[error(
    "原始会话已安全保存到 {source_path}，但派生索引同步失败；请重试保存或调用 RetrievalStore::rebuild: {source}"
)]
pub struct IndexSyncAfterSourceCommit {
    pub source_path: PathBuf,
    #[source]
    pub source: RetrievalError,
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
    retrieval: RetrievalStore,
    knowledge: KnowledgeStore,
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
        let retrieval = RetrievalStore::new(&root)?;
        let knowledge = KnowledgeStore::new(&root)?;
        Ok(Self {
            root,
            retrieval,
            knowledge,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn retrieval(&self) -> &RetrievalStore {
        &self.retrieval
    }

    pub fn knowledge(&self) -> &KnowledgeStore {
        &self.knowledge
    }

    pub fn create(
        &self,
        model: &str,
        ollama_host: &str,
        system_prompt: Option<&str>,
        budget: BudgetConfig,
        think: bool,
    ) -> Result<Session> {
        self.create_named(model, ollama_host, "LLM", system_prompt, budget, think)
    }

    pub fn create_named(
        &self,
        model: &str,
        ollama_host: &str,
        ai_name: &str,
        system_prompt: Option<&str>,
        budget: BudgetConfig,
        think: bool,
    ) -> Result<Session> {
        let id = format!(
            "{}-{}",
            Utc::now().format("%Y%m%d-%H%M%S"),
            &Uuid::new_v4().simple().to_string()[..8]
        );
        let mut session = Session::new_named(
            id,
            model.to_owned(),
            ollama_host.trim_end_matches('/').to_owned(),
            ai_name.to_owned(),
            system_prompt.unwrap_or(DEFAULT_SYSTEM_PROMPT).to_owned(),
            budget,
            think,
        )?;
        self.save(&mut session)?;
        Ok(session)
    }

    pub fn save(&self, session: &mut Session) -> Result<PathBuf> {
        validate_identifier(&session.id)?;
        session.normalize_legacy_provenance();
        session.validate()?;
        for turn in &session.turns {
            self.knowledge
                .verify_trace(&turn.context_trace.knowledge)
                .with_context(|| format!("轮次 {} 的知识证据无法由原始快照重建", turn.id))?;
        }
        let _root_guard = self
            .retrieval
            .acquire_root_write()
            .context("无法锁定会话目录以原子保存原文和派生索引")?;
        fs::create_dir_all(&self.root)
            .with_context(|| format!("无法创建会话目录 {}", self.root.display()))?;
        let target = self.root.join(format!("{}.json", session.id));
        if target.is_file() {
            let previous = load_session_snapshot(&target)?;
            validate_append_only(&previous, session)?;
        }
        session.schema_version = SCHEMA_VERSION;
        session.touch();
        session.refresh_cumulative_usage();
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
        self.retrieval
            .sync_session_under_root_write(session, &target)
            .map_err(|source| IndexSyncAfterSourceCommit {
                source_path: target.clone(),
                source,
            })?;
        Ok(target)
    }

    pub fn load(&self, identifier: &str) -> Result<Session> {
        let path = self.resolve(identifier)?;
        let raw =
            fs::read(&path).with_context(|| format!("无法读取会话文件 {}", path.display()))?;
        let mut session: Session = serde_json::from_slice(&raw)
            .map_err(|error| anyhow!("会话文件损坏或格式不受支持: {}: {error}", path.display()))?;
        session.normalize_legacy_provenance();
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

fn load_session_snapshot(path: &Path) -> Result<Session> {
    let raw = fs::read(path).with_context(|| format!("无法读取已有会话文件 {}", path.display()))?;
    let session: Session = serde_json::from_slice(&raw)
        .map_err(|error| anyhow!("已有会话文件损坏，拒绝覆盖: {}: {error}", path.display()))?;
    session
        .validate()
        .map_err(|error| anyhow!("已有会话文件无效，拒绝覆盖: {}: {error}", path.display()))?;
    Ok(session)
}

fn validate_append_only(previous: &Session, next: &Session) -> Result<()> {
    if previous.id != next.id || previous.created_at != next.created_at {
        bail!("会话 ID 与创建时间不可修改");
    }
    if previous.system_prompt != next.system_prompt {
        bail!("已保存的 system prompt 属于原始事件，不可修改");
    }
    if previous.ai_name != next.ai_name {
        bail!("已保存的 AI 名称不可修改");
    }
    if next.turns.len() < previous.turns.len() {
        bail!("已保存的轮次不可删除");
    }
    for (index, (old, new)) in previous.turns.iter().zip(&next.turns).enumerate() {
        if old.id != new.id {
            bail!("第 {} 个已保存轮次的 ID 或顺序不可修改", index + 1);
        }
        if old.created_at != new.created_at {
            bail!("轮次 {} 的创建时间不可修改", old.id);
        }
        if old.user_content != new.user_content {
            bail!("轮次 {} 的用户原文不可修改", old.id);
        }
        if let Some(started_at) = &old.request_started_at
            && new.request_started_at.as_ref() != Some(started_at)
        {
            bail!("轮次 {} 的模型请求开始时间不可修改", old.id);
        }
        if old.status != TurnStatus::Pending {
            if old.status != new.status {
                bail!("终态轮次 {} 的状态不可修改", old.id);
            }
            if old.assistant_content != new.assistant_content || old.thinking != new.thinking {
                bail!("终态轮次 {} 的模型原文不可修改", old.id);
            }
            if old.request_started_at != new.request_started_at {
                bail!("终态轮次 {} 的模型请求来源不可修改", old.id);
            }
            if !same_authoritative_usage(old.usage, new.usage)
                || !same_authoritative_usage(old.probe_usage, new.probe_usage)
                || old.context_trace != new.context_trace
                || old.done_reason != new.done_reason
                || old.error != new.error
            {
                bail!("终态轮次 {} 的审计记录不可修改", old.id);
            }
        } else if (!new.assistant_content.is_empty() || !new.thinking.is_empty())
            && new.request_started_at.is_none()
        {
            bail!("轮次 {} 尚未记录模型请求，不能写入模型原文", old.id);
        }
    }
    Ok(())
}

fn same_authoritative_usage(
    left: crate::model::TokenUsage,
    right: crate::model::TokenUsage,
) -> bool {
    left.input_tokens == right.input_tokens && left.output_tokens == right.output_tokens
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
    use crate::model::{ContextTrace, ProvenanceQuality, TokenUsage, Turn, TurnStatus, utc_now};

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
            updated_at: now.clone(),
            status: TurnStatus::Complete,
            user_content: "hello".into(),
            assistant_content: "world".into(),
            thinking: "private".into(),
            usage: TokenUsage::new(Some(12), Some(4)),
            probe_usage: TokenUsage::new(Some(12), Some(1)),
            context_trace: ContextTrace {
                provenance_quality: ProvenanceQuality::LegacyInferred,
                ..ContextTrace::default()
            },
            request_started_at: Some(now.clone()),
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
        assert_eq!(session.ai_name, "LLM");
        assert_eq!(session.turns[0].thinking, "不会回注");
        assert_eq!(session.cumulative_usage.total_tokens, Some(16));
        store.save(&mut session).unwrap();
        let migrated = store.load(&session.id).unwrap();
        assert_eq!(migrated.schema_version, SCHEMA_VERSION);
        let answer_id = crate::model::event_id(
            &session.id,
            Some("turn-1"),
            crate::model::EventRole::Assistant,
        );
        let trace = store.retrieval().answer_context(&answer_id).unwrap();
        assert_eq!(trace.provenance_quality, ProvenanceQuality::LegacyInferred);
    }

    #[test]
    fn loads_schema_v2_without_name_as_llm() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let session = Session::new_named(
            "legacy-v2".into(),
            "model".into(),
            "http://localhost".into(),
            "temporary".into(),
            "original system".into(),
            BudgetConfig::default(),
            false,
        )
        .unwrap();
        let mut value = serde_json::to_value(session).unwrap();
        value["schema_version"] = serde_json::json!(2);
        value.as_object_mut().unwrap().remove("ai_name");
        fs::write(
            root.path().join("legacy-v2.json"),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();

        let mut loaded = store.load("legacy-v2").unwrap();
        assert_eq!(loaded.ai_name, "LLM");
        assert_eq!(loaded.system_prompt, "original system");
        store.save(&mut loaded).unwrap();
        assert_eq!(
            store.load("legacy-v2").unwrap().schema_version,
            SCHEMA_VERSION
        );
    }

    #[test]
    fn identifier_validation_blocks_path_traversal() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        assert!(store.resolve("../secret").is_err());
        assert!(store.resolve("/absolute").is_err());
    }

    #[test]
    fn persisted_events_are_logically_append_only() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create(
                "model",
                "http://localhost",
                Some("system"),
                BudgetConfig::default(),
                true,
            )
            .unwrap();
        session.turns.push(Turn::pending("原始用户文本".into()));
        store.save(&mut session).unwrap();

        session.system_prompt = "changed".into();
        assert!(store.save(&mut session).is_err());
        session = store.load(&session.id).unwrap();
        session.ai_name = "changed".into();
        assert!(store.save(&mut session).is_err());
        session = store.load(&session.id).unwrap();
        session.turns[0].user_content = "changed".into();
        assert!(store.save(&mut session).is_err());

        session = store.load(&session.id).unwrap();
        session.turns[0].context_trace.provenance_quality = ProvenanceQuality::LegacyInferred;
        session.turns[0].request_started_at = Some(utc_now());
        session.turns[0].assistant_content = "终态回复".into();
        session.turns[0].thinking = "审计 thinking".into();
        session.turns[0].usage = TokenUsage::new(Some(10), Some(4));
        session.turns[0].status = TurnStatus::Complete;
        store.save(&mut session).unwrap();

        let mut modified = store.load(&session.id).unwrap();
        modified.turns[0].assistant_content = "rewritten".into();
        assert!(store.save(&mut modified).is_err());

        let mut deleted = store.load(&session.id).unwrap();
        deleted.turns.clear();
        assert!(store.save(&mut deleted).is_err());

        let mut reordered = store.load(&session.id).unwrap();
        reordered.turns.push(Turn::pending("第二问".into()));
        store.save(&mut reordered).unwrap();
        reordered.turns.swap(0, 1);
        assert!(store.save(&mut reordered).is_err());
    }

    #[test]
    fn index_sync_failure_reports_that_source_was_committed() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create(
                "model",
                "http://localhost",
                None,
                BudgetConfig::default(),
                false,
            )
            .unwrap();
        let connection = rusqlite::Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .pragma_update(None, "user_version", 99_i64)
            .unwrap();
        drop(connection);

        session.title = "source committed".into();
        let error = store.save(&mut session).unwrap_err();
        assert!(error.downcast_ref::<IndexSyncAfterSourceCommit>().is_some());
        assert!(error.to_string().contains("原始会话已安全保存"));
        assert_eq!(store.load(&session.id).unwrap().title, "source committed");
        let source_path = root.path().join(format!("{}.json", session.id));
        let source_bytes = fs::read(&source_path).unwrap();
        let index_path = store.retrieval().index_path().to_path_buf();
        let index_bytes = fs::read(&index_path).unwrap();
        assert!(matches!(
            store.retrieval().rebuild(),
            Err(RetrievalError::UnsupportedIndexVersion(99))
        ));
        assert_eq!(fs::read(&source_path).unwrap(), source_bytes);
        assert_eq!(fs::read(&index_path).unwrap(), index_bytes);
        assert_eq!(
            rusqlite::Connection::open(&index_path)
                .unwrap()
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            99
        );
    }
}
