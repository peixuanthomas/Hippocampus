use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::knowledge::{KnowledgeStore, KnowledgeTrace};
use crate::model::{
    ChatMessage, ContextItemTrace, EventRole, EvidenceKind, ModelRequestTrace, ProvenanceQuality,
    RankedCandidate, RetrievalConfig, RetrievalDocumentGranularity, RetrievalTrace, SCHEMA_VERSION,
    SelectedEvidence, Session, SourceSpan, Turn, TurnStatus, WebTrace, content_sha256,
    context_sha256, event_id,
};

pub const INDEX_FILENAME: &str = ".hippocampus-index.sqlite3";
const INDEX_SCHEMA_VERSION: i64 = 4;

#[derive(Debug, Error)]
pub enum RetrievalError {
    #[error("无法访问派生索引 {path}: {source}")]
    Database {
        path: PathBuf,
        #[source]
        source: rusqlite::Error,
    },
    #[error("无法访问原始会话文件 {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("原始会话文件无效 {path}: {message}")]
    InvalidSource { path: PathBuf, message: String },
    #[error("派生索引版本不受支持：{0}")]
    UnsupportedIndexVersion(i64),
    #[error("索引中找不到会话 {0}")]
    SessionNotFound(String),
    #[error("索引中找不到事件 {0}")]
    EventNotFound(String),
    #[error("索引中找不到回答上下文 {0}")]
    AnswerContextNotFound(String),
    #[error("会话 {session_id} 的派生索引已过期或原文件缺失，请重新同步或重建")]
    StaleIndex { session_id: String },
    #[error(
        "原文片段范围无效：事件 {event_id} 共有 {char_count} 个字符，请求 [{start_char}..{end_char}]"
    )]
    InvalidSpan {
        event_id: String,
        start_char: usize,
        end_char: usize,
        char_count: usize,
    },
    #[error("派生索引内容校验失败：{0}")]
    CorruptIndex(String),
}

pub type RetrievalResult<T> = std::result::Result<T, RetrievalError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexedSession {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub source_file: String,
    pub source_sha256: String,
    pub source_schema_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredEvent {
    pub id: String,
    pub session_id: String,
    pub turn_id: Option<String>,
    pub sequence: usize,
    pub role: EventRole,
    pub created_at: String,
    pub content: String,
    pub content_sha256: String,
    pub reply_to_event_id: Option<String>,
    pub token_count: Option<u64>,
    pub turn_status: Option<TurnStatus>,
    pub done_reason: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResolvedSpan {
    pub span: SourceSpan,
    pub content: String,
    pub content_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnswerContextItem {
    pub ordinal: usize,
    pub role: EventRole,
    pub resolved: ResolvedSpan,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnswerContext {
    pub answer_event_id: String,
    pub turn_id: String,
    pub context_sha256: String,
    pub estimated_upper_tokens: Option<u64>,
    pub exact_input_tokens: Option<u64>,
    pub input_budget: u64,
    pub decision: String,
    pub provenance_quality: ProvenanceQuality,
    pub request: Option<ModelRequestTrace>,
    pub identity_instruction: Option<String>,
    pub items: Vec<AnswerContextItem>,
    pub retrieval_trace: RetrievalTrace,
    pub knowledge_trace: KnowledgeTrace,
    pub web_trace: WebTrace,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SyncReport {
    pub sessions: usize,
    pub events: usize,
    pub spans: usize,
    pub answer_contexts: usize,
    pub documents: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecalledEvidence {
    pub selected: SelectedEvidence,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecallResult {
    pub trace: RetrievalTrace,
    pub evidence: Vec<RecalledEvidence>,
}

#[derive(Debug, Clone)]
pub struct RetrievalStore {
    root: PathBuf,
    index_path: PathBuf,
}

impl RetrievalStore {
    pub fn new(root: impl AsRef<Path>) -> RetrievalResult<Self> {
        let path = root.as_ref();
        let root = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|source| RetrievalError::Io {
                    path: path.to_path_buf(),
                    source,
                })?
                .join(path)
        };
        Ok(Self {
            index_path: root.join(INDEX_FILENAME),
            root,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    pub fn sync_session(
        &self,
        expected_session: &Session,
        source_path: &Path,
    ) -> RetrievalResult<SyncReport> {
        let source = self.read_source(source_path)?;
        if source.session.id != expected_session.id {
            return Err(RetrievalError::InvalidSource {
                path: source_path.to_path_buf(),
                message: format!(
                    "落盘会话 ID {} 与待同步会话 ID {} 不一致",
                    source.session.id, expected_session.id
                ),
            });
        }
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction()
            .map_err(|source| self.database_error(source))?;
        let report = self.write_session(&transaction, &source, true)?;
        transaction
            .commit()
            .map_err(|source| self.database_error(source))?;
        Ok(report)
    }

    pub fn rebuild(&self) -> RetrievalResult<SyncReport> {
        let sources = self.load_all_sources()?;
        let mut connection = match self.open_connection() {
            Ok(connection) => connection,
            Err(RetrievalError::UnsupportedIndexVersion(_)) => {
                self.remove_index_files()?;
                self.open_connection()?
            }
            Err(error) => return Err(error),
        };
        let transaction = connection
            .transaction()
            .map_err(|source| self.database_error(source))?;
        transaction
            .execute_batch(
                "DELETE FROM retrieval_documents_fts;
                 DELETE FROM retrieval_documents;
                 DELETE FROM retrieval_runs;
                 DELETE FROM answer_context_items;
                 DELETE FROM answer_contexts;
                 DELETE FROM source_spans;
                 DELETE FROM events;
                 DELETE FROM indexed_sessions;",
            )
            .map_err(|source| self.database_error(source))?;
        let mut total = SyncReport::default();
        for source in &sources {
            add_report(&mut total, self.write_session(&transaction, source, false)?);
        }
        // Every immutable event/span now exists, independent of filename order.
        // The second pass only needs materialize answer references; it may
        // refresh that source's own derived documents without deleting events.
        for source in &sources {
            let report = self.write_session(&transaction, source, true)?;
            total.answer_contexts += report.answer_contexts;
        }
        transaction
            .commit()
            .map_err(|source| self.database_error(source))?;
        Ok(total)
    }

    pub fn get_session(&self, session_id: &str) -> RetrievalResult<IndexedSession> {
        let connection = self.open_connection()?;
        let session = connection
            .query_row(
                "SELECT session_id, title, created_at, updated_at, source_file, source_sha256, source_schema_version
                 FROM indexed_sessions WHERE session_id = ?1",
                [session_id],
                map_session,
            )
            .optional()
            .map_err(|source| self.database_error(source))?
            .ok_or_else(|| RetrievalError::SessionNotFound(session_id.to_owned()))?;
        self.verify_fresh(&session)?;
        Ok(session)
    }

    pub fn get_event(&self, event_id: &str) -> RetrievalResult<StoredEvent> {
        let connection = self.open_connection()?;
        let event = self.get_event_from_connection(&connection, event_id)?;
        let session = self.get_session_from_connection(&connection, &event.session_id)?;
        self.verify_fresh(&session)?;
        verify_event_hash(&event)?;
        Ok(event)
    }

    pub fn replay_session(&self, session_id: &str) -> RetrievalResult<Vec<StoredEvent>> {
        let connection = self.open_connection()?;
        let session = self.get_session_from_connection(&connection, session_id)?;
        self.verify_fresh(&session)?;
        let mut statement = connection
            .prepare(
                "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                        content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error
                 FROM events WHERE session_id = ?1 ORDER BY sequence",
            )
            .map_err(|source| self.database_error(source))?;
        let rows = statement
            .query_map([session_id], map_event)
            .map_err(|source| self.database_error(source))?;
        let mut events = Vec::new();
        for row in rows {
            let event = row.map_err(|source| self.database_error(source))?;
            verify_event_hash(&event)?;
            events.push(event);
        }
        Ok(events)
    }

    pub fn resolve_span(&self, span: &SourceSpan) -> RetrievalResult<ResolvedSpan> {
        let connection = self.open_connection()?;
        let event = self.get_event_from_connection(&connection, &span.event_id)?;
        let session = self.get_session_from_connection(&connection, &event.session_id)?;
        self.verify_fresh(&session)?;
        verify_event_hash(&event)?;
        let start_char =
            usize_to_i64(span.start_char).map_err(|source| self.database_error(source))?;
        let end_char = usize_to_i64(span.end_char).map_err(|source| self.database_error(source))?;
        let saved_hash = connection
            .query_row(
                "SELECT content_sha256 FROM source_spans
                 WHERE event_id = ?1 AND start_char = ?2 AND end_char = ?3",
                params![span.event_id, start_char, end_char],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|source| self.database_error(source))?;
        let content = slice_chars(&event.content, span)?;
        let actual_hash = content_sha256(&content);
        if saved_hash.is_some_and(|saved_hash| actual_hash != saved_hash) {
            return Err(RetrievalError::CorruptIndex(format!(
                "片段 {}[{}..{}] 的哈希不匹配",
                span.event_id, span.start_char, span.end_char
            )));
        }
        Ok(ResolvedSpan {
            span: span.clone(),
            content,
            content_sha256: actual_hash,
        })
    }

    /// Deterministic FTS5 recall.  The FTS expression is assembled solely
    /// from quoted tokenizer output; user punctuation never becomes syntax.
    pub fn keyword_recall(
        &self,
        raw_query: &str,
        current_user_event_id: &str,
        recent_event_ids: &[String],
        config: RetrievalConfig,
    ) -> RetrievalResult<RecallResult> {
        config
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let terms = query_terms(raw_query);
        let mut trace = RetrievalTrace {
            status: "ok".into(),
            current_query_event_id: current_user_event_id.into(),
            query_terms: terms.clone(),
            config,
            ..Default::default()
        };
        if terms.is_empty() {
            trace.status = "empty_query".into();
            return Ok(RecallResult {
                trace,
                evidence: Vec::new(),
            });
        }
        let expression = terms
            .iter()
            .map(|term| format!("\"{}\"", term.replace('"', "")))
            .collect::<Vec<_>>()
            .join(" OR ");
        let connection = self.open_connection()?;
        let mut statement = connection.prepare(
            "SELECT d.document_id, d.granularity, d.event_id, d.start_char, d.end_char, d.content_sha256, d.exact_content,
                    e.role, e.session_id, e.created_at, bm25(retrieval_documents_fts) AS score
             FROM retrieval_documents_fts JOIN retrieval_documents d ON d.rowid = retrieval_documents_fts.rowid
             JOIN events e ON e.event_id = d.event_id
             WHERE retrieval_documents_fts MATCH ?1
             ORDER BY score ASC, d.document_id ASC LIMIT ?2"
        ).map_err(|e| self.database_error(e))?;
        let rows = statement
            .query_map(
                params![expression, (trace.config.candidate_limit * 4) as i64],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        i64_to_usize(row.get(3)?)?,
                        i64_to_usize(row.get(4)?)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        parse_role(&row.get::<_, String>(7)?)?,
                        row.get::<_, String>(8)?,
                        row.get::<_, String>(9)?,
                        row.get::<_, f64>(10)?,
                    ))
                },
            )
            .map_err(|e| self.database_error(e))?;
        let fetched = rows
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| self.database_error(e))?;
        drop(statement);
        let recent: HashSet<&str> = recent_event_ids.iter().map(String::as_str).collect();
        let mut used_events = HashSet::new();
        let mut used_hashes = HashSet::new();
        let mut core_chars = 0usize;
        // Keep the entire deterministic overfetch in the trace. Exclusions
        // must not consume the usable candidate pool.
        for (idx, row) in fetched.into_iter().enumerate() {
            let (
                document_id,
                granularity,
                event_id_value,
                start,
                end,
                hash,
                stored_content,
                role,
                session_id,
                created_at,
                score,
            ) = row;
            let span = SourceSpan {
                event_id: event_id_value.clone(),
                start_char: start,
                end_char: end,
            };
            let mut candidate = RankedCandidate {
                raw_rank: idx + 1,
                document_id,
                granularity: if granularity == "fragment" {
                    RetrievalDocumentGranularity::Fragment
                } else {
                    RetrievalDocumentGranularity::Message
                },
                span: span.clone(),
                role,
                session_id,
                created_at,
                content_sha256: hash.clone(),
                bm25_score: score,
                selected: false,
                reason: String::new(),
            };
            if event_id_value == current_user_event_id {
                candidate.reason = "current_message".into();
            } else if recent.contains(event_id_value.as_str()) {
                candidate.reason = "recent_context".into();
            } else if used_events.contains(&event_id_value) {
                candidate.reason = "duplicate_event".into();
            } else if used_hashes.contains(&hash) {
                candidate.reason = "duplicate_content".into();
            } else {
                let source_event = self.get_event_from_connection(&connection, &span.event_id)?;
                let source_session =
                    self.get_session_from_connection(&connection, &source_event.session_id)?;
                self.verify_fresh(&source_session)?;
                verify_event_hash(&source_event)?;
                let span_content = slice_chars(&source_event.content, &span)?;
                let span_hash = connection.query_row("SELECT content_sha256 FROM source_spans WHERE event_id=?1 AND start_char=?2 AND end_char=?3", params![span.event_id, span.start_char as i64, span.end_char as i64], |row| row.get::<_, String>(0)).optional().map_err(|e| self.database_error(e))?;
                if stored_content != span_content
                    || hash != content_sha256(&span_content)
                    || span_hash.as_deref() != Some(hash.as_str())
                {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "检索文档 {} 与原始片段不一致",
                        candidate.document_id
                    )));
                }
                if span_content.chars().count() + core_chars > trace.config.evidence_char_budget {
                    candidate.reason = "evidence_budget".into();
                } else if trace
                    .selected_evidence
                    .iter()
                    .filter(|e| e.kind == EvidenceKind::Core)
                    .count()
                    >= trace.config.max_selected
                {
                    candidate.reason = "selection_limit".into();
                } else {
                    candidate.selected = true;
                    candidate.reason = "selected_core".into();
                    core_chars += span_content.chars().count();
                    used_events.insert(event_id_value);
                    used_hashes.insert(hash.clone());
                    trace.selected_evidence.push(SelectedEvidence {
                        span,
                        content_sha256: hash,
                        role,
                        kind: EvidenceKind::Core,
                        originating_candidate_rank: Some(candidate.raw_rank),
                        reason: "bm25_core".into(),
                    });
                }
            }
            trace.candidates.push(candidate);
        }
        let mut evidence = Vec::new();
        for selected in trace.selected_evidence.clone() {
            let event = self.get_event_from_connection(&connection, &selected.span.event_id)?;
            verify_event_hash(&event)?;
            let content = slice_chars(&event.content, &selected.span)?;
            evidence.push(RecalledEvidence { selected, content });
        }
        self.expand_context(
            &connection,
            &mut trace,
            &mut evidence,
            current_user_event_id,
            &recent,
            &mut used_events,
            &mut used_hashes,
        )?;
        Ok(RecallResult { trace, evidence })
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_context(
        &self,
        connection: &Connection,
        trace: &mut RetrievalTrace,
        evidence: &mut Vec<RecalledEvidence>,
        current: &str,
        recent: &HashSet<&str>,
        used_events: &mut HashSet<String>,
        used_hashes: &mut HashSet<String>,
    ) -> RetrievalResult<()> {
        let mut budget = 0usize;
        let cores = evidence.clone();
        for core in cores {
            let event = self.get_event_from_connection(connection, &core.selected.span.event_id)?;
            let candidates = [
                (event.reply_to_event_id.clone(), "reply_parent"),
                (connection.query_row("SELECT event_id FROM events WHERE reply_to_event_id=?1 ORDER BY sequence LIMIT 1", [&event.id], |r| r.get(0)).optional().map_err(|e| self.database_error(e))?, "reply_child"),
                (connection.query_row("SELECT event_id FROM events WHERE session_id=?1 AND sequence<?2 ORDER BY sequence DESC LIMIT 1", params![event.session_id, event.sequence as i64], |r| r.get(0)).optional().map_err(|e| self.database_error(e))?, "adjacent_before"),
                (connection.query_row("SELECT event_id FROM events WHERE session_id=?1 AND sequence>?2 ORDER BY sequence LIMIT 1", params![event.session_id, event.sequence as i64], |r| r.get(0)).optional().map_err(|e| self.database_error(e))?, "adjacent_after"),
            ];
            for (id, reason) in candidates {
                let Some(id) = id else { continue };
                if id == current
                    || recent.contains(id.as_str())
                    || used_events.contains(&id)
                    || evidence.iter().any(|e| e.selected.span.event_id == id)
                {
                    continue;
                }
                let adjacent = self.get_event_from_connection(connection, &id)?;
                let session = self.get_session_from_connection(connection, &adjacent.session_id)?;
                self.verify_fresh(&session)?;
                verify_event_hash(&adjacent)?;
                if adjacent.role == EventRole::System
                    || used_hashes.contains(&adjacent.content_sha256)
                {
                    continue;
                }
                let chars = adjacent.content.chars().count();
                let stored_span_hash = connection.query_row("SELECT content_sha256 FROM source_spans WHERE event_id=?1 AND start_char=0 AND end_char=?2", params![adjacent.id, chars as i64], |row| row.get::<_, String>(0)).optional().map_err(|e| self.database_error(e))?;
                if stored_span_hash.as_deref() != Some(adjacent.content_sha256.as_str()) {
                    return Err(RetrievalError::CorruptIndex(
                        "扩展上下文片段与原文不一致".into(),
                    ));
                }
                if budget + chars > trace.config.expansion_char_budget {
                    continue;
                }
                budget += chars;
                let selected = SelectedEvidence {
                    span: SourceSpan {
                        event_id: id,
                        start_char: 0,
                        end_char: chars,
                    },
                    content_sha256: adjacent.content_sha256.clone(),
                    role: adjacent.role,
                    kind: EvidenceKind::Context,
                    originating_candidate_rank: core.selected.originating_candidate_rank,
                    reason: reason.into(),
                };
                trace.selected_evidence.push(selected.clone());
                used_events.insert(selected.span.event_id.clone());
                used_hashes.insert(selected.content_sha256.clone());
                evidence.push(RecalledEvidence {
                    selected,
                    content: adjacent.content,
                });
            }
        }
        Ok(())
    }

    pub fn answer_context(&self, answer_event_id: &str) -> RetrievalResult<AnswerContext> {
        let connection = self.open_connection()?;
        let event = self.get_event_from_connection(&connection, answer_event_id)?;
        let session = self.get_session_from_connection(&connection, &event.session_id)?;
        self.verify_fresh(&session)?;
        verify_event_hash(&event)?;
        let mut answer = connection
            .query_row(
                "SELECT answer_event_id, turn_id, context_sha256, estimated_upper_tokens,
                        exact_input_tokens, input_budget, decision, provenance_quality,
                        request_model, request_think, request_context_window, request_max_output_tokens,
                        identity_instruction
                 FROM answer_contexts WHERE answer_event_id = ?1",
                [answer_event_id],
                map_answer_context,
            )
            .optional()
            .map_err(|source| self.database_error(source))?
            .ok_or_else(|| {
                RetrievalError::AnswerContextNotFound(answer_event_id.to_owned())
            })?;
        answer.retrieval_trace = connection
            .query_row(
                "SELECT trace_json FROM retrieval_runs WHERE answer_event_id=?1",
                [answer_event_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| self.database_error(e))?
            .map(|json| {
                serde_json::from_str(&json).map_err(|e| RetrievalError::CorruptIndex(e.to_string()))
            })
            .transpose()?
            .unwrap_or_default();
        let source = self.read_source(&self.root.join(&session.source_file))?;
        let turn = source
            .session
            .turns
            .iter()
            .find(|turn| turn.id == answer.turn_id)
            .ok_or_else(|| {
                RetrievalError::CorruptIndex(format!(
                    "回答 {answer_event_id} 在原始会话中缺少对应轮次"
                ))
            })?;
        answer.knowledge_trace = turn.context_trace.knowledge.clone();
        KnowledgeStore::new(&self.root)
            .and_then(|store| store.verify_trace(&answer.knowledge_trace))
            .map_err(|error| {
                RetrievalError::CorruptIndex(format!("知识证据校验失败：{error:#}"))
            })?;
        answer.web_trace = turn.context_trace.web.clone();
        answer.web_trace.validate().map_err(|error| {
            RetrievalError::CorruptIndex(format!("联网 trace 校验失败：{error:#}"))
        })?;
        let mut statement = connection
            .prepare(
                "SELECT i.ordinal, i.role, i.event_id, i.start_char, i.end_char,
                        i.content_sha256, e.content
                 FROM answer_context_items i
                 JOIN events e ON e.event_id = i.event_id
                 WHERE i.answer_event_id = ?1 ORDER BY i.ordinal",
            )
            .map_err(|source| self.database_error(source))?;
        let rows = statement
            .query_map([answer_event_id], |row| {
                let ordinal = i64_to_usize(row.get(0)?)?;
                let role = parse_role(&row.get::<_, String>(1)?)?;
                let span = SourceSpan {
                    event_id: row.get(2)?,
                    start_char: i64_to_usize(row.get(3)?)?,
                    end_char: i64_to_usize(row.get(4)?)?,
                };
                let expected_hash: String = row.get(5)?;
                let event_content: String = row.get(6)?;
                let content = slice_chars_sql(&event_content, &span)?;
                Ok((ordinal, role, span, expected_hash, content))
            })
            .map_err(|source| self.database_error(source))?;
        let mut messages = Vec::new();
        let mut inserted_generated = false;
        for row in rows {
            let (ordinal, role, span, expected_hash, content) =
                row.map_err(|source| self.database_error(source))?;
            let source_event = self.get_event_from_connection(&connection, &span.event_id)?;
            let source_session =
                self.get_session_from_connection(&connection, &source_event.session_id)?;
            self.verify_fresh(&source_session)?;
            verify_event_hash(&source_event)?;
            if source_event.role != role {
                return Err(RetrievalError::CorruptIndex(
                    "回答上下文角色与原始事件不匹配".into(),
                ));
            }
            let span_hash = connection.query_row("SELECT content_sha256 FROM source_spans WHERE event_id=?1 AND start_char=?2 AND end_char=?3", params![span.event_id, span.start_char as i64, span.end_char as i64], |row| row.get::<_, String>(0)).optional().map_err(|e| self.database_error(e))?.ok_or_else(|| RetrievalError::CorruptIndex("回答上下文缺少原始片段".into()))?;
            let actual_hash = content_sha256(&content);
            if actual_hash != expected_hash || actual_hash != span_hash {
                return Err(RetrievalError::CorruptIndex(format!(
                    "回答上下文片段 {} 的哈希不匹配",
                    span.event_id
                )));
            }
            if !inserted_generated && role != EventRole::System {
                push_generated_messages(
                    &mut messages,
                    answer.identity_instruction.as_deref(),
                    answer.knowledge_trace.injected_message.as_deref(),
                );
                inserted_generated = true;
            }
            messages.push(ChatMessage {
                role: role.as_str().to_owned(),
                content: content.clone(),
            });
            if !inserted_generated && role == EventRole::System {
                push_generated_messages(
                    &mut messages,
                    answer.identity_instruction.as_deref(),
                    answer.knowledge_trace.injected_message.as_deref(),
                );
                inserted_generated = true;
            }
            answer.items.push(AnswerContextItem {
                ordinal,
                role,
                resolved: ResolvedSpan {
                    span,
                    content,
                    content_sha256: actual_hash,
                },
            });
        }
        if context_sha256(&messages) != answer.context_sha256 {
            return Err(RetrievalError::CorruptIndex(format!(
                "回答 {answer_event_id} 的整体上下文哈希不匹配"
            )));
        }
        Ok(answer)
    }

    fn write_session(
        &self,
        transaction: &Transaction<'_>,
        source: &SessionSource,
        materialize_answers: bool,
    ) -> RetrievalResult<SyncReport> {
        let source_file = source_file_name(&self.root, &source.path)?;
        transaction.execute("DELETE FROM retrieval_documents_fts WHERE rowid IN (SELECT rowid FROM retrieval_documents WHERE event_id IN (SELECT event_id FROM events WHERE session_id=?1))", [source.session.id.as_str()]).map_err(|e| self.database_error(e))?;
        transaction.execute("DELETE FROM retrieval_documents WHERE event_id IN (SELECT event_id FROM events WHERE session_id=?1)", [source.session.id.as_str()]).map_err(|e| self.database_error(e))?;
        transaction.execute("DELETE FROM retrieval_runs WHERE answer_event_id IN (SELECT event_id FROM events WHERE session_id=?1)", [source.session.id.as_str()]).map_err(|e| self.database_error(e))?;
        transaction.execute("DELETE FROM answer_contexts WHERE answer_event_id IN (SELECT event_id FROM events WHERE session_id=?1)", [source.session.id.as_str()]).map_err(|e| self.database_error(e))?;
        transaction
            .execute(
                "INSERT INTO indexed_sessions
                 (session_id, title, created_at, updated_at, source_file, source_sha256,
                  source_schema_version, indexed_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(session_id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at,
                 source_file=excluded.source_file, source_sha256=excluded.source_sha256,
                 source_schema_version=excluded.source_schema_version, indexed_at=excluded.indexed_at",
                params![
                    source.session.id,
                    source.session.title,
                    source.session.created_at,
                    source.session.updated_at,
                    source_file,
                    source.sha256,
                    source.session.schema_version,
                    Utc::now().to_rfc3339(),
                ],
            )
            .map_err(|error| self.database_error(error))?;

        let events = derive_events(&source.session);
        let event_by_id = events
            .iter()
            .map(|event| (event.id.clone(), event))
            .collect::<HashMap<_, _>>();
        let expected_ids: HashSet<_> = events.iter().map(|event| event.id.as_str()).collect();
        let mut existing_statement = transaction
            .prepare("SELECT event_id FROM events WHERE session_id=?1")
            .map_err(|e| self.database_error(e))?;
        let existing_ids = existing_statement
            .query_map([source.session.id.as_str()], |row| row.get::<_, String>(0))
            .map_err(|e| self.database_error(e))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| self.database_error(e))?;
        drop(existing_statement);
        if existing_ids
            .iter()
            .any(|id| !expected_ids.contains(id.as_str()))
        {
            return Err(RetrievalError::InvalidSource {
                path: source.path.clone(),
                message: "源会话删除了已索引的不可变事件".into(),
            });
        }
        let mut spans = HashSet::new();
        let mut document_count = 0;
        for event in &events {
            if let Some(existing) = transaction.query_row("SELECT event_id, session_id, turn_id, sequence, role, created_at, content, content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error FROM events WHERE event_id=?1", [&event.id], map_event).optional().map_err(|e| self.database_error(e))? {
                let transition = existing.role == EventRole::Assistant && existing.content.is_empty() && !event.content.is_empty();
                if existing.session_id != event.session_id || existing.turn_id != event.turn_id || existing.sequence != event.sequence || existing.role != event.role || existing.created_at != event.created_at || (!transition && existing.content != event.content) {
                    return Err(RetrievalError::InvalidSource { path: source.path.clone(), message: format!("索引中的不可变事件 {} 与源文件不一致", event.id) });
                }
            }
            insert_event(transaction, event).map_err(|error| self.database_error(error))?;
            let full_span = SourceSpan {
                event_id: event.id.clone(),
                start_char: 0,
                end_char: event.content.chars().count(),
            };
            insert_span(transaction, &full_span, &event.content)
                .map_err(|error| self.database_error(error))?;
            spans.insert((full_span.event_id, full_span.start_char, full_span.end_char));
            if event.role != EventRole::System && !event.content.trim().is_empty() {
                for (granularity, span) in document_spans(event) {
                    let text = slice_chars(&event.content, &span)?;
                    insert_span(transaction, &span, &text)
                        .map_err(|error| self.database_error(error))?;
                    insert_document(transaction, event, &span, granularity, &text)
                        .map_err(|error| self.database_error(error))?;
                    document_count += 1;
                }
            }
        }

        if !materialize_answers {
            return Ok(SyncReport {
                sessions: 1,
                events: events.len(),
                spans: spans.len(),
                answer_contexts: 0,
                documents: document_count,
            });
        }
        let mut answer_context_count = 0;
        for turn in &source.session.turns {
            let answer_id = event_id(&source.session.id, Some(&turn.id), EventRole::Assistant);
            if !event_by_id.contains_key(&answer_id) {
                continue;
            }
            let derived = derive_context(
                &source.session,
                turn,
                &event_by_id,
                source.legacy,
                &source.path,
            )?;
            insert_answer_context(transaction, &answer_id, turn, &derived)
                .map_err(|error| self.database_error(error))?;
            transaction
                .execute(
                    "INSERT INTO retrieval_runs(answer_event_id, trace_json) VALUES(?1,?2)",
                    params![
                        answer_id,
                        serde_json::to_string(&turn.context_trace.retrieval)
                            .map_err(|e| self.database_error(
                                rusqlite::Error::ToSqlConversionFailure(Box::new(e))
                            ))?
                    ],
                )
                .map_err(|e| self.database_error(e))?;
            let store =
                KnowledgeStore::new(&self.root).map_err(|error| RetrievalError::InvalidSource {
                    path: source.path.clone(),
                    message: error.to_string(),
                })?;
            store
                .verify_trace(&derived.knowledge_trace)
                .map_err(|error| RetrievalError::InvalidSource {
                    path: source.path.clone(),
                    message: format!("回答 {} 的知识证据无效：{error:#}", turn.id),
                })?;
            let mut context_messages = Vec::with_capacity(derived.items.len() + 2);
            let mut inserted_generated = false;
            for (ordinal, item) in derived.items.iter().enumerate() {
                let local = event_by_id.get(&item.span.event_id).copied();
                let external = if local.is_none() {
                    transaction.query_row("SELECT event_id, session_id, turn_id, sequence, role, created_at, content, content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error FROM events WHERE event_id=?1", [&item.span.event_id], map_event).optional().map_err(|e| self.database_error(e))?
                } else {
                    None
                };
                let event =
                    local
                        .or(external.as_ref())
                        .ok_or_else(|| RetrievalError::InvalidSource {
                            path: source.path.clone(),
                            message: format!(
                                "回答 {} 引用了不存在的事件 {}",
                                turn.id, item.span.event_id
                            ),
                        })?;
                let indexed = transaction.query_row("SELECT session_id, title, created_at, updated_at, source_file, source_sha256, source_schema_version FROM indexed_sessions WHERE session_id=?1", [&event.session_id], map_session).map_err(|e| self.database_error(e))?;
                self.verify_fresh(&indexed)?;
                verify_event_hash(event)?;
                if item.role != event.role {
                    return Err(RetrievalError::InvalidSource {
                        path: source.path.clone(),
                        message: format!(
                            "回答 {} 的上下文角色与事件 {} 不一致",
                            turn.id, item.span.event_id
                        ),
                    });
                }
                let selected = slice_chars(&event.content, &item.span)?;
                let actual_hash = content_sha256(&selected);
                if actual_hash != item.content_sha256 {
                    return Err(RetrievalError::InvalidSource {
                        path: source.path.clone(),
                        message: format!(
                            "回答 {} 的上下文片段 {} 哈希不匹配",
                            turn.id, item.span.event_id
                        ),
                    });
                }
                if !inserted_generated && item.role != EventRole::System {
                    push_generated_messages(
                        &mut context_messages,
                        derived.identity_instruction.as_deref(),
                        derived.knowledge_trace.injected_message.as_deref(),
                    );
                    inserted_generated = true;
                }
                context_messages.push(ChatMessage {
                    role: item.role.as_str().to_owned(),
                    content: selected.clone(),
                });
                if !inserted_generated && item.role == EventRole::System {
                    push_generated_messages(
                        &mut context_messages,
                        derived.identity_instruction.as_deref(),
                        derived.knowledge_trace.injected_message.as_deref(),
                    );
                    inserted_generated = true;
                }
                insert_span(transaction, &item.span, &selected)
                    .map_err(|error| self.database_error(error))?;
                spans.insert((
                    item.span.event_id.clone(),
                    item.span.start_char,
                    item.span.end_char,
                ));
                let ordinal = usize_to_i64(ordinal).map_err(|error| self.database_error(error))?;
                let start_char = usize_to_i64(item.span.start_char)
                    .map_err(|error| self.database_error(error))?;
                let end_char =
                    usize_to_i64(item.span.end_char).map_err(|error| self.database_error(error))?;
                transaction
                    .execute(
                        "INSERT INTO answer_context_items
                         (answer_event_id, ordinal, role, event_id, start_char, end_char, content_sha256)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            answer_id,
                            ordinal,
                            item.role.as_str(),
                            item.span.event_id,
                            start_char,
                            end_char,
                            item.content_sha256,
                        ],
                    )
                    .map_err(|error| self.database_error(error))?;
            }
            if context_sha256(&context_messages) != derived.context_sha256 {
                return Err(RetrievalError::InvalidSource {
                    path: source.path.clone(),
                    message: format!("回答 {} 的整体上下文哈希不匹配", turn.id),
                });
            }
            answer_context_count += 1;
        }
        Ok(SyncReport {
            sessions: 1,
            events: events.len(),
            spans: spans.len(),
            answer_contexts: answer_context_count,
            documents: document_count,
        })
    }

    fn read_source(&self, path: &Path) -> RetrievalResult<SessionSource> {
        let bytes = fs::read(path).map_err(|source| RetrievalError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut session: Session =
            serde_json::from_slice(&bytes).map_err(|error| RetrievalError::InvalidSource {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        session.normalize_legacy_provenance();
        session
            .validate()
            .map_err(|error| RetrievalError::InvalidSource {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        if path.file_stem().and_then(|value| value.to_str()) != Some(session.id.as_str()) {
            return Err(RetrievalError::InvalidSource {
                path: path.to_path_buf(),
                message: "文件名必须与会话 ID 一致".into(),
            });
        }
        session.refresh_cumulative_usage();
        Ok(SessionSource {
            legacy: session.schema_version == crate::model::LEGACY_SCHEMA_VERSION,
            session,
            path: path.to_path_buf(),
            sha256: bytes_sha256(&bytes),
        })
    }

    fn load_all_sources(&self) -> RetrievalResult<Vec<SessionSource>> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let entries = fs::read_dir(&self.root).map_err(|source| RetrievalError::Io {
            path: self.root.clone(),
            source,
        })?;
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| RetrievalError::Io {
                path: self.root.clone(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) == Some("json")
                && path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| !value.starts_with('.'))
            {
                paths.push(path);
            }
        }
        paths.sort();
        paths.iter().map(|path| self.read_source(path)).collect()
    }

    pub(crate) fn open_connection(&self) -> RetrievalResult<Connection> {
        fs::create_dir_all(&self.root).map_err(|source| RetrievalError::Io {
            path: self.root.clone(),
            source,
        })?;
        let mut connection =
            Connection::open(&self.index_path).map_err(|source| self.database_error(source))?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|source| self.database_error(source))?;
        let version = connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .map_err(|source| self.database_error(source))?;
        if !matches!(version, 0 | 1 | 2 | 3 | INDEX_SCHEMA_VERSION) {
            return Err(RetrievalError::UnsupportedIndexVersion(version));
        }
        connection
            .execute_batch(
                "PRAGMA foreign_keys = ON;
                 PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = FULL;",
            )
            .map_err(|source| self.database_error(source))?;
        if matches!(version, 1 | 2) {
            // v1 contains all immutable events, so a transactional rebuild of
            // only derived tables is deterministic and loses no source data.
            let transaction = connection
                .transaction()
                .map_err(|e| self.database_error(e))?;
            transaction
                .execute_batch(SCHEMA_SQL)
                .map_err(|e| self.database_error(e))?;
            if !table_has_column(&transaction, "answer_contexts", "identity_instruction")
                .map_err(|e| self.database_error(e))?
            {
                transaction
                    .execute_batch(
                        "ALTER TABLE answer_contexts ADD COLUMN identity_instruction TEXT;",
                    )
                    .map_err(|e| self.database_error(e))?;
            }
            if version == 1 {
                transaction
                    .execute_batch(
                        "DELETE FROM retrieval_documents_fts; DELETE FROM retrieval_documents;",
                    )
                    .map_err(|e| self.database_error(e))?;
                let mut statement = transaction.prepare("SELECT event_id, session_id, turn_id, sequence, role, created_at, content, content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error FROM events").map_err(|e| self.database_error(e))?;
                let rows = statement
                    .query_map([], map_event)
                    .map_err(|e| self.database_error(e))?;
                let events: Vec<_> = rows
                    .collect::<Result<_, _>>()
                    .map_err(|e| self.database_error(e))?;
                drop(statement);
                for event in &events {
                    if event.role != EventRole::System && !event.content.trim().is_empty() {
                        for (granularity, span) in document_spans(event) {
                            let text = slice_chars(&event.content, &span)?;
                            insert_span(&transaction, &span, &text)
                                .map_err(|e| self.database_error(e))?;
                            insert_document(&transaction, event, &span, granularity, &text)
                                .map_err(|e| self.database_error(e))?;
                        }
                    }
                }
            }
            transaction
                .pragma_update(None, "user_version", INDEX_SCHEMA_VERSION)
                .map_err(|e| self.database_error(e))?;
            transaction.commit().map_err(|e| self.database_error(e))?;
        } else if matches!(version, 0 | 3) {
            let transaction = connection
                .transaction()
                .map_err(|e| self.database_error(e))?;
            transaction
                .execute_batch(SCHEMA_SQL)
                .map_err(|e| self.database_error(e))?;
            transaction
                .pragma_update(None, "user_version", INDEX_SCHEMA_VERSION)
                .map_err(|e| self.database_error(e))?;
            transaction.commit().map_err(|e| self.database_error(e))?;
        } else {
            connection
                .execute_batch(SCHEMA_SQL)
                .map_err(|source| self.database_error(source))?;
        }
        Ok(connection)
    }

    fn get_session_from_connection(
        &self,
        connection: &Connection,
        session_id: &str,
    ) -> RetrievalResult<IndexedSession> {
        connection
            .query_row(
                "SELECT session_id, title, created_at, updated_at, source_file, source_sha256, source_schema_version
                 FROM indexed_sessions WHERE session_id = ?1",
                [session_id],
                map_session,
            )
            .optional()
            .map_err(|source| self.database_error(source))?
            .ok_or_else(|| RetrievalError::SessionNotFound(session_id.to_owned()))
    }

    fn get_event_from_connection(
        &self,
        connection: &Connection,
        event_id: &str,
    ) -> RetrievalResult<StoredEvent> {
        connection
            .query_row(
                "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                        content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error
                 FROM events WHERE event_id = ?1",
                [event_id],
                map_event,
            )
            .optional()
            .map_err(|source| self.database_error(source))?
            .ok_or_else(|| RetrievalError::EventNotFound(event_id.to_owned()))
    }

    fn verify_fresh(&self, session: &IndexedSession) -> RetrievalResult<()> {
        if !is_safe_source_file(&session.source_file) {
            return Err(RetrievalError::CorruptIndex(format!(
                "会话 {} 的源文件名不安全",
                session.id
            )));
        }
        let path = self.root.join(&session.source_file);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => {
                return Err(RetrievalError::StaleIndex {
                    session_id: session.id.clone(),
                });
            }
        };
        if bytes_sha256(&bytes) != session.source_sha256 {
            return Err(RetrievalError::StaleIndex {
                session_id: session.id.clone(),
            });
        }
        Ok(())
    }

    pub(crate) fn database_error(&self, source: rusqlite::Error) -> RetrievalError {
        RetrievalError::Database {
            path: self.index_path.clone(),
            source,
        }
    }

    fn remove_index_files(&self) -> RetrievalResult<()> {
        for path in [
            self.index_path.clone(),
            index_sidecar(&self.index_path, "-wal"),
            index_sidecar(&self.index_path, "-shm"),
        ] {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => return Err(RetrievalError::Io { path, source }),
            }
        }
        Ok(())
    }
}

fn document_spans(event: &StoredEvent) -> Vec<(RetrievalDocumentGranularity, SourceSpan)> {
    let len = event.content.chars().count();
    let mut spans = vec![(
        RetrievalDocumentGranularity::Message,
        SourceSpan {
            event_id: event.id.clone(),
            start_char: 0,
            end_char: len,
        },
    )];
    if len > 240 {
        let mut start = 0;
        while start < len {
            let end = (start + 240).min(len);
            spans.push((
                RetrievalDocumentGranularity::Fragment,
                SourceSpan {
                    event_id: event.id.clone(),
                    start_char: start,
                    end_char: end,
                },
            ));
            if end == len {
                break;
            }
            start += 200;
        }
    }
    spans
}

fn insert_document(
    transaction: &Transaction<'_>,
    event: &StoredEvent,
    span: &SourceSpan,
    granularity: RetrievalDocumentGranularity,
    content: &str,
) -> rusqlite::Result<()> {
    let id = format!("{}:{}:{}", event.id, span.start_char, span.end_char);
    transaction.execute("INSERT INTO retrieval_documents (document_id,event_id,start_char,end_char,granularity,content_sha256,exact_content,lexical_content,ngram_content) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)", params![id, event.id, span.start_char as i64, span.end_char as i64, match granularity { RetrievalDocumentGranularity::Message => "message", RetrievalDocumentGranularity::Fragment => "fragment", RetrievalDocumentGranularity::Episode => "episode", RetrievalDocumentGranularity::Session => "session" }, content_sha256(content), content, lexical_field(content), ngram_field(content)])?;
    let rowid = transaction.last_insert_rowid();
    transaction.execute("INSERT INTO retrieval_documents_fts(rowid, lexical_content, ngram_content) VALUES(?1,?2,?3)", params![rowid, lexical_field(content), ngram_field(content)])?;
    Ok(())
}

fn jieba() -> &'static jieba_rs::Jieba {
    static JIEBA: OnceLock<jieba_rs::Jieba> = OnceLock::new();
    JIEBA.get_or_init(jieba_rs::Jieba::new)
}
pub(crate) fn lexical_field(content: &str) -> String {
    jieba()
        .cut(content, false)
        .iter()
        .map(|token| token.word)
        .collect::<Vec<_>>()
        .join(" ")
}
fn is_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF)
}
pub(crate) fn ngram_field(content: &str) -> String {
    let chars: Vec<_> = content.chars().collect();
    let mut output = Vec::new();
    for n in [2, 3] {
        for window in chars.windows(n) {
            if window.iter().all(|c| is_cjk(*c)) {
                output.push(window.iter().collect::<String>());
            }
        }
    }
    output.join(" ")
}
pub(crate) fn query_terms(raw: &str) -> Vec<String> {
    let mut terms = jieba()
        .cut(raw, false)
        .into_iter()
        .filter(|s| !s.word.trim().is_empty())
        .map(|s| s.word.to_owned())
        .collect::<Vec<_>>();
    let chars: Vec<_> = raw.chars().collect();
    for n in [2, 3] {
        for w in chars.windows(n).take(128) {
            if w.iter().all(|c| is_cjk(*c)) {
                terms.push(w.iter().collect());
            }
        }
    }
    terms.retain(|term| term.chars().any(|c| c.is_alphanumeric() || is_cjk(c)));
    terms.sort();
    terms.dedup();
    terms.truncate(128);
    terms
}

#[derive(Debug)]
struct SessionSource {
    session: Session,
    path: PathBuf,
    sha256: String,
    legacy: bool,
}

#[derive(Debug)]
struct DerivedContext {
    items: Vec<ContextItemTrace>,
    context_sha256: String,
    provenance_quality: ProvenanceQuality,
    request: Option<ModelRequestTrace>,
    identity_instruction: Option<String>,
    knowledge_trace: KnowledgeTrace,
}

fn derive_events(session: &Session) -> Vec<StoredEvent> {
    let mut events = Vec::new();
    if !session.system_prompt.is_empty() {
        events.push(StoredEvent {
            id: event_id(&session.id, None, EventRole::System),
            session_id: session.id.clone(),
            turn_id: None,
            sequence: 0,
            role: EventRole::System,
            created_at: session.created_at.clone(),
            content: session.system_prompt.clone(),
            content_sha256: content_sha256(&session.system_prompt),
            reply_to_event_id: None,
            token_count: None,
            turn_status: None,
            done_reason: None,
            error: None,
        });
    }
    let mut previous_assistant = None;
    for (index, turn) in session.turns.iter().enumerate() {
        let user_id = event_id(&session.id, Some(&turn.id), EventRole::User);
        events.push(StoredEvent {
            id: user_id.clone(),
            session_id: session.id.clone(),
            turn_id: Some(turn.id.clone()),
            sequence: index * 2 + 1,
            role: EventRole::User,
            created_at: turn.created_at.clone(),
            content: turn.user_content.clone(),
            content_sha256: content_sha256(&turn.user_content),
            reply_to_event_id: previous_assistant.clone(),
            token_count: None,
            turn_status: Some(turn.status),
            done_reason: None,
            error: turn.error.clone(),
        });
        if has_assistant_event(session, turn) {
            let assistant_id = event_id(&session.id, Some(&turn.id), EventRole::Assistant);
            let token_count = if turn.usage.input_tokens.is_some() {
                turn.usage.output_tokens
            } else {
                None
            };
            events.push(StoredEvent {
                id: assistant_id.clone(),
                session_id: session.id.clone(),
                turn_id: Some(turn.id.clone()),
                sequence: index * 2 + 2,
                role: EventRole::Assistant,
                created_at: turn
                    .request_started_at
                    .clone()
                    .unwrap_or_else(|| turn.updated_at.clone()),
                content: turn.assistant_content.clone(),
                content_sha256: content_sha256(&turn.assistant_content),
                reply_to_event_id: Some(user_id),
                token_count,
                turn_status: Some(turn.status),
                done_reason: turn.done_reason.clone(),
                error: turn.error.clone(),
            });
            previous_assistant = Some(assistant_id);
        }
    }
    events
}

fn has_assistant_event(session: &Session, turn: &Turn) -> bool {
    turn.request_started_at.is_some()
        || ((session.schema_version < SCHEMA_VERSION
            || turn.context_trace.provenance_quality == ProvenanceQuality::LegacyInferred)
            && (!turn.assistant_content.is_empty()
                || !turn.thinking.is_empty()
                || turn.usage.input_tokens.is_some()
                || turn.usage.output_tokens.is_some()))
}

fn derive_context(
    session: &Session,
    turn: &Turn,
    events: &HashMap<String, &StoredEvent>,
    legacy: bool,
    source_path: &Path,
) -> RetrievalResult<DerivedContext> {
    if !legacy && turn.context_trace.provenance_quality == ProvenanceQuality::Exact {
        if turn.context_trace.context_items.is_empty() {
            return Err(RetrievalError::InvalidSource {
                path: source_path.to_path_buf(),
                message: format!("回答 {} 缺少 v2 精确上下文溯源", turn.id),
            });
        }
        let context_hash = turn.context_trace.context_sha256.clone().ok_or_else(|| {
            RetrievalError::InvalidSource {
                path: source_path.to_path_buf(),
                message: format!("回答 {} 缺少 v2 上下文哈希", turn.id),
            }
        })?;
        let request =
            turn.context_trace
                .request
                .clone()
                .ok_or_else(|| RetrievalError::InvalidSource {
                    path: source_path.to_path_buf(),
                    message: format!("回答 {} 缺少 v2 请求元数据", turn.id),
                })?;
        return Ok(DerivedContext {
            items: turn.context_trace.context_items.clone(),
            context_sha256: context_hash,
            provenance_quality: ProvenanceQuality::Exact,
            request: Some(request),
            identity_instruction: turn.context_trace.identity_instruction.clone(),
            knowledge_trace: turn.context_trace.knowledge.clone(),
        });
    }

    let mut items = Vec::new();
    if let Some(system) = events.get(&event_id(&session.id, None, EventRole::System)) {
        items.push(full_item(system));
    }
    for included_turn_id in &turn.context_trace.included_turn_ids {
        for role in [EventRole::User, EventRole::Assistant] {
            let id = event_id(&session.id, Some(included_turn_id), role);
            if let Some(event) = events.get(&id) {
                items.push(full_item(event));
            }
        }
    }
    let current_user_id = event_id(&session.id, Some(&turn.id), EventRole::User);
    let current_user = events.get(&current_user_id).ok_or_else(|| {
        RetrievalError::CorruptIndex(format!("回答 {} 缺少当前用户事件", turn.id))
    })?;
    items.push(full_item(current_user));
    let mut messages = Vec::with_capacity(items.len());
    for item in &items {
        let event = events.get(&item.span.event_id).ok_or_else(|| {
            RetrievalError::CorruptIndex(format!(
                "推断上下文引用了不存在的事件 {}",
                item.span.event_id
            ))
        })?;
        messages.push(ChatMessage {
            role: item.role.as_str().to_owned(),
            content: event.content.clone(),
        });
    }
    Ok(DerivedContext {
        items,
        context_sha256: context_sha256(&messages),
        provenance_quality: ProvenanceQuality::LegacyInferred,
        request: None,
        identity_instruction: None,
        knowledge_trace: KnowledgeTrace::default(),
    })
}

fn full_item(event: &StoredEvent) -> ContextItemTrace {
    ContextItemTrace {
        role: event.role,
        span: SourceSpan {
            event_id: event.id.clone(),
            start_char: 0,
            end_char: event.content.chars().count(),
        },
        content_sha256: event.content_sha256.clone(),
    }
}

fn insert_event(transaction: &Transaction<'_>, event: &StoredEvent) -> rusqlite::Result<()> {
    let sequence = usize_to_i64(event.sequence)?;
    let token_count = event.token_count.map(u64_to_i64).transpose()?;
    transaction.execute(
        "INSERT INTO events
         (event_id, session_id, turn_id, sequence, role, created_at, content, content_sha256,
          reply_to_event_id, token_count, turn_status, done_reason, error)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
         ON CONFLICT(event_id) DO UPDATE SET sequence=excluded.sequence, content=excluded.content,
         content_sha256=excluded.content_sha256, reply_to_event_id=excluded.reply_to_event_id,
         token_count=excluded.token_count, turn_status=excluded.turn_status,
         done_reason=excluded.done_reason, error=excluded.error",
        params![
            event.id,
            event.session_id,
            event.turn_id,
            sequence,
            event.role.as_str(),
            event.created_at,
            event.content,
            event.content_sha256,
            event.reply_to_event_id,
            token_count,
            event.turn_status.map(TurnStatus::as_str),
            event.done_reason,
            event.error,
        ],
    )?;
    Ok(())
}

fn insert_span(
    transaction: &Transaction<'_>,
    span: &SourceSpan,
    content: &str,
) -> rusqlite::Result<()> {
    let start_char = usize_to_i64(span.start_char)?;
    let end_char = usize_to_i64(span.end_char)?;
    let hash = content_sha256(content);
    let existing = transaction.query_row("SELECT content_sha256 FROM source_spans WHERE event_id=?1 AND start_char=?2 AND end_char=?3", params![span.event_id, start_char, end_char], |row| row.get::<_, String>(0)).optional()?;
    if let Some(existing) = existing {
        if existing != hash {
            return Err(rusqlite::Error::ToSqlConversionFailure(Box::new(
                std::io::Error::other("source span hash mismatch"),
            )));
        }
        return Ok(());
    }
    transaction.execute(
        "INSERT INTO source_spans
         (event_id, start_char, end_char, content_sha256) VALUES (?1, ?2, ?3, ?4)",
        params![span.event_id, start_char, end_char, hash],
    )?;
    Ok(())
}

fn insert_answer_context(
    transaction: &Transaction<'_>,
    answer_event_id: &str,
    turn: &Turn,
    derived: &DerivedContext,
) -> rusqlite::Result<()> {
    let request = derived.request.as_ref();
    let estimated_upper_tokens = turn
        .context_trace
        .estimated_upper_tokens
        .map(u64_to_i64)
        .transpose()?;
    let exact_input_tokens = turn
        .context_trace
        .exact_input_tokens
        .map(u64_to_i64)
        .transpose()?;
    let input_budget = u64_to_i64(turn.context_trace.input_budget)?;
    let request_context_window = request
        .map(|value| u64_to_i64(value.context_window))
        .transpose()?;
    let request_max_output_tokens = request
        .map(|value| u64_to_i64(value.max_output_tokens))
        .transpose()?;
    transaction.execute(
        "INSERT INTO answer_contexts
         (answer_event_id, turn_id, context_sha256, estimated_upper_tokens, exact_input_tokens,
          input_budget, decision, provenance_quality, request_model, request_think,
          request_context_window, request_max_output_tokens, identity_instruction)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            answer_event_id,
            turn.id,
            derived.context_sha256,
            estimated_upper_tokens,
            exact_input_tokens,
            input_budget,
            turn.context_trace.decision,
            match derived.provenance_quality {
                ProvenanceQuality::Exact => "exact",
                ProvenanceQuality::LegacyInferred => "legacy_inferred",
            },
            request.map(|value| value.model.as_str()),
            request.map(|value| value.think),
            request_context_window,
            request_max_output_tokens,
            derived.identity_instruction,
        ],
    )?;
    Ok(())
}

fn map_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<IndexedSession> {
    Ok(IndexedSession {
        id: row.get(0)?,
        title: row.get(1)?,
        created_at: row.get(2)?,
        updated_at: row.get(3)?,
        source_file: row.get(4)?,
        source_sha256: row.get(5)?,
        source_schema_version: i64_to_u32(row.get(6)?)?,
    })
}

fn map_event(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredEvent> {
    let role = parse_role(&row.get::<_, String>(4)?)?;
    let status = row
        .get::<_, Option<String>>(10)?
        .map(|value| parse_status(&value))
        .transpose()?;
    Ok(StoredEvent {
        id: row.get(0)?,
        session_id: row.get(1)?,
        turn_id: row.get(2)?,
        sequence: i64_to_usize(row.get(3)?)?,
        role,
        created_at: row.get(5)?,
        content: row.get(6)?,
        content_sha256: row.get(7)?,
        reply_to_event_id: row.get(8)?,
        token_count: row.get::<_, Option<i64>>(9)?.map(i64_to_u64).transpose()?,
        turn_status: status,
        done_reason: row.get(11)?,
        error: row.get(12)?,
    })
}

fn map_answer_context(row: &rusqlite::Row<'_>) -> rusqlite::Result<AnswerContext> {
    let quality = match row.get::<_, String>(7)?.as_str() {
        "exact" => ProvenanceQuality::Exact,
        "legacy_inferred" => ProvenanceQuality::LegacyInferred,
        value => {
            return Err(conversion_error(format!(
                "unknown provenance quality {value}"
            )));
        }
    };
    let model: Option<String> = row.get(8)?;
    let think: Option<bool> = row.get(9)?;
    let context_window: Option<i64> = row.get(10)?;
    let max_output_tokens: Option<i64> = row.get(11)?;
    let request = match (model, think, context_window, max_output_tokens) {
        (Some(model), Some(think), Some(context_window), Some(max_output_tokens)) => {
            Some(ModelRequestTrace {
                model,
                think,
                context_window: i64_to_u64(context_window)?,
                max_output_tokens: i64_to_u64(max_output_tokens)?,
            })
        }
        (None, None, None, None) => None,
        _ => return Err(conversion_error("partial request metadata")),
    };
    Ok(AnswerContext {
        answer_event_id: row.get(0)?,
        turn_id: row.get(1)?,
        context_sha256: row.get(2)?,
        estimated_upper_tokens: row.get::<_, Option<i64>>(3)?.map(i64_to_u64).transpose()?,
        exact_input_tokens: row.get::<_, Option<i64>>(4)?.map(i64_to_u64).transpose()?,
        input_budget: i64_to_u64(row.get(5)?)?,
        decision: row.get(6)?,
        provenance_quality: quality,
        request,
        identity_instruction: row.get(12)?,
        items: Vec::new(),
        retrieval_trace: RetrievalTrace::default(),
        knowledge_trace: KnowledgeTrace::default(),
        web_trace: WebTrace::default(),
    })
}

fn push_generated_messages(
    messages: &mut Vec<ChatMessage>,
    identity_instruction: Option<&str>,
    knowledge_message: Option<&str>,
) {
    for content in [identity_instruction, knowledge_message]
        .into_iter()
        .flatten()
    {
        messages.push(ChatMessage {
            role: EventRole::System.as_str().to_owned(),
            content: content.to_owned(),
        });
    }
}

fn parse_role(value: &str) -> rusqlite::Result<EventRole> {
    match value {
        "system" => Ok(EventRole::System),
        "user" => Ok(EventRole::User),
        "assistant" => Ok(EventRole::Assistant),
        _ => Err(conversion_error(format!("unknown event role {value}"))),
    }
}

fn parse_status(value: &str) -> rusqlite::Result<TurnStatus> {
    match value {
        "pending" => Ok(TurnStatus::Pending),
        "complete" => Ok(TurnStatus::Complete),
        "truncated" => Ok(TurnStatus::Truncated),
        "blocked" => Ok(TurnStatus::Blocked),
        "interrupted" => Ok(TurnStatus::Interrupted),
        "failed" => Ok(TurnStatus::Failed),
        "no_answer" => Ok(TurnStatus::NoAnswer),
        _ => Err(conversion_error(format!("unknown turn status {value}"))),
    }
}

fn conversion_error(message: impl Into<String>) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        rusqlite::types::Type::Text,
        Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            message.into(),
        )),
    )
}

fn i64_to_usize(value: i64) -> rusqlite::Result<usize> {
    usize::try_from(value).map_err(|_| conversion_error(format!("invalid usize {value}")))
}

fn i64_to_u64(value: i64) -> rusqlite::Result<u64> {
    u64::try_from(value).map_err(|_| conversion_error(format!("invalid u64 {value}")))
}

fn i64_to_u32(value: i64) -> rusqlite::Result<u32> {
    u32::try_from(value).map_err(|_| conversion_error(format!("invalid u32 {value}")))
}

fn usize_to_i64(value: usize) -> rusqlite::Result<i64> {
    i64::try_from(value)
        .map_err(|_| conversion_error(format!("usize exceeds SQLite INTEGER: {value}")))
}

fn u64_to_i64(value: u64) -> rusqlite::Result<i64> {
    i64::try_from(value)
        .map_err(|_| conversion_error(format!("u64 exceeds SQLite INTEGER: {value}")))
}

fn table_has_column(connection: &Connection, table: &str, column: &str) -> rusqlite::Result<bool> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(names.iter().any(|name| name == column))
}

fn slice_chars(content: &str, span: &SourceSpan) -> RetrievalResult<String> {
    let char_count = content.chars().count();
    if span.start_char > span.end_char || span.end_char > char_count {
        return Err(RetrievalError::InvalidSpan {
            event_id: span.event_id.clone(),
            start_char: span.start_char,
            end_char: span.end_char,
            char_count,
        });
    }
    Ok(content
        .chars()
        .skip(span.start_char)
        .take(span.end_char - span.start_char)
        .collect())
}

fn slice_chars_sql(content: &str, span: &SourceSpan) -> rusqlite::Result<String> {
    let char_count = content.chars().count();
    if span.start_char > span.end_char || span.end_char > char_count {
        return Err(conversion_error(format!(
            "invalid span {}..{} for {char_count} chars",
            span.start_char, span.end_char
        )));
    }
    Ok(content
        .chars()
        .skip(span.start_char)
        .take(span.end_char - span.start_char)
        .collect())
}

fn verify_event_hash(event: &StoredEvent) -> RetrievalResult<()> {
    if content_sha256(&event.content) != event.content_sha256 {
        return Err(RetrievalError::CorruptIndex(format!(
            "事件 {} 的内容哈希不匹配",
            event.id
        )));
    }
    Ok(())
}

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn index_sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn source_file_name(root: &Path, path: &Path) -> RetrievalResult<String> {
    if path.parent() != Some(root) {
        return Err(RetrievalError::InvalidSource {
            path: path.to_path_buf(),
            message: "会话源文件必须直接位于 sessions 目录".into(),
        });
    }
    let value = path
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| is_safe_source_file(value))
        .ok_or_else(|| RetrievalError::InvalidSource {
            path: path.to_path_buf(),
            message: "会话源文件名不安全".into(),
        })?;
    Ok(value.to_owned())
}

fn is_safe_source_file(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn add_report(total: &mut SyncReport, report: SyncReport) {
    total.sessions += report.sessions;
    total.events += report.events;
    total.spans += report.spans;
    total.answer_contexts += report.answer_contexts;
    total.documents += report.documents;
}

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS indexed_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_file TEXT NOT NULL UNIQUE,
    source_sha256 TEXT NOT NULL,
    source_schema_version INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES indexed_sessions(session_id) ON DELETE CASCADE,
    turn_id TEXT,
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
    created_at TEXT NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    reply_to_event_id TEXT REFERENCES events(event_id),
    token_count INTEGER CHECK(token_count IS NULL OR token_count >= 0),
    turn_status TEXT,
    done_reason TEXT,
    error TEXT,
    UNIQUE(session_id, sequence),
    UNIQUE(session_id, turn_id, role),
    CHECK((role = 'system' AND turn_id IS NULL AND sequence = 0)
       OR (role != 'system' AND turn_id IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS source_spans (
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char >= start_char),
    content_sha256 TEXT NOT NULL,
    PRIMARY KEY(event_id, start_char, end_char)
);

CREATE TABLE IF NOT EXISTS answer_contexts (
    answer_event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL,
    context_sha256 TEXT NOT NULL,
    estimated_upper_tokens INTEGER,
    exact_input_tokens INTEGER,
    input_budget INTEGER NOT NULL,
    decision TEXT NOT NULL,
    provenance_quality TEXT NOT NULL CHECK(provenance_quality IN ('exact', 'legacy_inferred')),
    request_model TEXT,
    request_think INTEGER,
    request_context_window INTEGER,
    request_max_output_tokens INTEGER,
    identity_instruction TEXT
);

CREATE TABLE IF NOT EXISTS answer_context_items (
    answer_event_id TEXT NOT NULL REFERENCES answer_contexts(answer_event_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
    event_id TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    PRIMARY KEY(answer_event_id, ordinal),
    FOREIGN KEY(event_id, start_char, end_char)
        REFERENCES source_spans(event_id, start_char, end_char)
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    answer_event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    trace_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_documents (
    document_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    granularity TEXT NOT NULL CHECK(granularity IN ('message','fragment')),
    content_sha256 TEXT NOT NULL,
    exact_content TEXT NOT NULL,
    lexical_content TEXT NOT NULL,
    ngram_content TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_documents_fts USING fts5(
    lexical_content, ngram_content, tokenize='unicode61'
);

CREATE INDEX IF NOT EXISTS events_session_sequence
    ON events(session_id, sequence);
CREATE INDEX IF NOT EXISTS events_reply_to
    ON events(reply_to_event_id);

CREATE TABLE IF NOT EXISTS consolidation_watermarks (
    session_id TEXT PRIMARY KEY,
    through_sequence INTEGER NOT NULL CHECK(through_sequence >= 0),
    through_event_id TEXT,
    through_event_sha256 TEXT,
    updated_at TEXT,
    CHECK((through_event_id IS NULL AND through_event_sha256 IS NULL)
       OR (through_event_id IS NOT NULL AND through_event_sha256 IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS consolidation_batches (
    attempt_id TEXT PRIMARY KEY,
    batch_key TEXT NOT NULL,
    session_id TEXT NOT NULL,
    from_sequence INTEGER NOT NULL CHECK(from_sequence >= 0),
    through_sequence INTEGER NOT NULL CHECK(through_sequence >= 0),
    trigger TEXT NOT NULL,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    input_event_ids TEXT NOT NULL,
    input_event_hashes TEXT NOT NULL,
    response_json TEXT,
    response_sha256 TEXT,
    status TEXT NOT NULL CHECK(status IN ('applied', 'rejected', 'model_error', 'cancelled')),
    input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens >= 0),
    latency_ms INTEGER NOT NULL CHECK(latency_ms >= 0),
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    validation_json TEXT,
    error_json TEXT,
    CHECK(from_sequence <= through_sequence),
    CHECK((response_json IS NULL AND response_sha256 IS NULL)
       OR (response_json IS NOT NULL AND response_sha256 IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS consolidation_batches_session_started
    ON consolidation_batches(session_id, started_at, attempt_id);
CREATE INDEX IF NOT EXISTS consolidation_batches_batch_key
    ON consolidation_batches(batch_key);
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextAssembler;
    use crate::model::{ContextTrace, ModelRequestTrace, TokenUsage, Turn, TurnStatus, utc_now};
    use crate::store::SessionStore;

    fn append_complete_turn(
        session: &mut Session,
        user: &str,
        assistant: &str,
        thinking: &str,
    ) -> String {
        session.turns.push(Turn::pending(user.to_owned()));
        let index = session.turns.len() - 1;
        let plan = ContextAssembler.assemble(session, user, None, Some(index));
        let started_at = utc_now();
        let turn = &mut session.turns[index];
        turn.context_trace = ContextTrace {
            included_turn_ids: plan.included_turn_ids,
            omitted_turn_ids: plan.omitted_turn_ids,
            estimated_upper_tokens: plan.estimated_upper_tokens,
            exact_input_tokens: Some(42),
            input_budget: plan.input_budget,
            decision: "ready".into(),
            active_context_start_before: session.active_context_start_index,
            active_context_start_after: session.active_context_start_index,
            context_items: plan.context_items,
            context_sha256: Some(plan.context_sha256),
            request: Some(ModelRequestTrace {
                model: session.model.clone(),
                think: session.think,
                context_window: session.budget.context_window,
                max_output_tokens: session.budget.max_output_tokens,
            }),
            identity_instruction: Some(plan.identity_instruction),
            provenance_quality: ProvenanceQuality::Exact,
            retrieval: RetrievalTrace::default(),
            knowledge: KnowledgeTrace::default(),
            web: Default::default(),
        };
        turn.request_started_at = Some(started_at);
        turn.assistant_content = assistant.to_owned();
        turn.thinking = thinking.to_owned();
        turn.usage = TokenUsage::new(Some(42), Some(3));
        turn.status = TurnStatus::Complete;
        turn.done_reason = Some("stop".into());
        event_id(&session.id, Some(&turn.id), EventRole::Assistant)
    }

    #[test]
    fn replays_events_resolves_unicode_and_reconstructs_answer_context() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create(
                "model",
                "http://localhost",
                Some("system 原文"),
                Default::default(),
                true,
            )
            .unwrap();
        append_complete_turn(&mut session, "第一问", "第一答", "不得索引的 thinking");
        let second_answer =
            append_complete_turn(&mut session, "你a\u{301}🙂x", "第二答", "private");
        store.save(&mut session).unwrap();
        assert_eq!(
            store.retrieval().rebuild().unwrap(),
            SyncReport {
                sessions: 1,
                events: 5,
                spans: 5,
                answer_contexts: 2,
                documents: 4,
            }
        );

        let events = store.retrieval().replay_session(&session.id).unwrap();
        assert_eq!(events.len(), 5);
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(
            events.iter().map(|event| event.role).collect::<Vec<_>>(),
            vec![
                EventRole::System,
                EventRole::User,
                EventRole::Assistant,
                EventRole::User,
                EventRole::Assistant,
            ]
        );
        assert_eq!(
            events[2].reply_to_event_id.as_deref(),
            Some(events[1].id.as_str())
        );
        assert_eq!(
            events[3].reply_to_event_id.as_deref(),
            Some(events[2].id.as_str())
        );
        assert_eq!(
            events[4].reply_to_event_id.as_deref(),
            Some(events[3].id.as_str())
        );
        assert_eq!(events[2].token_count, Some(3));
        assert_eq!(events[1].token_count, None);
        assert!(
            events
                .iter()
                .all(|event| !event.content.contains("thinking"))
        );

        let span = SourceSpan {
            event_id: events[3].id.clone(),
            start_char: 1,
            end_char: 4,
        };
        let resolved = store.retrieval().resolve_span(&span).unwrap();
        assert_eq!(resolved.content, "a\u{301}🙂");
        assert_eq!(resolved.content_sha256, content_sha256("a\u{301}🙂"));
        assert!(matches!(
            store.retrieval().resolve_span(&SourceSpan {
                event_id: events[3].id.clone(),
                start_char: 0,
                end_char: 99,
            }),
            Err(RetrievalError::InvalidSpan { .. })
        ));

        let answer = store.retrieval().answer_context(&second_answer).unwrap();
        assert_eq!(answer.provenance_quality, ProvenanceQuality::Exact);
        assert_eq!(answer.request.as_ref().unwrap().model, "model");
        assert_eq!(
            answer
                .items
                .iter()
                .map(|item| item.resolved.content.as_str())
                .collect::<Vec<_>>(),
            vec!["system 原文", "第一问", "第一答", "你a\u{301}🙂x"]
        );
    }

    #[test]
    fn keyword_recall_returns_exact_old_chinese_span_and_traces_exclusions() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create(
                "model",
                "http://localhost",
                Some("system"),
                Default::default(),
                false,
            )
            .unwrap();
        let old_user = "两年前唐波说他喜欢杭州，生日是2021年4月3日，偏好乌龙茶。";
        append_complete_turn(&mut session, old_user, "收到", "secret");
        for index in 0..20 {
            append_complete_turn(&mut session, &format!("无关消息{index}"), "无关回答", "");
        }
        store.save(&mut session).unwrap();
        let current = "请问唐波偏好什么？";
        let current_event = event_id(&session.id, Some("pending"), EventRole::User);
        let recall = store
            .retrieval()
            .keyword_recall(current, &current_event, &[], RetrievalConfig::default())
            .unwrap();
        assert!(recall.evidence.iter().any(|item| item.content == old_user));
        assert!(
            recall
                .trace
                .candidates
                .iter()
                .any(|candidate| candidate.bm25_score.is_finite())
        );
        assert!(
            recall
                .trace
                .selected_evidence
                .iter()
                .any(|item| item.kind == EvidenceKind::Core)
        );
    }

    #[test]
    fn keyword_recall_large_irrelevant_corpus_recovers_exact_fact_types() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create(
                "model",
                "http://localhost",
                Some("system"),
                Default::default(),
                false,
            )
            .unwrap();
        let facts = [
            ("唐波", "唐波是早期项目负责人。"),
            ("杭州", "会议地点明确在杭州西湖区。"),
            ("2021年4月3日", "签约日期是2021年4月3日。"),
            ("乌龙茶", "我的明确偏好是乌龙茶。"),
            ("蓝鲸", "明确事实：项目代号是蓝鲸。"),
        ];
        for (_, fact) in facts {
            append_complete_turn(&mut session, fact, "已记录", "");
        }
        for index in 0..205 {
            session.turns.push(Turn::pending(format!(
                "无关历史消息{index}：讨论天气和书籍"
            )));
        }
        store.save(&mut session).unwrap();
        for (query, exact) in facts {
            let recall = store
                .retrieval()
                .keyword_recall(
                    query,
                    "current",
                    &[],
                    RetrievalConfig {
                        candidate_limit: 64,
                        max_selected: 4,
                        evidence_char_budget: 1600,
                        expansion_char_budget: 0,
                    },
                )
                .unwrap();
            let hit = recall
                .evidence
                .iter()
                .find(|item| item.content == exact)
                .expect("old exact fact selected");
            assert_eq!(hit.selected.content_sha256, content_sha256(exact));
            let candidate = recall
                .trace
                .candidates
                .iter()
                .find(|candidate| candidate.selected && candidate.span == hit.selected.span)
                .unwrap();
            assert!(candidate.raw_rank <= 256 && candidate.bm25_score.is_finite());
            assert_eq!(candidate.reason, "selected_core");
        }
    }

    #[test]
    fn document_fragments_use_unicode_scalar_240_with_40_overlap() {
        let make = |len| StoredEvent {
            id: "evt".into(),
            session_id: "s".into(),
            turn_id: Some("t".into()),
            sequence: 1,
            role: EventRole::User,
            created_at: "now".into(),
            content: "🙂".repeat(len),
            content_sha256: String::new(),
            reply_to_event_id: None,
            token_count: None,
            turn_status: None,
            done_reason: None,
            error: None,
        };
        assert_eq!(document_spans(&make(240)).len(), 1);
        let spans_241 = document_spans(&make(241));
        assert_eq!(spans_241.len(), 3);
        assert_eq!(
            (spans_241[1].1.start_char, spans_241[1].1.end_char),
            (0, 240)
        );
        assert_eq!(spans_241[1].1.end_char - spans_241[1].1.start_char, 240);
        assert_eq!(spans_241[2].1.start_char, 200);
        assert_eq!(spans_241[1].1.end_char - spans_241[2].1.start_char, 40);
        let spans_440 = document_spans(&make(440));
        assert_eq!(spans_440.len(), 3);
        assert_eq!(spans_440[2].1.start_char, 200);
        assert_eq!(spans_440[2].1.end_char, 440);
        for (_, span) in spans_241.iter().skip(1) {
            assert!(span.end_char - span.start_char <= 240);
        }
        for (_, span) in spans_440.iter().skip(1) {
            assert!(span.end_char - span.start_char <= 240);
        }
        assert_eq!(spans_241.last().unwrap().1.end_char, 241);
        assert_eq!(spans_440.last().unwrap().1.end_char, 440);
    }

    #[test]
    fn rebuild_repairs_modified_and_deleted_derived_rows_and_refreshes_source_hash() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        let answer_id = append_complete_turn(&mut session, "原文", "回复", "secret");
        let source_path = store.save(&mut session).unwrap();

        {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            connection
                .execute(
                    "UPDATE events SET content = 'tampered' WHERE event_id = ?1",
                    [&answer_id],
                )
                .unwrap();
        }
        assert!(matches!(
            store.retrieval().get_event(&answer_id),
            Err(RetrievalError::CorruptIndex(_))
        ));
        store.retrieval().rebuild().unwrap();
        assert_eq!(
            store.retrieval().get_event(&answer_id).unwrap().content,
            "回复"
        );

        {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            connection
                .execute("DELETE FROM events WHERE event_id = ?1", [&answer_id])
                .unwrap();
        }
        assert!(matches!(
            store.retrieval().get_event(&answer_id),
            Err(RetrievalError::EventNotFound(_))
        ));
        store.retrieval().rebuild().unwrap();
        assert_eq!(
            store.retrieval().get_event(&answer_id).unwrap().content,
            "回复"
        );

        let raw = fs::read_to_string(&source_path).unwrap();
        fs::write(&source_path, format!("{raw} ")).unwrap();
        assert!(matches!(
            store.retrieval().get_session(&session.id),
            Err(RetrievalError::StaleIndex { .. })
        ));
        store.retrieval().rebuild().unwrap();
        assert_eq!(
            store.retrieval().get_session(&session.id).unwrap().id,
            session.id
        );
    }

    #[test]
    fn migrates_real_v1_index_transactionally_with_wal() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create(
                "model",
                "http://localhost",
                Some("system"),
                Default::default(),
                false,
            )
            .unwrap();
        let answer_id = append_complete_turn(
            &mut session,
            &format!("{}杭州唐波", "甲".repeat(241)),
            "原始回复",
            "",
        );
        store.save(&mut session).unwrap();
        let expected = store.retrieval().answer_context(&answer_id).unwrap();
        {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            connection.execute_batch("DROP TABLE retrieval_documents_fts; DROP TABLE retrieval_documents; DROP TABLE retrieval_runs; PRAGMA user_version=1;").unwrap();
        }
        let migrated = RetrievalStore::new(root.path()).unwrap();
        let replay = migrated.replay_session(&session.id).unwrap();
        assert!(!replay.is_empty());
        let connection = Connection::open(migrated.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "wal"
        );
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            INDEX_SCHEMA_VERSION
        );
        assert!(
            connection
                .query_row("SELECT count(*) FROM retrieval_documents", [], |r| r
                    .get::<_, i64>(0))
                .unwrap()
                >= 3
        );
        let recall = migrated
            .keyword_recall("唐波", "current", &[], RetrievalConfig::default())
            .unwrap();
        assert!(
            recall
                .evidence
                .iter()
                .any(|item| item.content.contains("杭州唐波"))
        );
        let restored = migrated.answer_context(&answer_id).unwrap();
        assert_eq!(restored.context_sha256, expected.context_sha256);
        assert_eq!(
            restored
                .items
                .iter()
                .map(|item| &item.resolved.content)
                .collect::<Vec<_>>(),
            expected
                .items
                .iter()
                .map(|item| &item.resolved.content)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn unknown_index_version_errors_before_v2_ddl() {
        let root = tempfile::tempdir().unwrap();
        let index = root.path().join(INDEX_FILENAME);
        let connection = Connection::open(&index).unwrap();
        connection
            .pragma_update(None, "user_version", 99_i64)
            .unwrap();
        drop(connection);
        let store = RetrievalStore::new(root.path()).unwrap();
        assert!(matches!(
            store.replay_session("none"),
            Err(RetrievalError::UnsupportedIndexVersion(99))
        ));
        let connection = Connection::open(index).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_master WHERE name='retrieval_documents'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn resync_rejects_nonempty_event_content_mutation() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        let answer = append_complete_turn(&mut session, "原文甲", "回复", "");
        let path = store.save(&mut session).unwrap();
        let before = store.retrieval().get_event(&answer).unwrap();
        session.turns[0].assistant_content = "篡改甲".into();
        fs::write(&path, serde_json::to_vec(&session).unwrap()).unwrap();
        assert!(matches!(
            store.retrieval().sync_session(&session, &path),
            Err(RetrievalError::InvalidSource { .. })
        ));
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT content FROM events WHERE event_id=?1",
                    [&answer],
                    |r| r.get::<_, String>(0)
                )
                .unwrap(),
            before.content
        );
    }

    #[test]
    fn resync_rejects_obsolete_missing_event() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "一", "甲", "");
        let second = append_complete_turn(&mut session, "二", "乙", "");
        let path = store.save(&mut session).unwrap();
        session.turns.pop();
        fs::write(&path, serde_json::to_vec(&session).unwrap()).unwrap();
        assert!(matches!(
            store.retrieval().sync_session(&session, &path),
            Err(RetrievalError::InvalidSource { .. })
        ));
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT content FROM events WHERE event_id=?1",
                    [&second],
                    |r| r.get::<_, String>(0)
                )
                .unwrap(),
            "乙"
        );
    }

    #[test]
    fn resync_allows_empty_assistant_to_first_terminal_content() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        let mut turn = Turn::pending("问题".into());
        turn.request_started_at = Some(utc_now());
        turn.context_trace.provenance_quality = ProvenanceQuality::LegacyInferred;
        session.turns.push(turn);
        let path = store.save(&mut session).unwrap();
        let answer = event_id(
            &session.id,
            Some(&session.turns[0].id),
            EventRole::Assistant,
        );
        session.turns[0].assistant_content = "首次完成".into();
        session.turns[0].status = TurnStatus::Complete;
        session.turns[0].context_trace.provenance_quality = ProvenanceQuality::LegacyInferred;
        fs::write(&path, serde_json::to_vec(&session).unwrap()).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        assert_eq!(
            store.retrieval().get_event(&answer).unwrap().content,
            "首次完成"
        );
        assert_eq!(
            store
                .retrieval()
                .resolve_span(&SourceSpan {
                    event_id: answer.clone(),
                    start_char: 0,
                    end_char: "首次完成".chars().count()
                })
                .unwrap()
                .content,
            "首次完成"
        );
        session.turns[0].assistant_content = "二次篡改".into();
        fs::write(&path, serde_json::to_vec(&session).unwrap()).unwrap();
        assert!(matches!(
            store.retrieval().sync_session(&session, &path),
            Err(RetrievalError::InvalidSource { .. })
        ));
    }

    #[test]
    fn external_answer_context_survives_source_append_resync() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut a = store
            .create(
                "model",
                "http://localhost",
                Some("a-system"),
                Default::default(),
                false,
            )
            .unwrap();
        let external_answer = append_complete_turn(&mut a, "A问题", "A原始证据", "");
        store.save(&mut a).unwrap();
        let mut b = store
            .create(
                "model",
                "http://localhost",
                Some("b-system"),
                Default::default(),
                false,
            )
            .unwrap();
        let mut turn = Turn::pending("B问题".into());
        turn.request_started_at = Some(utc_now());
        turn.assistant_content = "B回答".into();
        turn.status = TurnStatus::Complete;
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: b.system_prompt.clone(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "A原始证据".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "B问题".into(),
            },
        ];
        turn.context_trace = ContextTrace {
            context_items: vec![
                ContextItemTrace {
                    role: EventRole::System,
                    span: SourceSpan {
                        event_id: event_id(&b.id, None, EventRole::System),
                        start_char: 0,
                        end_char: b.system_prompt.chars().count(),
                    },
                    content_sha256: content_sha256(&b.system_prompt),
                },
                ContextItemTrace {
                    role: EventRole::Assistant,
                    span: SourceSpan {
                        event_id: external_answer.clone(),
                        start_char: 0,
                        end_char: "A原始证据".chars().count(),
                    },
                    content_sha256: content_sha256("A原始证据"),
                },
                ContextItemTrace {
                    role: EventRole::User,
                    span: SourceSpan {
                        event_id: event_id(&b.id, Some(&turn.id), EventRole::User),
                        start_char: 0,
                        end_char: "B问题".chars().count(),
                    },
                    content_sha256: content_sha256("B问题"),
                },
            ],
            context_sha256: Some(context_sha256(&messages)),
            request: Some(ModelRequestTrace {
                model: b.model.clone(),
                think: false,
                context_window: b.budget.context_window,
                max_output_tokens: b.budget.max_output_tokens,
            }),
            provenance_quality: ProvenanceQuality::Exact,
            ..Default::default()
        };
        b.turns.push(turn);
        store.save(&mut b).unwrap();
        let b_answer = event_id(&b.id, Some(&b.turns[0].id), EventRole::Assistant);
        let before = store.retrieval().answer_context(&b_answer).unwrap();
        append_complete_turn(&mut a, "新增", "A新增", "");
        store.save(&mut a).unwrap();
        let after = store.retrieval().answer_context(&b_answer).unwrap();
        assert_eq!(before.context_sha256, after.context_sha256);
        assert_eq!(
            before.items[1].resolved.content,
            after.items[1].resolved.content
        );
    }

    #[test]
    fn rebuild_materializes_cross_session_answers_after_all_events() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut a = Session::new(
            "z-source".into(),
            "model".into(),
            "http://localhost".into(),
            "a".into(),
            Default::default(),
            false,
        )
        .unwrap();
        let external = append_complete_turn(&mut a, "A问", "A证据", "");
        store.save(&mut a).unwrap();
        let mut b = Session::new(
            "a-dependent".into(),
            "model".into(),
            "http://localhost".into(),
            "b".into(),
            Default::default(),
            false,
        )
        .unwrap();
        let mut turn = Turn::pending("B问".into());
        turn.request_started_at = Some(utc_now());
        turn.assistant_content = "B答".into();
        turn.status = TurnStatus::Complete;
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "b".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "A证据".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "B问".into(),
            },
        ];
        turn.context_trace = ContextTrace {
            context_items: vec![
                ContextItemTrace {
                    role: EventRole::System,
                    span: SourceSpan {
                        event_id: event_id(&b.id, None, EventRole::System),
                        start_char: 0,
                        end_char: 1,
                    },
                    content_sha256: content_sha256("b"),
                },
                ContextItemTrace {
                    role: EventRole::Assistant,
                    span: SourceSpan {
                        event_id: external,
                        start_char: 0,
                        end_char: 3,
                    },
                    content_sha256: content_sha256("A证据"),
                },
                ContextItemTrace {
                    role: EventRole::User,
                    span: SourceSpan {
                        event_id: event_id(&b.id, Some(&turn.id), EventRole::User),
                        start_char: 0,
                        end_char: 2,
                    },
                    content_sha256: content_sha256("B问"),
                },
            ],
            context_sha256: Some(context_sha256(&messages)),
            request: Some(ModelRequestTrace {
                model: "model".into(),
                think: false,
                context_window: b.budget.context_window,
                max_output_tokens: b.budget.max_output_tokens,
            }),
            provenance_quality: ProvenanceQuality::Exact,
            ..Default::default()
        };
        b.turns.push(turn);
        store.save(&mut b).unwrap();
        let answer = event_id(&b.id, Some(&b.turns[0].id), EventRole::Assistant);
        let before = store.retrieval().answer_context(&answer).unwrap();
        let report = store.retrieval().rebuild().unwrap();
        assert_eq!(report.sessions, 2);
        let after = store.retrieval().answer_context(&answer).unwrap();
        assert_eq!(after.context_sha256, before.context_sha256);
        assert_eq!(
            after
                .items
                .iter()
                .map(|i| &i.resolved.content)
                .collect::<Vec<_>>(),
            before
                .items
                .iter()
                .map(|i| &i.resolved.content)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn only_real_or_legacy_model_requests_create_assistant_events() {
        let mut session = Session::new(
            "session".into(),
            "model".into(),
            "http://localhost".into(),
            "system".into(),
            Default::default(),
            false,
        )
        .unwrap();
        let mut blocked = Turn::pending("blocked".into());
        blocked.status = TurnStatus::Blocked;
        session.turns.push(blocked);
        let mut preparation_failed = Turn::pending("preparation failed".into());
        preparation_failed.status = TurnStatus::Failed;
        session.turns.push(preparation_failed);
        for status in [
            TurnStatus::Complete,
            TurnStatus::Truncated,
            TurnStatus::Interrupted,
            TurnStatus::NoAnswer,
            TurnStatus::Failed,
        ] {
            let mut requested = Turn::pending(format!("requested {}", status.as_str()));
            requested.request_started_at = Some(utc_now());
            requested.context_trace.provenance_quality = ProvenanceQuality::LegacyInferred;
            requested.status = status;
            session.turns.push(requested);
        }

        let events = derive_events(&session);
        assert_eq!(
            events
                .iter()
                .filter(|event| event.role == EventRole::Assistant)
                .count(),
            5
        );
        assert_eq!(events.last().unwrap().content, "");
    }
}
