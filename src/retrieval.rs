use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::time::{Duration, Instant};

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use rusqlite::{
    Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior, params,
    types::ValueRef,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::config::{MemoryBudgetConfig, MemoryConfig};
use crate::consolidation::{
    consolidation_ledger_snapshot, normalize_match, original_claim_valid_to_by_id,
    prepare_consolidation_replay, replay_prepared_consolidation,
    restore_compatible_legacy_watermarks, snapshot_compatible_legacy_watermarks,
    validate_consolidation_ledger_semantics, validate_full_derived_integrity,
};
use crate::context::{WrappedHistoryCursor, wrapped_history_identity};
use crate::control::{ControlLog, ControlState};
use crate::episode::{
    EMBEDDING_COSINE_SIMILARITY_THRESHOLD, EPISODE_ALGORITHM_VERSION, EpisodeBoundaryDecision,
    EpisodeBoundarySuggestion, EpisodeDocument, EpisodeInputMessage, EpisodeMaterializationReport,
    EpisodeMember, EpisodePlanInput, aggregate_members_hash, ledger_snapshot_hash, plan_episodes,
    session_document_id,
};
use crate::graph::GraphRecallSeed;
use crate::knowledge::{KnowledgeStore, KnowledgeTrace};
use crate::model::{
    BudgetAllocationTrace, ChannelTrace, ChatMessage, ContextItemTrace, EntityMatchTrace,
    EventRole, EvidenceKind, FusionCandidateTrace, ModelRequestTrace, ProvenanceQuality, QueryKind,
    RankedCandidate, RetrievalChannel, RetrievalConfig, RetrievalDocumentGranularity,
    RetrievalTrace, SCHEMA_VERSION, SelectedEvidence, Session, SourceSpan, StateSelectionTrace,
    Turn, TurnStatus, WebTrace, content_sha256, context_sha256, event_id,
};
use crate::ollama::{ChatBackend, EmbeddingRequest};
use crate::vector::{
    EmbeddingCoverage, EmbeddingWrite, HnswVectorIndex, StoredEmbedding, VectorIndexSpec,
    decode_f32_le, encode_f32_le, l2_normalize,
};

pub const INDEX_FILENAME: &str = ".hippocampus-index.sqlite3";
const INDEX_SCHEMA_VERSION: i64 = 7;
const MEMORY_STATE_SCHEMA_VERSION: i64 = 4;
const DEFERRED_HARD_LIMIT: usize = usize::MAX;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct RecallChannels {
    pub bm25: bool,
    pub vector: bool,
    pub entity: bool,
    pub state: bool,
    pub episode: bool,
    pub graph: bool,
}

impl RecallChannels {
    pub const fn all() -> Self {
        Self {
            bm25: true,
            vector: true,
            entity: true,
            state: true,
            episode: true,
            graph: true,
        }
    }

    pub const fn bm25_only() -> Self {
        Self {
            bm25: true,
            vector: false,
            entity: false,
            state: false,
            episode: false,
            graph: false,
        }
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if !(self.bm25 || self.vector || self.entity || self.state || self.episode || self.graph) {
            return Err("at least one recall channel must be enabled");
        }
        if (self.entity || self.state || self.episode || self.graph) && !self.vector {
            return Err("entity, state, episode, and graph recall require vector recall");
        }
        Ok(())
    }
}

impl Default for RecallChannels {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecallQueryOrigin {
    #[default]
    IndexedEvent,
    Synthetic {
        reference_time: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct HybridRecallOptions {
    pub channels: RecallChannels,
    pub query_origin: RecallQueryOrigin,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RecallVisibility {
    cutoff: Option<DateTime<Utc>>,
    allow_aggregate_documents: bool,
}

impl RecallVisibility {
    fn from_options(options: &HybridRecallOptions) -> RetrievalResult<Self> {
        let cutoff = match &options.query_origin {
            RecallQueryOrigin::IndexedEvent => None,
            RecallQueryOrigin::Synthetic { reference_time } => Some(parse_retrieval_time(
                reference_time,
                "synthetic reference_time",
            )?),
        };
        Ok(Self {
            cutoff,
            allow_aggregate_documents: options.channels.episode,
        })
    }

    fn event_created_at_is_visible(&self, created_at: &str) -> RetrievalResult<bool> {
        self.cutoff.map_or(Ok(true), |cutoff| {
            parse_retrieval_time(created_at, "event.created_at").map(|time| time <= cutoff)
        })
    }

    fn document_is_visible(
        &self,
        store: &RetrievalStore,
        connection: &Connection,
        document_id: &str,
        granularity: RetrievalDocumentGranularity,
    ) -> RetrievalResult<bool> {
        if matches!(
            granularity,
            RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session
        ) && !self.allow_aggregate_documents
        {
            return Ok(false);
        }
        let Some(cutoff) = self.cutoff else {
            return Ok(true);
        };
        let mut statement = connection
            .prepare(
                "SELECT e.created_at FROM memory_document_members m
                 JOIN events e ON e.event_id=m.event_id
                 WHERE m.document_id=?1 ORDER BY m.ordinal",
            )
            .map_err(|error| store.database_error(error))?;
        let rows = statement
            .query_map([document_id], |row| row.get::<_, String>(0))
            .map_err(|error| store.database_error(error))?;
        let mut count = 0usize;
        for row in rows {
            count += 1;
            if parse_retrieval_time(
                &row.map_err(|error| store.database_error(error))?,
                "event.created_at",
            )? > cutoff
            {
                return Ok(false);
            }
        }
        if count == 0 {
            return Err(RetrievalError::CorruptIndex(format!(
                "文档 {document_id} 缺少 ordered members"
            )));
        }
        Ok(true)
    }

    pub(crate) fn cutoff(&self) -> Option<DateTime<Utc>> {
        self.cutoff
    }

    pub(crate) fn allows_aggregate_documents(&self) -> bool {
        self.allow_aggregate_documents
    }
}

fn final_hard_limit(value: usize) -> Option<usize> {
    (value != DEFERRED_HARD_LIMIT).then_some(value)
}

pub fn classify_query(raw_query: &str) -> QueryKind {
    let normalized = normalize_match(raw_query);
    if normalized.is_empty() {
        return QueryKind::GeneralSemantic;
    }

    if contains_any(&normalized, TEMPORAL_CHINESE)
        || contains_english_cue(&normalized, TEMPORAL_ENGLISH)
        || contains_ascii_date(&normalized)
    {
        QueryKind::TemporalState
    } else if contains_any(&normalized, MULTI_HOP_CHINESE)
        || contains_english_cue(&normalized, MULTI_HOP_ENGLISH)
    {
        QueryKind::MultiHop
    } else if contains_any(&normalized, EVENT_RECAP_CHINESE)
        || contains_english_cue(&normalized, EVENT_RECAP_ENGLISH)
    {
        QueryKind::EventRecap
    } else if contains_any(&normalized, EXACT_FACT_CHINESE)
        || contains_english_cue(&normalized, EXACT_FACT_ENGLISH)
        || normalized.chars().any(|character| character.is_numeric())
        || contains_paired_quote(&normalized)
    {
        QueryKind::ExactFact
    } else {
        QueryKind::GeneralSemantic
    }
}

const TEMPORAL_CHINESE: &[&str] = &[
    "现在",
    "目前",
    "当前",
    "最新",
    "截至",
    "什么时候",
    "何时",
    "曾经",
    "以前",
    "过去",
    "原来",
    "后来",
    "最近",
    "更新后",
    "还在",
    "还叫",
    "还住",
    "当时",
];
const TEMPORAL_ENGLISH: &[&str] = &[
    "now",
    "currently",
    "current",
    "latest",
    "as of",
    "when",
    "formerly",
    "previously",
    "before",
    "after",
    "recently",
    "used to",
    "still",
];
const MULTI_HOP_CHINESE: &[&str] = &[
    "关系",
    "关联",
    "共同",
    "都有哪些",
    "两者",
    "彼此",
    "为什么",
    "原因",
    "因果",
    "比较",
    "区别",
    "通过谁",
    "如何联系",
];
const MULTI_HOP_ENGLISH: &[&str] = &[
    "relationship",
    "related",
    "connection",
    "both",
    "common",
    "why",
    "because",
    "cause",
    "compare",
    "difference",
    "through whom",
    "how are",
];
const EVENT_RECAP_CHINESE: &[&str] = &[
    "回顾",
    "聊过什么",
    "说过什么",
    "发生了什么",
    "做了什么",
    "那次",
    "过程",
    "经过",
];
const EVENT_RECAP_ENGLISH: &[&str] = &[
    "recap",
    "what happened",
    "what did we",
    "discussed before",
    "conversation recap",
    "remember when",
];
const EXACT_FACT_CHINESE: &[&str] = &[
    "谁", "什么", "哪里", "哪儿", "哪个", "多少", "名称", "名字", "地址", "电话", "邮箱", "生日",
    "日期", "时间", "暗号", "编号", "数字", "精确", "具体",
];
const EXACT_FACT_ENGLISH: &[&str] = &[
    "who", "what", "where", "which", "how many", "name", "address", "phone", "email", "birthday",
    "date", "time", "code", "number", "exact", "specific",
];

fn contains_any(value: &str, cues: &[&str]) -> bool {
    cues.iter().any(|cue| value.contains(cue))
}

fn contains_english_cue(value: &str, cues: &[&str]) -> bool {
    cues.iter().any(|cue| {
        value.match_indices(cue).any(|(start, matched)| {
            let end = start + matched.len();
            ascii_word_boundary(value[..start].chars().next_back())
                && ascii_word_boundary(value[end..].chars().next())
        })
    })
}

fn ascii_word_boundary(character: Option<char>) -> bool {
    character.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
}

fn contains_ascii_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.windows(10).enumerate().any(|(start, candidate)| {
        let end = start + candidate.len();
        (start == 0 || !bytes[start - 1].is_ascii_digit())
            && (end == bytes.len() || !bytes[end].is_ascii_digit())
            && candidate[0..4].iter().all(u8::is_ascii_digit)
            && candidate[4] == b'-'
            && candidate[5..7].iter().all(u8::is_ascii_digit)
            && candidate[7] == b'-'
            && candidate[8..10].iter().all(u8::is_ascii_digit)
    })
}

fn contains_paired_quote(value: &str) -> bool {
    [
        ('"', '"'),
        ('\'', '\''),
        ('“', '”'),
        ('‘', '’'),
        ('「', '」'),
        ('『', '』'),
    ]
    .into_iter()
    .any(|(open, close)| {
        value
            .find(open)
            .is_some_and(|start| value[start + open.len_utf8()..].contains(close))
    })
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn latency_percentiles(mut values: Vec<u64>) -> MemoryLatencyPercentiles {
    values.sort_unstable();
    let nearest = |percent: usize| {
        (!values.is_empty()).then(|| {
            let rank = values.len().saturating_mul(percent).div_ceil(100).max(1);
            values[rank - 1]
        })
    };
    MemoryLatencyPercentiles {
        samples: values.len(),
        p50_ms: nearest(50),
        p95_ms: nearest(95),
    }
}

fn is_status_validation_error(error: &RetrievalError) -> bool {
    matches!(
        error,
        RetrievalError::CorruptIndex(_)
            | RetrievalError::StaleIndex { .. }
            | RetrievalError::ControlProjectionStale
            | RetrievalError::SessionNotFound(_)
            | RetrievalError::ExcludedSession(_)
    )
}

fn scoped_entity_count(
    store: &RetrievalStore,
    connection: &Connection,
    session_id: Option<&str>,
) -> RetrievalResult<usize> {
    let count: i64 = connection
        .query_row(
            "SELECT count(DISTINCT e.entity_id) FROM memory_entities e
             WHERE ?1 IS NULL OR e.created_session_id=?1
                OR EXISTS(SELECT 1 FROM memory_entity_mentions m WHERE m.entity_id=e.entity_id AND m.session_id=?1)
                OR EXISTS(SELECT 1 FROM memory_entity_aliases a WHERE a.entity_id=e.entity_id AND a.session_id=?1)
                OR EXISTS(SELECT 1 FROM memory_claims c JOIN memory_claim_evidence ce ON ce.claim_id=c.claim_id
                          WHERE ce.session_id=?1 AND (c.subject_entity_id=e.entity_id OR c.object_entity_id=e.entity_id))",
            [session_id],
            |row| row.get(0),
        )
        .map_err(|error| store.database_error(error))?;
    usize::try_from(count)
        .map_err(|_| RetrievalError::CorruptIndex("entity count 不是非负整数".into()))
}

fn scoped_episode_count(
    store: &RetrievalStore,
    connection: &Connection,
    session_id: Option<&str>,
) -> RetrievalResult<usize> {
    let count: i64 = connection
        .query_row(
            "SELECT count(DISTINCT d.document_id) FROM memory_documents d
             WHERE d.granularity='episode' AND (?1 IS NULL OR EXISTS(
                 SELECT 1 FROM memory_document_members m JOIN events e ON e.event_id=m.event_id
                 WHERE m.document_id=d.document_id AND e.session_id=?1))",
            [session_id],
            |row| row.get(0),
        )
        .map_err(|error| store.database_error(error))?;
    usize::try_from(count)
        .map_err(|_| RetrievalError::CorruptIndex("episode count 不是非负整数".into()))
}

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
    #[error("记忆派生状态版本不受支持：{0}")]
    UnsupportedMemoryStateVersion(i64),
    #[error("记忆图版本不受支持：{0}")]
    UnsupportedGraphSchemaVersion(i64),
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
    #[error("会话 {0} 已被控制日志排除")]
    ExcludedSession(String),
    #[error("事件 {0} 已被控制日志排除")]
    ExcludedEvent(String),
    #[error("control_projection_stale")]
    ControlProjectionStale,
    #[error("control_state_changed; retry the operation")]
    ControlStateChanged,
    #[error("控制日志校验失败，拒绝访问：{0}")]
    Control(String),
    #[error("{kind} embedding catalog 已变化，请重新获取 snapshot")]
    EmbeddingCatalogStale { kind: String },
    #[error("会话根目录锁已损坏，拒绝继续访问：{path}")]
    RootLockPoisoned { path: PathBuf },
    #[error("向量索引缓存锁已损坏")]
    VectorCachePoisoned,
    #[error("{0}")]
    HybridRecall(#[from] HybridRecallFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridRecallStage {
    Vector,
    EntityState,
    Graph,
}

#[derive(Debug, Error)]
#[error("{cause}")]
pub struct HybridRecallFailure {
    stage: HybridRecallStage,
    cause: String,
    elapsed_ms: u64,
}

impl HybridRecallFailure {
    fn new(stage: HybridRecallStage, error: impl std::fmt::Display) -> Self {
        Self {
            stage,
            cause: error.to_string(),
            elapsed_ms: 0,
        }
    }

    fn timed(stage: HybridRecallStage, error: impl std::fmt::Display, started: Instant) -> Self {
        Self {
            stage,
            cause: error.to_string(),
            elapsed_ms: elapsed_ms(started),
        }
    }
}

pub type RetrievalResult<T> = std::result::Result<T, RetrievalError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryLatencyPercentiles {
    pub samples: usize,
    pub p50_ms: Option<u64>,
    pub p95_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryAttemptCounts {
    pub applied: usize,
    pub rejected: usize,
    pub model_error: usize,
    pub cancelled: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryStatusMetrics {
    pub active_sessions: usize,
    pub active_events: usize,
    pub embedding_total: usize,
    pub embedding_compatible: usize,
    pub embedding_stale: usize,
    pub pending_consolidation_events: usize,
    pub entity_count: usize,
    pub episode_count: usize,
    pub graph_current: bool,
    pub graph_node_count: Option<usize>,
    pub graph_edge_count: Option<usize>,
    pub consolidation_attempts: MemoryAttemptCounts,
    pub consolidation_latency_ms: MemoryLatencyPercentiles,
    pub retrieval_runs: usize,
    pub retrieval_failures: usize,
    pub retrieval_latency_ms: MemoryLatencyPercentiles,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryStatus {
    pub session_id: Option<String>,
    pub memory_enabled: bool,
    pub healthy: bool,
    pub projection_current: bool,
    pub expected_control_generation_sha256: String,
    pub indexed_control_generation_sha256: Option<String>,
    pub validation_error: Option<String>,
    pub metrics: Option<MemoryStatusMetrics>,
}

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RebuildOptions {
    pub reuse_compatible_embeddings: bool,
}

impl Default for RebuildOptions {
    fn default() -> Self {
        Self {
            reuse_compatible_embeddings: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RebuildReport {
    pub sync: SyncReport,
    pub control_generation_sha256: String,
    pub ledger_attempts_preserved: usize,
    pub consolidation_attempts_replayed: usize,
    pub consolidation_attempts_skipped_inactive: usize,
    pub consolidation_attempts_skipped_dependency: usize,
    pub embeddings_reused: usize,
}

#[derive(Debug, Clone)]
struct ReusableLeafEmbedding {
    document_id: String,
    session_id: String,
    granularity: String,
    document_source_sha256: String,
    model: String,
    dimensions: i64,
    embedding_source_sha256: String,
    index_fingerprint: String,
    vector_blob: Vec<u8>,
    embedded_at: String,
}

fn repair_row_text(row: &rusqlite::Row<'_>, index: usize) -> Option<String> {
    match row.get_ref(index).ok()? {
        ValueRef::Text(bytes) => std::str::from_utf8(bytes).ok().map(ToOwned::to_owned),
        _ => None,
    }
}

fn repair_row_i64(row: &rusqlite::Row<'_>, index: usize) -> Option<i64> {
    match row.get_ref(index).ok()? {
        ValueRef::Integer(value) => Some(value),
        _ => None,
    }
}

fn repair_row_blob(row: &rusqlite::Row<'_>, index: usize) -> Option<Vec<u8>> {
    match row.get_ref(index).ok()? {
        ValueRef::Blob(bytes) => Some(bytes.to_vec()),
        _ => None,
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingPublishReport {
    pub documents: usize,
    pub reused: usize,
    pub changed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LeafEmbeddingDocument {
    pub document_id: String,
    pub session_id: String,
    pub granularity: RetrievalDocumentGranularity,
    pub source_sha256: String,
    pub content: String,
    pub source_event_id: String,
    pub start_char: usize,
    pub end_char: usize,
    pub message_document_id: String,
    pub reusable_vector: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LeafEmbeddingSnapshot {
    pub catalog_sha256: String,
    pub control_generation_sha256: String,
    pub session_ids: Vec<String>,
    pub documents: Vec<LeafEmbeddingDocument>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DirectMessageEmbedding {
    pub message_document_id: String,
    pub source_event_id: String,
    pub source_sha256: String,
    pub start_char: usize,
    pub end_char: usize,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateEmbeddingDocument {
    pub document_id: String,
    pub session_id: String,
    pub granularity: RetrievalDocumentGranularity,
    pub source_sha256: String,
    pub direct_messages: Vec<DirectMessageEmbedding>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateEmbeddingSnapshot {
    pub catalog_sha256: String,
    pub control_generation_sha256: String,
    pub documents: Vec<AggregateEmbeddingDocument>,
}

#[derive(Debug, Clone)]
pub struct RetrievalStore {
    root: PathBuf,
    index_path: PathBuf,
    root_lock: Arc<RwLock<()>>,
    vector_cache: Arc<Mutex<Option<VectorIndexCache>>>,
    #[cfg(test)]
    test_hooks: RetrievalTestHooks,
}

#[derive(Debug)]
struct VectorIndexCache {
    fingerprint: String,
    catalog_sha256: String,
    index: Arc<HnswVectorIndex>,
}

struct PendingVectorIndexCache {
    fingerprint: String,
    catalog_sha256: String,
    index: Arc<HnswVectorIndex>,
    observed_identity: Option<(String, String)>,
}

#[derive(Clone)]
struct ProjectedVectorCandidate {
    document_id: String,
    source_document_id: String,
    granularity: RetrievalDocumentGranularity,
    span: SourceSpan,
    role: EventRole,
    session_id: String,
    content_sha256: String,
    content: String,
    episode_id: Option<String>,
    vector_rank: usize,
    similarity: f64,
    contribution_divisor: usize,
    vector: Vec<f32>,
}

#[derive(Clone)]
struct FusedRawCandidate {
    pre_cap_rank: usize,
    document_id: String,
    granularity: RetrievalDocumentGranularity,
    span: SourceSpan,
    role: EventRole,
    session_id: String,
    content_sha256: String,
    content: String,
    episode_id: Option<String>,
    source_document_ids: Vec<String>,
    bm25_rank: Option<usize>,
    bm25_score: Option<f64>,
    bm25_contribution: f64,
    vector_rank: Option<usize>,
    vector_score: Option<f64>,
    vector_contribution: f64,
    vector_source_document_id: Option<String>,
    rrf_score: f64,
    protected_exact: bool,
    selected: bool,
    reason: String,
    vector: Option<Vec<f32>>,
}

struct StateEvidenceCandidate {
    trace: StateSelectionTrace,
    content: String,
    content_sha256: String,
    episode_id: Option<String>,
    conflict_group: Vec<String>,
}

struct StateSidecar {
    entity_matches: Vec<EntityMatchTrace>,
    candidates: Vec<StateEvidenceCandidate>,
    warnings: Vec<String>,
    entity_ms: u64,
    state_ms: u64,
}

struct GraphEvidenceCandidate {
    path_index: usize,
    raw: ProjectedVectorCandidate,
}

struct GraphSidecar {
    paths: Vec<crate::model::GraphPathTrace>,
    candidates: Vec<GraphEvidenceCandidate>,
    candidate_count: usize,
    aggregate_source_count: usize,
    elapsed_ms: u64,
    warning: Option<String>,
}

struct ClaimEvidenceSelection {
    evidence_id: String,
    span: SourceSpan,
    role: EventRole,
    content: String,
    content_sha256: String,
}

struct RecallTransition {
    to_state: String,
    reason: String,
    related_claim_id: Option<String>,
    created_at: DateTime<Utc>,
}

struct ClaimSnapshot {
    state: String,
    certainty: String,
    valid_to: Option<String>,
    related_claim_ids: Vec<String>,
}

enum RecallMoment {
    Indexed(Box<StoredEvent>),
    Synthetic(DateTime<Utc>),
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConsolidationHookPoint {
    AfterPendingBatchCheck,
    AfterTransactionSourceCheck,
}

#[cfg(test)]
pub(crate) type ConsolidationHook = Arc<dyn Fn(ConsolidationHookPoint) + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AggregateAuditPhase {
    PreWrite,
    FinalWrite,
    Read,
    MaterializeFinal,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum AggregateAuditHookPoint {
    DerivedIntegrity,
    Materialization {
        session_id: String,
        phase: AggregateAuditPhase,
    },
}

#[cfg(test)]
type AggregateAuditHook = Arc<dyn Fn(AggregateAuditHookPoint) + Send + Sync>;

#[cfg(test)]
#[derive(Clone, Default)]
struct RetrievalTestHooks {
    consolidation: Arc<Mutex<Option<ConsolidationHook>>>,
    aggregate_audit: Arc<Mutex<Option<AggregateAuditHook>>>,
}

#[cfg(test)]
impl std::fmt::Debug for RetrievalTestHooks {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RetrievalTestHooks(..)")
    }
}

static ROOT_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Weak<RwLock<()>>>>> = OnceLock::new();

fn shared_root_lock(key: &Path) -> RetrievalResult<Arc<RwLock<()>>> {
    let registry = ROOT_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut registry = registry
        .lock()
        .map_err(|_| RetrievalError::RootLockPoisoned {
            path: key.to_path_buf(),
        })?;
    registry.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = registry.get(key).and_then(Weak::upgrade) {
        return Ok(lock);
    }
    let lock = Arc::new(RwLock::new(()));
    registry.insert(key.to_path_buf(), Arc::downgrade(&lock));
    Ok(lock)
}

fn canonical_root_key(root: &Path) -> PathBuf {
    canonicalize_with_missing_suffix(root).unwrap_or_else(|| {
        let normalized = lexical_normalize_absolute(root);
        canonicalize_with_missing_suffix(&normalized).unwrap_or(normalized)
    })
}

fn canonicalize_with_missing_suffix(path: &Path) -> Option<PathBuf> {
    let mut missing = Vec::new();
    let mut ancestor = path.to_path_buf();
    while !ancestor.exists() {
        let Some(Component::Normal(name)) = ancestor.components().next_back() else {
            return None;
        };
        missing.push(name.to_os_string());
        if !ancestor.pop() {
            return None;
        }
    }
    let mut key = fs::canonicalize(&ancestor).ok()?;
    for name in missing.into_iter().rev() {
        key.push(name);
    }
    Some(key)
}

fn lexical_normalize_absolute(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !matches!(
                    normalized.components().next_back(),
                    Some(Component::RootDir)
                ) {
                    normalized.pop();
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
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
        let resolved_root = canonical_root_key(&root);
        if resolved_root.exists() && !resolved_root.is_dir() {
            return Err(RetrievalError::Io {
                path: resolved_root,
                source: std::io::Error::new(
                    std::io::ErrorKind::NotADirectory,
                    "session root is not a directory",
                ),
            });
        }
        fs::create_dir_all(&resolved_root).map_err(|source| RetrievalError::Io {
            path: resolved_root.clone(),
            source,
        })?;
        let root = fs::canonicalize(&resolved_root).map_err(|source| RetrievalError::Io {
            path: resolved_root,
            source,
        })?;
        if !root.is_dir() {
            return Err(RetrievalError::Io {
                path: root,
                source: std::io::Error::new(
                    std::io::ErrorKind::NotADirectory,
                    "session root is not a directory",
                ),
            });
        }
        let root_lock = shared_root_lock(&root)?;
        Ok(Self {
            index_path: root.join(INDEX_FILENAME),
            root,
            root_lock,
            vector_cache: Arc::new(Mutex::new(None)),
            #[cfg(test)]
            test_hooks: RetrievalTestHooks::default(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    pub fn control_state(&self) -> RetrievalResult<ControlState> {
        let _guard = self.acquire_root_read()?;
        self.replay_control_state_under_guard()
    }

    pub fn memory_status(
        &self,
        memory: &MemoryConfig,
        session_id: Option<&str>,
    ) -> RetrievalResult<MemoryStatus> {
        memory
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let _guard = self.acquire_root_read()?;
        let control = self.replay_control_state_under_guard()?;
        let expected = control.generation_sha256().to_owned();
        let mut connection = self.open_status_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|error| self.database_error(error))?;
        let marker = transaction
            .query_row(
                "SELECT value FROM memory_schema_meta WHERE key='active_control_generation_sha256'",
                [],
                |row| {
                    Ok(match row.get_ref(0)? {
                        ValueRef::Text(bytes) => std::str::from_utf8(bytes)
                            .map(str::to_owned)
                            .unwrap_or_else(|_| "<malformed-utf8>".into()),
                        _ => "<malformed-non-text>".into(),
                    })
                },
            )
            .optional()
            .map_err(|error| self.database_error(error))?;
        let marker_valid = marker.as_deref().is_none_or(is_lower_hex_sha256);
        let projection_current = marker_valid
            && marker
                .as_deref()
                .map_or(control.last_sequence() == 0, |actual| actual == expected);
        let mut status = MemoryStatus {
            session_id: session_id.map(str::to_owned),
            memory_enabled: memory.enabled,
            healthy: false,
            projection_current,
            expected_control_generation_sha256: expected,
            indexed_control_generation_sha256: marker.clone(),
            validation_error: None,
            metrics: None,
        };
        if !marker_valid {
            status.validation_error =
                Some("active_control_generation_sha256 marker 必须为 64 位小写十六进制".into());
        } else if !projection_current {
            status.validation_error = Some(match marker {
                Some(_) => "control projection generation 与当前控制日志不一致".into(),
                None => "非空控制日志缺少 control projection generation marker".into(),
            });
        } else {
            let metrics = (|| -> RetrievalResult<MemoryStatusMetrics> {
                validate_full_derived_integrity(&transaction)?;
                if let Some(scope) = session_id {
                    Self::require_active_session(&control, scope)?;
                    self.verify_indexed_session_source_projection_with_control(
                        &transaction,
                        scope,
                        &control,
                    )?;
                }
                let mut active_sessions = 0usize;
                let mut active_events = 0usize;
                let mut sessions = transaction
                    .prepare("SELECT session_id FROM indexed_sessions ORDER BY session_id")
                    .map_err(|error| self.database_error(error))?;
                for row in sessions
                    .query_map([], |row| row.get::<_, String>(0))
                    .map_err(|error| self.database_error(error))?
                {
                    let id = row.map_err(|error| self.database_error(error))?;
                    if control.allows_session(&id) && session_id.is_none_or(|scope| scope == id) {
                        active_sessions += 1;
                    }
                }
                let mut events = transaction
                    .prepare("SELECT event_id,session_id FROM events ORDER BY session_id,sequence")
                    .map_err(|error| self.database_error(error))?;
                for row in events
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })
                    .map_err(|error| self.database_error(error))?
                {
                    let (event_id, event_session) =
                        row.map_err(|error| self.database_error(error))?;
                    if control.allows_session(&event_session)
                        && control.allows_event(&event_session, &event_id)
                        && session_id.is_none_or(|scope| scope == event_session)
                    {
                        active_events += 1;
                    }
                }
                let spec = VectorIndexSpec::from_config(memory)
                    .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
                let fingerprint = embedding_fingerprint(&spec)?;
                let embedding_total =
                    self.active_memory_document_count(&transaction, &control, session_id)?;
                let embedding_compatible = self
                    .compatible_embeddings_from_connection_with_control(
                        &transaction,
                        &spec,
                        &fingerprint,
                        session_id,
                        &control,
                    )?
                    .len();
                let embedding_stale = embedding_total
                    .checked_sub(embedding_compatible)
                    .ok_or_else(|| {
                        RetrievalError::CorruptIndex(
                            "兼容 embedding 数超过 active document 数".into(),
                        )
                    })?;
                let pending_consolidation_events = self
                    .pending_consolidation_events_from_connection(
                        &transaction,
                        &control,
                        session_id,
                    )?;
                let entity_count = scoped_entity_count(self, &transaction, session_id)?;
                let episode_count = scoped_episode_count(self, &transaction, session_id)?;
                let graph_active_documents =
                    self.active_memory_document_count(&transaction, &control, None)?;
                let graph = crate::graph::graph_status_from_connection(
                    self,
                    &transaction,
                    memory,
                    &control,
                    graph_active_documents,
                    session_id,
                )?;
                let attempts = self.status_consolidation_attempts_from_connection(
                    &transaction,
                    &control,
                    session_id,
                )?;
                let mut consolidation_attempts = MemoryAttemptCounts::default();
                let mut consolidation_latencies = Vec::with_capacity(attempts.len());
                for attempt in attempts {
                    use crate::consolidation::ConsolidationAttemptStatus;
                    match attempt.status {
                        ConsolidationAttemptStatus::Applied => consolidation_attempts.applied += 1,
                        ConsolidationAttemptStatus::Rejected => {
                            consolidation_attempts.rejected += 1
                        }
                        ConsolidationAttemptStatus::ModelError => {
                            consolidation_attempts.model_error += 1
                        }
                        ConsolidationAttemptStatus::Cancelled => {
                            consolidation_attempts.cancelled += 1
                        }
                    }
                    consolidation_latencies.push(attempt.latency_ms);
                }
                consolidation_attempts.failed =
                    consolidation_attempts.rejected + consolidation_attempts.model_error;
                let mut retrieval_latencies = Vec::new();
                let mut retrieval_failures = 0usize;
                let mut retrieval_runs = 0usize;
                for (answer_event_id, trace) in self
                    .validated_retrieval_runs_from_connection_with_control(&transaction, &control)?
                {
                    if let Some(scope) = session_id {
                        let answer_session: String = transaction
                            .query_row(
                                "SELECT session_id FROM events WHERE event_id=?1",
                                [&answer_event_id],
                                |row| row.get(0),
                            )
                            .map_err(|error| self.database_error(error))?;
                        if answer_session != scope {
                            continue;
                        }
                    }
                    retrieval_runs += 1;
                    if trace.status != "ok"
                        || trace
                            .channels
                            .iter()
                            .any(|channel| channel.status == "error" || channel.error.is_some())
                    {
                        retrieval_failures += 1;
                    }
                    retrieval_latencies.push(trace.elapsed_ms);
                }
                Ok(MemoryStatusMetrics {
                    active_sessions,
                    active_events,
                    embedding_total,
                    embedding_compatible,
                    embedding_stale,
                    pending_consolidation_events,
                    entity_count,
                    episode_count,
                    graph_current: graph.current,
                    graph_node_count: graph.node_count,
                    graph_edge_count: graph.edge_count,
                    consolidation_attempts,
                    consolidation_latency_ms: latency_percentiles(consolidation_latencies),
                    retrieval_runs,
                    retrieval_failures,
                    retrieval_latency_ms: latency_percentiles(retrieval_latencies),
                })
            })();
            match metrics {
                Ok(metrics) => status.metrics = Some(metrics),
                Err(error) if is_status_validation_error(&error) => {
                    status.validation_error = Some(error.to_string());
                }
                Err(error) => return Err(error),
            }
        }
        if let Some(metrics) = status.metrics.as_ref() {
            status.healthy = status.projection_current
                && status.validation_error.is_none()
                && (!memory.enabled || (metrics.embedding_stale == 0 && metrics.graph_current));
        }
        self.require_unchanged_control_state(&control)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(status)
    }

    pub(crate) fn replay_control_state_under_guard(&self) -> RetrievalResult<ControlState> {
        ControlLog::new(&self.root)
            .and_then(|log| log.replay())
            .map_err(|error| RetrievalError::Control(error.to_string()))
    }

    pub(crate) fn require_unchanged_control_state(
        &self,
        expected: &ControlState,
    ) -> RetrievalResult<()> {
        if self.replay_control_state_under_guard()? != *expected {
            return Err(RetrievalError::ControlStateChanged);
        }
        Ok(())
    }

    pub(crate) fn control_projection_is_current(
        &self,
        connection: &Connection,
        state: &ControlState,
    ) -> RetrievalResult<bool> {
        let marker = connection
            .query_row(
                "SELECT value FROM memory_schema_meta WHERE key='active_control_generation_sha256'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| self.database_error(error))?;
        Ok(match marker {
            Some(marker) => marker == state.generation_sha256(),
            None => state.last_sequence() == 0,
        })
    }

    pub(crate) fn require_current_control_projection(
        &self,
        connection: &Connection,
        state: &ControlState,
    ) -> RetrievalResult<()> {
        if !self.control_projection_is_current(connection, state)? {
            return Err(RetrievalError::ControlProjectionStale);
        }
        Ok(())
    }

    pub(crate) fn verify_embedding_refresh_generation(
        &self,
        expected_generation_sha256: &str,
    ) -> RetrievalResult<()> {
        {
            let _guard = self.acquire_root_read()?;
            let state = self.replay_control_state_under_guard()?;
            if state.generation_sha256() != expected_generation_sha256 {
                return Err(RetrievalError::ControlStateChanged);
            }
            let connection = self.open_connection()?;
            self.require_current_control_projection(&connection, &state)?;
            self.require_unchanged_control_state(&state)?;
        }
        Ok(())
    }

    fn require_active_session(state: &ControlState, session_id: &str) -> RetrievalResult<()> {
        if !state.allows_session(session_id) {
            return Err(RetrievalError::ExcludedSession(session_id.to_owned()));
        }
        Ok(())
    }

    fn require_active_event(state: &ControlState, event: &StoredEvent) -> RetrievalResult<()> {
        Self::require_active_session(state, &event.session_id)?;
        if !state.allows_event(&event.session_id, &event.id) {
            return Err(RetrievalError::ExcludedEvent(event.id.clone()));
        }
        Ok(())
    }

    fn validate_recall_input_id(
        &self,
        connection: &Connection,
        state: &ControlState,
        event_id: &str,
        session_filter: Option<&str>,
    ) -> RetrievalResult<()> {
        if state.event_is_excluded(event_id) {
            return Err(RetrievalError::ExcludedEvent(event_id.to_owned()));
        }
        let event = connection
            .query_row(
                "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                        content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error
                 FROM events WHERE event_id=?1",
                [event_id],
                map_event,
            )
            .optional()
            .map_err(|error| self.database_error(error))?;
        let Some(event) = event else {
            let Some(authoritative) = self.authoritative_raw_event(event_id)? else {
                return Ok(());
            };
            Self::require_active_event(state, &authoritative)?;
            if session_filter.is_some_and(|scope| scope != authoritative.session_id) {
                return Err(RetrievalError::CorruptIndex(
                    "检索输入事件不属于限定会话".into(),
                ));
            }
            return Err(RetrievalError::StaleIndex {
                session_id: authoritative.session_id,
            });
        };
        Self::require_active_event(state, &event)?;
        if session_filter.is_some_and(|scope| scope != event.session_id) {
            return Err(RetrievalError::CorruptIndex(
                "检索输入事件不属于限定会话".into(),
            ));
        }
        self.authoritative_indexed_event(connection, event_id)?;
        Ok(())
    }

    fn validate_synthetic_sentinel(
        &self,
        connection: &Connection,
        event_id: &str,
    ) -> RetrievalResult<()> {
        let indexed = connection
            .query_row("SELECT 1 FROM events WHERE event_id=?1", [event_id], |_| {
                Ok(())
            })
            .optional()
            .map_err(|error| self.database_error(error))?;
        if indexed.is_some() || self.authoritative_raw_event(event_id)?.is_some() {
            return Err(RetrievalError::CorruptIndex(
                "synthetic current_user_event_id must not identify a real event".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn memory_document_is_active(
        &self,
        connection: &Connection,
        state: &ControlState,
        document_id: &str,
    ) -> RetrievalResult<bool> {
        let row = connection
            .query_row(
                "SELECT session_id,granularity,source_sha256,start_sequence,end_sequence,member_count
                 FROM memory_documents WHERE document_id=?1",
                [document_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        i64_to_usize(row.get(3)?)?,
                        i64_to_usize(row.get(4)?)?,
                        i64_to_usize(row.get(5)?)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| self.database_error(error))?;
        let Some((
            session_id,
            granularity,
            document_source,
            start_sequence,
            end_sequence,
            member_count,
        )) = row
        else {
            return Err(RetrievalError::CorruptIndex(format!(
                "派生文档 {document_id} 缺少 memory_documents 来源"
            )));
        };
        let mut active = state.allows_session(&session_id);
        let mut statement = connection
            .prepare(
                "SELECT m.ordinal,m.event_id,m.start_char,m.end_char,m.content_sha256,
                    e.session_id,e.sequence,s.content_sha256
             FROM memory_document_members m
             JOIN events e ON e.event_id=m.event_id
             JOIN source_spans s ON s.event_id=m.event_id
                AND s.start_char=m.start_char AND s.end_char=m.end_char
             WHERE m.document_id=?1 ORDER BY m.ordinal",
            )
            .map_err(|error| self.database_error(error))?;
        let members = statement
            .query_map([document_id], |row| {
                Ok((
                    i64_to_usize(row.get(0)?)?,
                    row.get::<_, String>(1)?,
                    i64_to_usize(row.get(2)?)?,
                    i64_to_usize(row.get(3)?)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    i64_to_usize(row.get(6)?)?,
                    row.get::<_, String>(7)?,
                ))
            })
            .map_err(|error| self.database_error(error))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|error| self.database_error(error))?;
        if members.len() != member_count
            || members.iter().enumerate().any(|(ordinal, member)| {
                member.0 != ordinal
                    || member.5 != session_id
                    || member.4 != member.7
                    || member.2 > member.3
            })
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "文档 {document_id} 的成员来源不规范"
            )));
        }
        if matches!(granularity.as_str(), "message" | "fragment")
            && (members.len() != 1
                || members[0].6 != start_sequence
                || members[0].6 != end_sequence
                || members[0].4 != document_source
                || document_id != format!("{}:{}:{}", members[0].1, members[0].2, members[0].3))
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "leaf 文档 {document_id} 的成员来源不规范"
            )));
        }
        if matches!(granularity.as_str(), "episode" | "session")
            && (members.first().map(|member| member.6) != Some(start_sequence)
                || members.last().map(|member| member.6) != Some(end_sequence)
                || members.windows(2).any(|pair| pair[0].6 >= pair[1].6))
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "aggregate 文档 {document_id} 的成员来源不规范"
            )));
        }
        if !matches!(
            granularity.as_str(),
            "message" | "fragment" | "episode" | "session"
        ) {
            return Err(RetrievalError::CorruptIndex(format!(
                "文档 {document_id} 的粒度无效"
            )));
        }
        for member in &members {
            if !state.allows_event(&session_id, &member.1) {
                active = false;
            }
        }
        Ok(active && !members.is_empty())
    }

    fn authoritative_indexed_session_source(
        &self,
        connection: &Connection,
        session_id: &str,
    ) -> RetrievalResult<SessionSource> {
        let indexed_session = connection
            .query_row(
                "SELECT session_id, title, created_at, updated_at, source_file, source_sha256,
                        source_schema_version
                 FROM indexed_sessions WHERE session_id=?1",
                [session_id],
                map_session,
            )
            .optional()
            .map_err(|error| self.database_error(error))?
            .ok_or_else(|| RetrievalError::CorruptIndex("派生来源事件缺少权威会话绑定".into()))?;
        self.verify_fresh(&indexed_session)?;
        let source = self.read_source(&self.root.join(&indexed_session.source_file))?;
        if source.sha256 != indexed_session.source_sha256
            || source.session.id != indexed_session.id
            || source.session.title != indexed_session.title
            || source.session.created_at != indexed_session.created_at
            || source.session.updated_at != indexed_session.updated_at
            || source.session.schema_version != indexed_session.source_schema_version
        {
            return Err(RetrievalError::CorruptIndex(
                "权威会话元数据与派生索引不一致".into(),
            ));
        }
        Ok(source)
    }

    fn authoritative_indexed_event(
        &self,
        connection: &Connection,
        event_id: &str,
    ) -> RetrievalResult<StoredEvent> {
        let indexed_event = connection
            .query_row(
                "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                        content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error
                 FROM events WHERE event_id=?1",
                [event_id],
                map_event,
            )
            .optional()
            .map_err(|error| self.database_error(error))?
            .ok_or_else(|| {
                RetrievalError::CorruptIndex("派生来源事件缺少权威索引绑定".into())
            })?;
        verify_event_hash(&indexed_event)?;
        let source =
            self.authoritative_indexed_session_source(connection, &indexed_event.session_id)?;
        let mut matches = derive_events(&source.session)
            .into_iter()
            .filter(|event| event.id == event_id);
        let authoritative = matches
            .next()
            .ok_or_else(|| RetrievalError::CorruptIndex("派生来源事件在权威会话中不存在".into()))?;
        if matches.next().is_some() || authoritative != indexed_event {
            return Err(RetrievalError::CorruptIndex(
                "派生来源事件与权威会话不一致".into(),
            ));
        }
        Ok(authoritative)
    }

    fn authoritative_raw_event(&self, event_id: &str) -> RetrievalResult<Option<StoredEvent>> {
        let mut matches = self
            .load_all_sources()?
            .into_iter()
            .flat_map(|source| derive_events(&source.session))
            .filter(|event| event.id == event_id);
        let authoritative = matches.next();
        if matches.next().is_some() {
            return Err(RetrievalError::CorruptIndex(
                "event ID 未唯一绑定权威 raw event".into(),
            ));
        }
        if let Some(event) = &authoritative {
            verify_event_hash(event)?;
        }
        Ok(authoritative)
    }

    fn authoritative_trace_query_event(
        &self,
        connection: &Connection,
        control: &ControlState,
        event_id: &str,
    ) -> RetrievalResult<StoredEvent> {
        let indexed = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM events WHERE event_id=?1)",
                [event_id],
                |row| row.get::<_, bool>(0),
            )
            .map_err(|error| self.database_error(error))?;
        if indexed {
            return self.authoritative_indexed_event(connection, event_id);
        }

        let authoritative = self.authoritative_raw_event(event_id)?.ok_or_else(|| {
            RetrievalError::CorruptIndex("entity trace current query 缺少权威 raw event".into())
        })?;
        if control.allows_event(&authoritative.session_id, &authoritative.id) {
            return Err(RetrievalError::CorruptIndex(
                "active entity trace current query 缺少 indexed event".into(),
            ));
        }
        Ok(authoritative)
    }

    fn authoritative_trace_aggregate_document(
        &self,
        connection: &Connection,
        control: &ControlState,
        document_id: &str,
        expected_session: &str,
    ) -> RetrievalResult<(EpisodeDocument, bool)> {
        let aggregate =
            load_persisted_aggregate_document(connection, document_id, expected_session)?;
        let source = self.authoritative_indexed_session_source(connection, expected_session)?;
        let mut authoritative_events = HashMap::new();
        for event in derive_events(&source.session) {
            if authoritative_events
                .insert(event.id.clone(), event)
                .is_some()
            {
                return Err(RetrievalError::CorruptIndex(
                    "权威会话包含重复事件 ID".into(),
                ));
            }
        }

        for member in &aggregate.members {
            let authoritative = authoritative_events.get(&member.event_id).ok_or_else(|| {
                RetrievalError::CorruptIndex("aggregate member 在权威会话中不存在".into())
            })?;
            let indexed = connection
                .query_row(
                    "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                            content_sha256, reply_to_event_id, token_count, turn_status,
                            done_reason, error
                     FROM events WHERE event_id=?1",
                    [&member.event_id],
                    map_event,
                )
                .optional()
                .map_err(|error| self.database_error(error))?
                .ok_or_else(|| {
                    RetrievalError::CorruptIndex("aggregate member 缺少权威索引事件".into())
                })?;
            verify_event_hash(&indexed)?;
            let end_char = authoritative.content.chars().count();
            let expected_document_id = format!("{}:0:{end_char}", authoritative.id);
            let sequence = u64::try_from(authoritative.sequence).map_err(|_| {
                RetrievalError::CorruptIndex("aggregate member sequence 溢出".into())
            })?;
            if indexed != *authoritative
                || authoritative.session_id != expected_session
                || !matches!(authoritative.role, EventRole::User | EventRole::Assistant)
                || authoritative.content.trim().is_empty()
                || member.document_id != expected_document_id
                || member.event_id != authoritative.id
                || member.sequence != sequence
                || member.role != authoritative.role
                || member.span.event_id != authoritative.id
                || member.span.start_char != 0
                || member.span.end_char != end_char
                || member.content_sha256 != authoritative.content_sha256
            {
                return Err(RetrievalError::CorruptIndex(
                    "aggregate member 与权威会话来源不一致".into(),
                ));
            }
        }

        let first_event_id = &aggregate
            .members
            .first()
            .expect("strict aggregate loader rejects empty members")
            .event_id;
        let expected_document_id = if aggregate.granularity == "episode" {
            trace_episode_document_id(expected_session, first_event_id)
        } else {
            session_document_id(expected_session)
        };
        if aggregate.document_id != expected_document_id {
            return Err(RetrievalError::CorruptIndex(
                "aggregate document ID 与权威成员来源不一致".into(),
            ));
        }
        let active = control.allows_session(expected_session)
            && aggregate
                .members
                .iter()
                .all(|member| control.allows_event(expected_session, &member.event_id));
        Ok((aggregate, active))
    }

    fn document_reference_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        document_id: &str,
    ) -> RetrievalResult<bool> {
        let raw = connection
            .query_row(
                "SELECT r.event_id,e.session_id,r.start_char,r.end_char,r.granularity,
                        r.content_sha256,r.exact_content
                 FROM retrieval_documents r
             LEFT JOIN events e ON e.event_id=r.event_id WHERE r.document_id=?1",
                [document_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        i64_to_usize(row.get(2)?)?,
                        i64_to_usize(row.get(3)?)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| self.database_error(error))?;
        if let Some((_, session_id, _, _, _, _, _)) = &raw
            && session_id.is_none()
        {
            return Err(RetrievalError::CorruptIndex(
                "派生文档缺少权威来源事件".into(),
            ));
        }
        let memory = connection.query_row(
            "SELECT session_id,granularity,source_sha256,start_sequence,end_sequence,member_count
             FROM memory_documents WHERE document_id=?1",
            [document_id],
            |row| Ok((
                row.get::<_, String>(0)?, row.get::<_, String>(1)?,
                row.get::<_, String>(2)?, i64_to_usize(row.get(3)?)?,
                i64_to_usize(row.get(4)?)?, i64_to_usize(row.get(5)?)?,
            )),
        ).optional().map_err(|error| self.database_error(error))?
            .ok_or_else(|| RetrievalError::CorruptIndex(
                "trace 文档引用缺少 memory_documents 来源".into()
            ))?;
        match memory.1.as_str() {
            "message" | "fragment" => {
                let memory_active =
                    self.memory_document_is_active(connection, control, document_id)?;
                let (event_id, session_id, start, end, granularity, hash, exact_content) = raw
                    .ok_or_else(|| {
                        RetrievalError::CorruptIndex(
                            "active leaf 文档缺少 retrieval_documents 来源".into(),
                        )
                    })?;
                let authoritative = self.authoritative_indexed_event(connection, &event_id)?;
                if authoritative.role == EventRole::System
                    || authoritative.content.trim().is_empty()
                {
                    return Err(RetrievalError::CorruptIndex(
                        "leaf 文档绑定了不可索引的权威事件".into(),
                    ));
                }
                let expected = document_spans(&authoritative)
                    .into_iter()
                    .filter(|(_, span)| {
                        document_id
                            == format!("{}:{}:{}", authoritative.id, span.start_char, span.end_char)
                    })
                    .collect::<Vec<_>>();
                if expected.len() != 1 {
                    return Err(RetrievalError::CorruptIndex(
                        "leaf 文档不是权威事件的 canonical 文档".into(),
                    ));
                }
                let (expected_granularity, expected_span) = expected
                    .into_iter()
                    .next()
                    .expect("checked one canonical leaf document");
                let expected_granularity = match expected_granularity {
                    RetrievalDocumentGranularity::Message => "message",
                    RetrievalDocumentGranularity::Fragment => "fragment",
                    RetrievalDocumentGranularity::Episode
                    | RetrievalDocumentGranularity::Session => {
                        unreachable!("document_spans only derives leaf documents")
                    }
                };
                let expected_content = slice_chars(&authoritative.content, &expected_span)?;
                let expected_hash = content_sha256(&expected_content);
                let member = connection
                    .query_row(
                        "SELECT event_id,start_char,end_char,content_sha256
                     FROM memory_document_members WHERE document_id=?1 AND ordinal=0",
                        [document_id],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                i64_to_usize(row.get(1)?)?,
                                i64_to_usize(row.get(2)?)?,
                                row.get::<_, String>(3)?,
                            ))
                        },
                    )
                    .map_err(|error| self.database_error(error))?;
                if session_id.as_deref() != Some(memory.0.as_str())
                    || authoritative.session_id != memory.0
                    || event_id != member.0
                    || start != member.1
                    || end != member.2
                    || granularity != memory.1
                    || hash != memory.2
                    || hash != member.3
                    || memory.3 != authoritative.sequence
                    || memory.4 != authoritative.sequence
                    || memory.5 != 1
                    || event_id != authoritative.id
                    || start != expected_span.start_char
                    || end != expected_span.end_char
                    || granularity != expected_granularity
                    || memory.1 != expected_granularity
                    || hash != expected_hash
                    || exact_content != expected_content
                {
                    return Err(RetrievalError::CorruptIndex(
                        "active leaf 文档的 retrieval/memory 来源不一致".into(),
                    ));
                }
                let span_count: i64 = connection
                    .query_row(
                        "SELECT count(*) FROM source_spans
                         WHERE event_id=?1 AND start_char=?2 AND end_char=?3
                           AND content_sha256=?4",
                        params![
                            authoritative.id,
                            usize_to_i64(expected_span.start_char)
                                .map_err(|error| self.database_error(error))?,
                            usize_to_i64(expected_span.end_char)
                                .map_err(|error| self.database_error(error))?,
                            expected_hash,
                        ],
                        |row| row.get(0),
                    )
                    .map_err(|error| self.database_error(error))?;
                if span_count != 1 {
                    return Err(RetrievalError::CorruptIndex(
                        "leaf 文档缺少 canonical source span".into(),
                    ));
                }
                Ok(memory_active
                    && control.allows_event(&authoritative.session_id, &authoritative.id))
            }
            "episode" | "session" => {
                let (_, active) = self.authoritative_trace_aggregate_document(
                    connection,
                    control,
                    document_id,
                    &memory.0,
                )?;
                if let Some((event_id, session_id, start, end, granularity, hash, _)) = raw {
                    let member_count: i64 = connection
                        .query_row(
                            "SELECT count(*) FROM memory_document_members
                         WHERE document_id=?1 AND event_id=?2 AND start_char=?3
                           AND end_char=?4 AND content_sha256=?5",
                            params![
                                document_id,
                                event_id,
                                usize_to_i64(start).map_err(|error| self.database_error(error))?,
                                usize_to_i64(end).map_err(|error| self.database_error(error))?,
                                hash
                            ],
                            |row| row.get(0),
                        )
                        .map_err(|error| self.database_error(error))?;
                    if session_id.as_deref() != Some(memory.0.as_str())
                        || granularity != memory.1
                        || hash != memory.2
                        || member_count != 1
                    {
                        return Err(RetrievalError::CorruptIndex(
                            "aggregate 文档的 retrieval/memory 来源不一致".into(),
                        ));
                    }
                }
                Ok(active)
            }
            _ => Err(RetrievalError::CorruptIndex(
                "trace 文档引用包含无效粒度".into(),
            )),
        }
    }

    fn document_metadata_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        document_id: &str,
    ) -> RetrievalResult<(bool, String, RetrievalDocumentGranularity)> {
        let active = self.document_reference_is_active(connection, control, document_id)?;
        let (session_id, granularity) = connection
            .query_row(
                "SELECT session_id,granularity FROM memory_documents WHERE document_id=?1",
                [document_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()
            .map_err(|error| self.database_error(error))?
            .ok_or_else(|| {
                RetrievalError::CorruptIndex("trace 文档引用缺少 canonical 元数据".into())
            })?;
        Ok((active, session_id, parse_granularity(&granularity)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn document_span_binding_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        document_id: &str,
        expected_session: &str,
        expected_span: &SourceSpan,
        expected_granularity: Option<RetrievalDocumentGranularity>,
        require_leaf: bool,
    ) -> RetrievalResult<bool> {
        let (active, session_id, granularity) =
            self.document_metadata_is_active(connection, control, document_id)?;
        if session_id != expected_session
            || expected_granularity.is_some_and(|expected| expected != granularity)
        {
            return Err(RetrievalError::CorruptIndex(
                "trace 文档与 raw session/granularity 不一致".into(),
            ));
        }
        match granularity {
            RetrievalDocumentGranularity::Message | RetrievalDocumentGranularity::Fragment => {
                let member = connection
                    .query_row(
                        "SELECT event_id,start_char,end_char FROM memory_document_members
                         WHERE document_id=?1 AND ordinal=0",
                        [document_id],
                        |row| {
                            Ok(SourceSpan {
                                event_id: row.get(0)?,
                                start_char: i64_to_usize(row.get(1)?)?,
                                end_char: i64_to_usize(row.get(2)?)?,
                            })
                        },
                    )
                    .map_err(|error| self.database_error(error))?;
                if member != *expected_span {
                    return Err(RetrievalError::CorruptIndex(
                        "trace leaf 文档未精确绑定 raw span".into(),
                    ));
                }
            }
            RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session => {
                if require_leaf {
                    return Err(RetrievalError::CorruptIndex(
                        "trace primary 文档必须是 canonical leaf".into(),
                    ));
                }
                let aggregate =
                    load_persisted_aggregate_document(connection, document_id, expected_session)?;
                if aggregate
                    .members
                    .iter()
                    .filter(|member| member.span == *expected_span)
                    .count()
                    != 1
                {
                    return Err(RetrievalError::CorruptIndex(
                        "trace aggregate 文档未精确包含 raw member".into(),
                    ));
                }
            }
        }
        Ok(active)
    }

    fn episode_reference_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        document_id: &str,
        expected_session: &str,
        event_id: &str,
    ) -> RetrievalResult<bool> {
        let (active, session_id, granularity) =
            self.document_metadata_is_active(connection, control, document_id)?;
        if session_id != expected_session || granularity != RetrievalDocumentGranularity::Episode {
            return Err(RetrievalError::CorruptIndex(
                "fusion episode_id 未绑定 canonical episode".into(),
            ));
        }
        let aggregate =
            load_persisted_aggregate_document(connection, document_id, expected_session)?;
        if aggregate
            .members
            .iter()
            .filter(|member| member.event_id == event_id)
            .count()
            != 1
        {
            return Err(RetrievalError::CorruptIndex(
                "fusion episode_id 未唯一包含 raw event".into(),
            ));
        }
        Ok(active)
    }

    fn ranked_candidate_document_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        candidate: &RankedCandidate,
    ) -> RetrievalResult<bool> {
        if candidate.document_id.is_empty() {
            return Err(RetrievalError::CorruptIndex(
                "BM25 trace 缺少 document_id".into(),
            ));
        }
        let active =
            self.document_reference_is_active(connection, control, &candidate.document_id)?;
        let granularity = match candidate.granularity {
            RetrievalDocumentGranularity::Message => "message",
            RetrievalDocumentGranularity::Fragment => "fragment",
            RetrievalDocumentGranularity::Episode => "episode",
            RetrievalDocumentGranularity::Session => "session",
        };
        let count: i64 = connection
            .query_row(
                "SELECT count(*) FROM retrieval_documents r
             JOIN events e ON e.event_id=r.event_id
             JOIN source_spans s ON s.event_id=r.event_id
                AND s.start_char=r.start_char AND s.end_char=r.end_char
             WHERE r.document_id=?1 AND e.session_id=?2 AND r.event_id=?3
               AND r.start_char=?4 AND r.end_char=?5 AND r.granularity=?6
               AND r.content_sha256=?7 AND s.content_sha256=?7 AND e.role=?8
               AND e.created_at=?9",
                params![
                    candidate.document_id,
                    candidate.session_id,
                    candidate.span.event_id,
                    usize_to_i64(candidate.span.start_char)
                        .map_err(|error| self.database_error(error))?,
                    usize_to_i64(candidate.span.end_char)
                        .map_err(|error| self.database_error(error))?,
                    granularity,
                    candidate.content_sha256,
                    candidate.role.as_str(),
                    candidate.created_at,
                ],
                |row| row.get(0),
            )
            .map_err(|error| self.database_error(error))?;
        if count != 1 {
            return Err(RetrievalError::CorruptIndex(
                "BM25 trace 与 canonical retrieval 来源不一致".into(),
            ));
        }
        Ok(active)
    }

    fn active_memory_document_count(
        &self,
        connection: &Connection,
        state: &ControlState,
        session_filter: Option<&str>,
    ) -> RetrievalResult<usize> {
        let mut statement = connection
            .prepare("SELECT document_id,session_id FROM memory_documents ORDER BY document_id")
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| self.database_error(error))?;
        let mut total = 0;
        for row in rows {
            let (id, session_id) = row.map_err(|error| self.database_error(error))?;
            if session_filter.is_none_or(|scope| scope == session_id)
                && self.memory_document_is_active(connection, state, &id)?
            {
                total += 1;
            }
        }
        Ok(total)
    }

    pub fn sync_session(
        &self,
        expected_session: &Session,
        source_path: &Path,
    ) -> RetrievalResult<SyncReport> {
        let _guard = self.acquire_root_write()?;
        self.sync_session_under_root_write(expected_session, source_path)
    }

    pub(crate) fn sync_session_under_root_write(
        &self,
        expected_session: &Session,
        source_path: &Path,
    ) -> RetrievalResult<SyncReport> {
        let state = self.replay_control_state_under_guard()?;
        Self::require_active_session(&state, &expected_session.id)?;
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
        let report = self.write_session(&transaction, &source, true, &state)?;
        self.require_unchanged_control_state(&state)?;
        transaction
            .commit()
            .map_err(|source| self.database_error(source))?;
        Ok(report)
    }

    pub fn rebuild(&self) -> RetrievalResult<SyncReport> {
        Ok(self.rebuild_with_options(RebuildOptions::default())?.sync)
    }

    pub fn rebuild_with_options(&self, options: RebuildOptions) -> RetrievalResult<RebuildReport> {
        let _guard = self.acquire_root_write()?;
        self.rebuild_under_root_write(options)
    }

    pub fn refresh_graph(
        &self,
        config: &MemoryConfig,
    ) -> RetrievalResult<crate::graph::GraphMaterializationReport> {
        crate::graph::refresh_graph(self, config)
    }

    fn rebuild_under_root_write(&self, options: RebuildOptions) -> RetrievalResult<RebuildReport> {
        let state = self.replay_control_state_under_guard()?;
        let all_sources = self.load_all_sources()?;
        let all_events = all_sources
            .iter()
            .flat_map(|source| derive_events(&source.session))
            .collect::<Vec<_>>();
        let sources = all_sources
            .into_iter()
            .filter(|source| state.allows_session(&source.session.id))
            .collect::<Vec<_>>();
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|source| self.database_error(source))?;
        let ledger_before = consolidation_ledger_snapshot(&transaction)?;
        validate_consolidation_ledger_semantics(&transaction)?;
        let replay = prepare_consolidation_replay(&transaction, &state, &all_events)?;
        let legacy_watermarks =
            snapshot_compatible_legacy_watermarks(&transaction, &state, &all_events)?;
        let reusable_embeddings = if options.reuse_compatible_embeddings {
            self.snapshot_reusable_leaf_embeddings(&transaction, &state)?
        } else {
            Vec::new()
        };
        transaction
            .execute_batch(
                "DELETE FROM memory_graph_edges;
                 DELETE FROM memory_graph_nodes;
                 DELETE FROM memory_graph_materializations;
                 DELETE FROM memory_episode_materializations;
                 DELETE FROM memory_episode_boundaries;
                 DELETE FROM memory_embeddings;
                 DELETE FROM memory_document_members;
                 DELETE FROM memory_documents;
                 DELETE FROM retrieval_documents_fts;
                 DELETE FROM retrieval_documents;
                 DELETE FROM retrieval_runs;
                 DELETE FROM answer_context_items;
                 DELETE FROM answer_contexts;
                 DELETE FROM memory_claim_transitions;
                 DELETE FROM memory_claim_evidence;
                 DELETE FROM memory_entity_mentions;
                 DELETE FROM memory_boundary_suggestions;
                 DELETE FROM memory_claims;
                 DELETE FROM memory_entity_aliases;
                 DELETE FROM memory_entities;
                 DELETE FROM consolidation_watermarks;
                 DELETE FROM source_spans;
                 DELETE FROM events;
                 DELETE FROM indexed_sessions;",
            )
            .map_err(|source| self.database_error(source))?;
        let mut total = SyncReport::default();
        for source in &sources {
            add_report(
                &mut total,
                self.write_session(&transaction, source, false, &state)?,
            );
        }
        // Every immutable event/span now exists, independent of filename order.
        // The second pass only needs materialize answer references; it may
        // refresh that source's own derived documents without deleting events.
        for source in &sources {
            let report = self.write_session(&transaction, source, true, &state)?;
            total.answer_contexts += report.answer_contexts;
        }
        let replay_report = replay_prepared_consolidation(&transaction, &state, replay)?;
        restore_compatible_legacy_watermarks(&transaction, &legacy_watermarks)?;
        let embeddings_reused =
            self.restore_reusable_leaf_embeddings(&transaction, &state, &reusable_embeddings)?;
        transaction
            .execute(
                "INSERT INTO memory_schema_meta(key,value) VALUES('active_control_generation_sha256',?1)
                 ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [state.generation_sha256()],
            )
            .map_err(|source| self.database_error(source))?;
        self.require_unchanged_control_state(&state)?;
        let marker: String = transaction
            .query_row(
                "SELECT value FROM memory_schema_meta
                 WHERE key='active_control_generation_sha256'",
                [],
                |row| row.get(0),
            )
            .map_err(|source| self.database_error(source))?;
        if marker != state.generation_sha256() {
            return Err(RetrievalError::CorruptIndex(
                "重建后的 active control generation marker 不匹配".into(),
            ));
        }
        validate_full_derived_integrity(&transaction)?;
        let aggregate_or_graph_rows: i64 = transaction
            .query_row(
                "SELECT
                    (SELECT count(*) FROM memory_graph_edges) +
                    (SELECT count(*) FROM memory_graph_nodes) +
                    (SELECT count(*) FROM memory_graph_materializations) +
                    (SELECT count(*) FROM memory_episode_materializations) +
                    (SELECT count(*) FROM memory_episode_boundaries) +
                    (SELECT count(*) FROM memory_documents WHERE granularity IN ('episode','session'))",
                [],
                |row| row.get(0),
            )
            .map_err(|source| self.database_error(source))?;
        if aggregate_or_graph_rows != 0 {
            return Err(RetrievalError::CorruptIndex(
                "重建后 graph、episode 或 session aggregate 投影非空".into(),
            ));
        }
        if !options.reuse_compatible_embeddings {
            let embeddings: i64 = transaction
                .query_row("SELECT count(*) FROM memory_embeddings", [], |row| {
                    row.get(0)
                })
                .map_err(|source| self.database_error(source))?;
            if embeddings != 0 {
                return Err(RetrievalError::CorruptIndex(
                    "强制重建后 embedding 投影非空".into(),
                ));
            }
        }
        let ledger_after = consolidation_ledger_snapshot(&transaction)?;
        if ledger_before != ledger_after {
            return Err(RetrievalError::CorruptIndex(
                "重建期间 immutable consolidation ledger 发生变化".into(),
            ));
        }
        transaction
            .commit()
            .map_err(|source| self.database_error(source))?;
        *self
            .vector_cache
            .lock()
            .map_err(|_| RetrievalError::CorruptIndex("向量缓存锁已损坏".into()))? = None;
        Ok(RebuildReport {
            sync: total,
            control_generation_sha256: state.generation_sha256(),
            ledger_attempts_preserved: ledger_before.row_count,
            consolidation_attempts_replayed: replay_report.replayed,
            consolidation_attempts_skipped_inactive: replay_report.skipped_inactive,
            consolidation_attempts_skipped_dependency: replay_report.skipped_dependency,
            embeddings_reused,
        })
    }

    fn snapshot_reusable_leaf_embeddings(
        &self,
        connection: &Connection,
        state: &ControlState,
    ) -> RetrievalResult<Vec<ReusableLeafEmbedding>> {
        let mut statement = connection
            .prepare(
                "SELECT d.document_id,d.session_id,d.granularity,d.source_sha256,
                        e.model,e.dimensions,e.source_sha256,e.index_fingerprint,
                        e.vector_blob,e.embedded_at
                 FROM memory_documents d
                 JOIN memory_embeddings e ON e.document_id=d.document_id
                 WHERE d.granularity IN ('message','fragment')
                 ORDER BY d.document_id",
            )
            .map_err(|error| self.database_error(error))?;
        let mut rows = statement
            .query([])
            .map_err(|error| self.database_error(error))?;
        let mut reusable = Vec::new();
        while let Some(row) = rows.next().map_err(|error| self.database_error(error))? {
            let Some(row) = (|| {
                Some(ReusableLeafEmbedding {
                    document_id: repair_row_text(row, 0)?,
                    session_id: repair_row_text(row, 1)?,
                    granularity: repair_row_text(row, 2)?,
                    document_source_sha256: repair_row_text(row, 3)?,
                    model: repair_row_text(row, 4)?,
                    dimensions: repair_row_i64(row, 5)?,
                    embedding_source_sha256: repair_row_text(row, 6)?,
                    index_fingerprint: repair_row_text(row, 7)?,
                    vector_blob: repair_row_blob(row, 8)?,
                    embedded_at: repair_row_text(row, 9)?,
                })
            })() else {
                continue;
            };
            let dimensions = usize::try_from(row.dimensions).ok();
            let valid = state.allows_session(&row.session_id)
                && row.document_source_sha256 == row.embedding_source_sha256
                && !row.model.trim().is_empty()
                && is_sha256_hex(&row.document_source_sha256)
                && is_sha256_hex(&row.index_fingerprint)
                && DateTime::parse_from_rfc3339(&row.embedded_at).is_ok()
                && dimensions.is_some_and(|value| (32..=4096).contains(&value))
                && dimensions
                    .and_then(|value| value.checked_mul(4))
                    .is_some_and(|length| length == row.vector_blob.len())
                && dimensions
                    .and_then(|value| decode_f32_le(&row.vector_blob, value).ok())
                    .is_some_and(|vector| is_unit_vector(&vector));
            let active = match self.memory_document_is_active(connection, state, &row.document_id) {
                Ok(active) => active,
                Err(RetrievalError::Database { path, source }) => {
                    return Err(RetrievalError::Database { path, source });
                }
                Err(_) => false,
            };
            if valid && active {
                reusable.push(row);
            }
        }
        Ok(reusable)
    }

    fn restore_reusable_leaf_embeddings(
        &self,
        transaction: &Transaction<'_>,
        state: &ControlState,
        candidates: &[ReusableLeafEmbedding],
    ) -> RetrievalResult<usize> {
        let mut reused = 0;
        for candidate in candidates {
            let current = transaction
                .query_row(
                    "SELECT session_id,granularity,source_sha256 FROM memory_documents
                     WHERE document_id=?1",
                    [&candidate.document_id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    },
                )
                .optional()
                .map_err(|error| self.database_error(error))?;
            if current.as_ref()
                != Some(&(
                    candidate.session_id.clone(),
                    candidate.granularity.clone(),
                    candidate.document_source_sha256.clone(),
                ))
                || !self.memory_document_is_active(transaction, state, &candidate.document_id)?
            {
                continue;
            }
            transaction
                .execute(
                    "INSERT INTO memory_embeddings
                     (document_id,model,dimensions,source_sha256,index_fingerprint,vector_blob,embedded_at)
                     VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![
                        candidate.document_id,
                        candidate.model,
                        candidate.dimensions,
                        candidate.embedding_source_sha256,
                        candidate.index_fingerprint,
                        candidate.vector_blob,
                        candidate.embedded_at,
                    ],
                )
                .map_err(|error| self.database_error(error))?;
            reused += 1;
        }
        Ok(reused)
    }

    /// Atomically replace the persisted embeddings for a batch of immutable
    /// derived documents.  The expected source hash prevents a vector
    /// produced for an older source span from being committed after resync.
    pub fn upsert_embeddings(
        &self,
        spec: &VectorIndexSpec,
        writes: &[EmbeddingWrite],
    ) -> RetrievalResult<()> {
        spec.validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let _guard = self.acquire_root_write()?;
        let control = self.replay_control_state_under_guard()?;
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| self.database_error(error))?;
        self.require_current_control_projection(&transaction, &control)?;
        let mut document_ids = HashSet::with_capacity(writes.len());
        let mut prepared = Vec::with_capacity(writes.len());
        let mut leaf_sessions = HashSet::new();
        let mut aggregate_sessions = HashSet::new();
        for write in writes {
            if !document_ids.insert(write.document_id.as_str()) {
                return Err(RetrievalError::CorruptIndex(format!(
                    "批量写入包含重复文档 {}",
                    write.document_id
                )));
            }
            if write.expected_source_sha256.trim().is_empty() {
                return Err(RetrievalError::CorruptIndex(format!(
                    "文档 {} 缺少源哈希",
                    write.document_id
                )));
            }
            if write.vector.len() != spec.dimensions {
                return Err(RetrievalError::CorruptIndex(format!(
                    "文档 {} 的向量维度不匹配",
                    write.document_id
                )));
            }
            let bytes = encode_f32_le(&write.vector)
                .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
            let (current_hash, session_id, granularity) = transaction
                .query_row(
                    "SELECT source_sha256, session_id, granularity FROM memory_documents WHERE document_id=?1",
                    [&write.document_id],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)),
                )
                .optional()
                .map_err(|error| self.database_error(error))?
                .ok_or_else(|| {
                    RetrievalError::CorruptIndex(format!("未知派生文档 {}", write.document_id))
                })?;
            if !self.memory_document_is_active(&transaction, &control, &write.document_id)? {
                return Err(RetrievalError::ExcludedEvent(write.document_id.clone()));
            }
            if current_hash != write.expected_source_sha256 {
                return Err(RetrievalError::CorruptIndex(format!(
                    "文档 {} 的源哈希已变化",
                    write.document_id
                )));
            }
            match granularity.as_str() {
                "message" | "fragment" => {
                    leaf_sessions.insert(session_id.clone());
                }
                "episode" | "session" => {
                    aggregate_sessions.insert(session_id.clone());
                }
                _ => {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "文档 {} 的粒度无效 {granularity}",
                        write.document_id
                    )));
                }
            }
            prepared.push(PreparedEmbedding {
                document_id: write.document_id.clone(),
                session_id,
                granularity,
                source_sha256: write.expected_source_sha256.clone(),
                vector_blob: bytes,
            });
        }
        let fingerprint = spec
            .fingerprint()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        if leaf_sessions
            .iter()
            .any(|session| aggregate_sessions.contains(session))
        {
            return Err(RetrievalError::CorruptIndex(
                "同一会话不能在 leaf 更新时发布 aggregate 向量".into(),
            ));
        }
        let prewrite_readiness = self.aggregate_readiness_by_session(
            &transaction,
            &aggregate_sessions,
            spec,
            &fingerprint,
            true,
            AggregateAuditPhase::PreWrite,
        )?;
        for prepared_write in &prepared {
            if matches!(prepared_write.granularity.as_str(), "episode" | "session") {
                let Some(audit) = prewrite_readiness.get(&prepared_write.session_id) else {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "文档 {} 缺少缓存的 aggregate readiness",
                        prepared_write.document_id
                    )));
                };
                if audit.readiness != AggregateReadiness::Ready {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "文档 {} 的 aggregate 输入不完整或不兼容",
                        prepared_write.document_id
                    )));
                }
                validate_aggregate_document_source(
                    &transaction,
                    &prepared_write.document_id,
                    &prepared_write.session_id,
                    &prepared_write.source_sha256,
                )?;
                validate_canonical_aggregate_vector_blob(
                    &prepared_write.document_id,
                    &prepared_write.vector_blob,
                    &audit.canonical_vector_blobs,
                )?;
            }
        }
        let embedded_at = Utc::now().to_rfc3339();
        for prepared_write in &prepared {
            transaction
                .execute(
                    "INSERT INTO memory_embeddings
                     (document_id, model, dimensions, source_sha256, index_fingerprint, vector_blob, embedded_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                     ON CONFLICT(document_id) DO UPDATE SET
                     model=excluded.model, dimensions=excluded.dimensions,
                     source_sha256=excluded.source_sha256,
                     index_fingerprint=excluded.index_fingerprint,
                     vector_blob=excluded.vector_blob, embedded_at=excluded.embedded_at",
                    params![
                        prepared_write.document_id,
                        spec.model,
                        usize_to_i64(spec.dimensions).map_err(|error| self.database_error(error))?,
                        prepared_write.source_sha256,
                        &fingerprint,
                        &prepared_write.vector_blob,
                        &embedded_at,
                    ],
                )
                .map_err(|error| self.database_error(error))?;
            verify_embedding_writeback(
                &transaction,
                prepared_write,
                spec,
                &fingerprint,
                &embedded_at,
            )?;
        }
        for session_id in leaf_sessions {
            transaction.execute(
                "DELETE FROM memory_embeddings WHERE document_id IN (SELECT document_id FROM memory_documents WHERE session_id=?1 AND granularity IN ('episode','session'))",
                [&session_id],
            ).map_err(|error| self.database_error(error))?;
            transaction
                .execute(
                    "DELETE FROM memory_episode_materializations WHERE session_id=?1",
                    [session_id],
                )
                .map_err(|error| self.database_error(error))?;
        }
        for prepared_write in &prepared {
            verify_embedding_writeback(
                &transaction,
                prepared_write,
                spec,
                &fingerprint,
                &embedded_at,
            )?;
        }
        if !aggregate_sessions.is_empty() {
            self.validate_aggregate_derived_integrity(&transaction)?;
            let final_readiness = self.aggregate_readiness_by_session(
                &transaction,
                &aggregate_sessions,
                spec,
                &fingerprint,
                true,
                AggregateAuditPhase::FinalWrite,
            )?;
            for prepared_write in &prepared {
                if matches!(prepared_write.granularity.as_str(), "episode" | "session") {
                    let Some(audit) = final_readiness.get(&prepared_write.session_id) else {
                        return Err(RetrievalError::CorruptIndex(format!(
                            "文档 {} 缺少写后缓存的 aggregate readiness",
                            prepared_write.document_id
                        )));
                    };
                    if audit.readiness != AggregateReadiness::Ready {
                        return Err(RetrievalError::CorruptIndex(format!(
                            "文档 {} 的 aggregate 输入在批量写入后不完整或不兼容",
                            prepared_write.document_id
                        )));
                    }
                    validate_aggregate_document_source(
                        &transaction,
                        &prepared_write.document_id,
                        &prepared_write.session_id,
                        &prepared_write.source_sha256,
                    )?;
                    validate_canonical_aggregate_vector_blob(
                        &prepared_write.document_id,
                        &prepared_write.vector_blob,
                        &audit.canonical_vector_blobs,
                    )?;
                }
            }
        }
        self.require_unchanged_control_state(&control)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(())
    }

    pub fn leaf_embedding_snapshot(
        &self,
        spec: &VectorIndexSpec,
    ) -> RetrievalResult<LeafEmbeddingSnapshot> {
        validate_embedding_spec(spec)?;
        let _guard = self.acquire_root_read()?;
        let control = self.replay_control_state_under_guard()?;
        let connection = self.open_connection()?;
        self.require_current_control_projection(&connection, &control)?;
        let snapshot =
            load_leaf_embedding_snapshot_with_control(self, &connection, spec, &control)?;
        self.require_unchanged_control_state(&control)?;
        Ok(snapshot)
    }

    pub fn publish_leaf_embedding_catalog(
        &self,
        spec: &VectorIndexSpec,
        snapshot: &LeafEmbeddingSnapshot,
        writes: &[EmbeddingWrite],
    ) -> RetrievalResult<EmbeddingPublishReport> {
        validate_embedding_spec(spec)?;
        let prepared = prepare_complete_catalog_writes(
            spec,
            snapshot.documents.iter().map(|document| {
                (
                    document.document_id.as_str(),
                    document.source_sha256.as_str(),
                )
            }),
            writes,
        )?;
        let _guard = self.acquire_root_write()?;
        let control = self.replay_control_state_under_guard()?;
        if snapshot.control_generation_sha256 != control.generation_sha256() {
            return Err(RetrievalError::EmbeddingCatalogStale {
                kind: "leaf".into(),
            });
        }
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| self.database_error(error))?;
        self.require_current_control_projection(&transaction, &control)?;
        let current =
            load_leaf_embedding_snapshot_with_control(self, &transaction, spec, &control)?;
        if current.catalog_sha256 != snapshot.catalog_sha256
            || current.control_generation_sha256 != snapshot.control_generation_sha256
            || current.session_ids != snapshot.session_ids
            || current.documents != snapshot.documents
        {
            return Err(RetrievalError::EmbeddingCatalogStale {
                kind: "leaf".into(),
            });
        }
        let fingerprint = embedding_fingerprint(spec)?;
        let embedded_at = Utc::now().to_rfc3339();
        let mut changed_sessions = HashSet::new();
        let mut reused = 0;
        for (document, write) in current.documents.iter().zip(&prepared) {
            let existing = raw_embedding(&transaction, &document.document_id)?;
            if embedding_equals(
                existing.as_ref(),
                spec,
                &fingerprint,
                &document.source_sha256,
                &write.vector_blob,
            ) {
                reused += 1;
                continue;
            }
            transaction.execute(
                "INSERT INTO memory_embeddings(document_id,model,dimensions,source_sha256,index_fingerprint,vector_blob,embedded_at)
                 VALUES(?1,?2,?3,?4,?5,?6,?7)
                 ON CONFLICT(document_id) DO UPDATE SET model=excluded.model,dimensions=excluded.dimensions,
                 source_sha256=excluded.source_sha256,index_fingerprint=excluded.index_fingerprint,
                 vector_blob=excluded.vector_blob,embedded_at=excluded.embedded_at",
                params![document.document_id, spec.model, usize_to_i64(spec.dimensions).map_err(|e| self.database_error(e))?, document.source_sha256, fingerprint, write.vector_blob, embedded_at],
            ).map_err(|error| self.database_error(error))?;
            changed_sessions.insert(document.session_id.clone());
        }
        for session_id in &changed_sessions {
            transaction.execute(
                "DELETE FROM memory_embeddings WHERE document_id IN
                 (SELECT document_id FROM memory_documents WHERE session_id=?1 AND granularity IN ('episode','session'))",
                [session_id],
            ).map_err(|error| self.database_error(error))?;
            transaction
                .execute(
                    "DELETE FROM memory_episode_materializations WHERE session_id=?1",
                    [session_id],
                )
                .map_err(|error| self.database_error(error))?;
        }
        verify_published_catalog(
            &transaction,
            spec,
            &fingerprint,
            &prepared,
            "'message','fragment'",
        )?;
        self.require_unchanged_control_state(&control)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(EmbeddingPublishReport {
            documents: prepared.len(),
            reused,
            changed: !changed_sessions.is_empty(),
        })
    }

    pub fn aggregate_embedding_snapshot(
        &self,
        spec: &VectorIndexSpec,
    ) -> RetrievalResult<AggregateEmbeddingSnapshot> {
        validate_embedding_spec(spec)?;
        let _guard = self.acquire_root_read()?;
        let control = self.replay_control_state_under_guard()?;
        let connection = self.open_connection()?;
        self.require_current_control_projection(&connection, &control)?;
        let snapshot =
            load_aggregate_embedding_snapshot_with_control(self, &connection, spec, &control)?;
        self.require_unchanged_control_state(&control)?;
        Ok(snapshot)
    }

    pub fn publish_aggregate_embedding_catalog(
        &self,
        spec: &VectorIndexSpec,
        snapshot: &AggregateEmbeddingSnapshot,
        writes: &[EmbeddingWrite],
    ) -> RetrievalResult<EmbeddingPublishReport> {
        validate_embedding_spec(spec)?;
        let prepared = prepare_complete_catalog_writes(
            spec,
            snapshot.documents.iter().map(|document| {
                (
                    document.document_id.as_str(),
                    document.source_sha256.as_str(),
                )
            }),
            writes,
        )?;
        let _guard = self.acquire_root_write()?;
        let control = self.replay_control_state_under_guard()?;
        if snapshot.control_generation_sha256 != control.generation_sha256() {
            return Err(RetrievalError::EmbeddingCatalogStale {
                kind: "aggregate".into(),
            });
        }
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| self.database_error(error))?;
        self.require_current_control_projection(&transaction, &control)?;
        let current =
            load_aggregate_embedding_snapshot_with_control(self, &transaction, spec, &control)?;
        if current != *snapshot {
            return Err(RetrievalError::EmbeddingCatalogStale {
                kind: "aggregate".into(),
            });
        }
        let fingerprint = embedding_fingerprint(spec)?;
        let expected = canonical_aggregate_blobs_from_snapshot(&current, spec.dimensions)?;
        for write in &prepared {
            if expected.get(&write.document_id) != Some(&write.vector_blob) {
                return Err(RetrievalError::CorruptIndex(format!(
                    "aggregate document {} 的向量不是规范 equal_mean",
                    write.document_id
                )));
            }
        }
        let embedded_at = Utc::now().to_rfc3339();
        let mut reused = 0;
        for write in &prepared {
            let existing = raw_embedding(&transaction, &write.document_id)?;
            if embedding_equals(
                existing.as_ref(),
                spec,
                &fingerprint,
                &write.source_sha256,
                &write.vector_blob,
            ) {
                reused += 1;
                continue;
            }
            transaction.execute(
                "INSERT INTO memory_embeddings(document_id,model,dimensions,source_sha256,index_fingerprint,vector_blob,embedded_at)
                 VALUES(?1,?2,?3,?4,?5,?6,?7)
                 ON CONFLICT(document_id) DO UPDATE SET model=excluded.model,dimensions=excluded.dimensions,
                 source_sha256=excluded.source_sha256,index_fingerprint=excluded.index_fingerprint,
                 vector_blob=excluded.vector_blob,embedded_at=excluded.embedded_at",
                params![write.document_id, spec.model, usize_to_i64(spec.dimensions).map_err(|e| self.database_error(e))?, write.source_sha256, fingerprint, write.vector_blob, embedded_at],
            ).map_err(|error| self.database_error(error))?;
        }
        verify_published_catalog(
            &transaction,
            spec,
            &fingerprint,
            &prepared,
            "'episode','session'",
        )?;
        self.require_unchanged_control_state(&control)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(EmbeddingPublishReport {
            documents: prepared.len(),
            reused,
            changed: reused != prepared.len(),
        })
    }

    pub fn compatible_embeddings(
        &self,
        spec: &VectorIndexSpec,
    ) -> RetrievalResult<Vec<StoredEmbedding>> {
        spec.validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let _guard = self.acquire_root_read()?;
        let control = self.replay_control_state_under_guard()?;
        let connection = self.open_connection()?;
        self.require_current_control_projection(&connection, &control)?;
        let fingerprint = spec
            .fingerprint()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let rows =
            self.compatible_embeddings_from_connection(&connection, spec, &fingerprint, None)?;
        self.require_unchanged_control_state(&control)?;
        Ok(rows)
    }

    pub fn embedding_coverage(&self, spec: &VectorIndexSpec) -> RetrievalResult<EmbeddingCoverage> {
        spec.validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let _guard = self.acquire_root_read()?;
        let control = self.replay_control_state_under_guard()?;
        let connection = self.open_connection()?;
        self.require_current_control_projection(&connection, &control)?;
        let total = self.active_memory_document_count(&connection, &control, None)?;
        let fingerprint = spec
            .fingerprint()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let compatible = self
            .compatible_embeddings_from_connection(&connection, spec, &fingerprint, None)?
            .len();
        self.require_unchanged_control_state(&control)?;
        Ok(EmbeddingCoverage {
            total,
            compatible,
            stale: total.saturating_sub(compatible),
        })
    }

    pub(crate) fn compatible_embeddings_from_connection(
        &self,
        connection: &Connection,
        spec: &VectorIndexSpec,
        fingerprint: &str,
        session_filter: Option<&str>,
    ) -> RetrievalResult<Vec<StoredEmbedding>> {
        let control = self.replay_control_state_under_guard()?;
        self.compatible_embeddings_from_connection_with_control(
            connection,
            spec,
            fingerprint,
            session_filter,
            &control,
        )
    }

    pub(crate) fn compatible_embeddings_from_connection_with_control(
        &self,
        connection: &Connection,
        spec: &VectorIndexSpec,
        fingerprint: &str,
        session_filter: Option<&str>,
        control: &ControlState,
    ) -> RetrievalResult<Vec<StoredEmbedding>> {
        let scope_clause = if session_filter.is_some() {
            " AND d.session_id=?4"
        } else {
            ""
        };
        let mut statement = connection
            .prepare(&format!(
                "SELECT d.document_id, d.session_id, d.granularity, d.source_sha256,
                        e.source_sha256, e.model, e.dimensions, e.index_fingerprint,
                        e.vector_blob, e.embedded_at
                 FROM memory_documents d JOIN memory_embeddings e ON e.document_id=d.document_id
                 WHERE e.model=?1 AND e.dimensions=?2 AND e.index_fingerprint=?3{scope_clause}
                 ORDER BY d.document_id ASC"
            ))
            .map_err(|error| self.database_error(error))?;
        let dimensions =
            usize_to_i64(spec.dimensions).map_err(|error| self.database_error(error))?;
        let mut query_params = vec![
            rusqlite::types::Value::Text(spec.model.clone()),
            rusqlite::types::Value::Integer(dimensions),
            rusqlite::types::Value::Text(fingerprint.to_owned()),
        ];
        if let Some(session_id) = session_filter {
            query_params.push(rusqlite::types::Value::Text(session_id.to_owned()));
        }
        let rows = statement
            .query_map(rusqlite::params_from_iter(query_params), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    i64_to_usize(row.get(6)?)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, Vec<u8>>(8)?,
                    row.get::<_, String>(9)?,
                ))
            })
            .map_err(|error| self.database_error(error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| self.database_error(error))?;
        let mut active_rows = Vec::with_capacity(rows.len());
        for row in rows {
            if self.memory_document_is_active(connection, control, &row.0)? {
                active_rows.push(row);
            }
        }
        let rows = active_rows;
        let aggregate_sessions = rows
            .iter()
            .filter(|row| matches!(row.2.as_str(), "episode" | "session"))
            .map(|row| row.1.clone())
            .collect::<HashSet<_>>();
        let aggregate_readiness = if aggregate_sessions.is_empty() {
            HashMap::new()
        } else {
            if session_filter.is_none() {
                self.validate_aggregate_derived_integrity(connection)?;
            }
            self.aggregate_readiness_by_session(
                connection,
                &aggregate_sessions,
                spec,
                fingerprint,
                true,
                AggregateAuditPhase::Read,
            )?
        };
        let mut compatible = Vec::with_capacity(rows.len());
        for (
            document_id,
            session_id,
            granularity,
            source_sha256,
            embedding_source_sha256,
            model,
            dimensions,
            index_fingerprint,
            bytes,
            embedded_at,
        ) in rows
        {
            let parsed_granularity = parse_memory_granularity(&granularity)?;
            if embedding_source_sha256 != source_sha256 {
                continue;
            }
            if matches!(granularity.as_str(), "episode" | "session") {
                match aggregate_readiness.get(&session_id) {
                    Some(audit) if audit.readiness == AggregateReadiness::Ready => {
                        validate_aggregate_document_source(
                            connection,
                            &document_id,
                            &session_id,
                            &source_sha256,
                        )?;
                        validate_canonical_aggregate_vector_blob(
                            &document_id,
                            &bytes,
                            &audit.canonical_vector_blobs,
                        )?;
                    }
                    Some(audit) if audit.readiness == AggregateReadiness::Stale => continue,
                    Some(_) => unreachable!("aggregate readiness variants are exhaustive"),
                    None => {
                        return Err(aggregate_corruption(format!(
                            "会话 {session_id} 缺少缓存的 aggregate readiness"
                        )));
                    }
                }
            }
            let vector = decode_f32_le(&bytes, dimensions).map_err(|error| {
                RetrievalError::CorruptIndex(format!("文档 {document_id} 的兼容向量损坏：{error}"))
            })?;
            compatible.push(StoredEmbedding {
                document_id,
                session_id,
                granularity: parsed_granularity,
                source_sha256,
                model,
                dimensions,
                index_fingerprint,
                vector,
                embedded_at,
            });
        }
        Ok(compatible)
    }

    /// Rebuild the session's aggregate provenance catalog without generating
    /// text or calling a model. Existing aggregate vectors are deleted as the
    /// rebuilt provenance may require fresh canonical-message coverage.
    pub fn materialize_episode_documents(
        &self,
        session_id: &str,
        config: &MemoryConfig,
    ) -> RetrievalResult<EpisodeMaterializationReport> {
        config
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let spec = VectorIndexSpec::from_config(config)
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let fingerprint = spec
            .fingerprint()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let _guard = self.acquire_root_write()?;
        let control = self.replay_control_state_under_guard()?;
        Self::require_active_session(&control, session_id)?;
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| self.database_error(error))?;
        self.require_current_control_projection(&transaction, &control)?;
        let session = self.get_session_from_connection(&transaction, session_id)?;
        self.verify_fresh(&session)?;
        if validate_aggregate_raw_source(
            &transaction,
            &self.root,
            session_id,
            &session.source_file,
            &session.source_sha256,
        )? != AggregateReadiness::Ready
        {
            return Err(RetrievalError::StaleIndex {
                session_id: session_id.to_owned(),
            });
        }
        let (messages, watermark, suggestions, ledger_snapshot_sha256) =
            load_episode_snapshot(&transaction, session_id, &spec, &fingerprint)?;
        let plan = plan_episodes(&EpisodePlanInput {
            session_id: session_id.to_owned(),
            source_session_sha256: session.source_sha256.clone(),
            gap_minutes: config.episode_gap_minutes,
            consolidation_watermark: watermark,
            messages,
            suggestions,
        })
        .map_err(RetrievalError::CorruptIndex)?;
        persist_episode_plan(
            &transaction,
            &plan,
            ledger_snapshot_sha256,
            &fingerprint,
            config,
        )?;
        self.validate_aggregate_derived_integrity(&transaction)?;
        if self
            .validate_episode_materialization_phase(
                &transaction,
                session_id,
                &spec,
                &fingerprint,
                false,
                AggregateAuditPhase::MaterializeFinal,
            )?
            .readiness
            != AggregateReadiness::Ready
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "会话 {session_id} 的 episode materialization 写后审计未就绪"
            )));
        }
        self.require_unchanged_control_state(&control)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(plan)
    }

    pub(crate) fn acquire_root_read(&self) -> RetrievalResult<RwLockReadGuard<'_, ()>> {
        self.root_lock
            .read()
            .map_err(|_| RetrievalError::RootLockPoisoned {
                path: self.root.clone(),
            })
    }

    pub(crate) fn acquire_root_write(&self) -> RetrievalResult<RwLockWriteGuard<'_, ()>> {
        self.root_lock
            .write()
            .map_err(|_| RetrievalError::RootLockPoisoned {
                path: self.root.clone(),
            })
    }

    #[cfg(test)]
    pub(crate) fn episode_entity_sets_for_test(
        &self,
        session_id: &str,
        config: &MemoryConfig,
    ) -> RetrievalResult<Vec<(String, std::collections::BTreeSet<String>)>> {
        config
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let spec = VectorIndexSpec::from_config(config)
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let fingerprint = spec
            .fingerprint()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let connection = self.open_connection()?;
        let (messages, _, _, _) =
            load_episode_snapshot(&connection, session_id, &spec, &fingerprint)?;
        Ok(messages
            .into_iter()
            .map(|message| (message.member.event_id, message.resolved_entity_ids))
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn episode_plan_input_for_test(
        &self,
        session_id: &str,
        config: &MemoryConfig,
    ) -> RetrievalResult<EpisodePlanInput> {
        config
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let spec = VectorIndexSpec::from_config(config)
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let fingerprint = spec
            .fingerprint()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let connection = self.open_connection()?;
        let session = self.get_session_from_connection(&connection, session_id)?;
        let (messages, watermark, suggestions, _) =
            load_episode_snapshot(&connection, session_id, &spec, &fingerprint)?;
        Ok(EpisodePlanInput {
            session_id: session_id.to_owned(),
            source_session_sha256: session.source_sha256,
            gap_minutes: config.episode_gap_minutes,
            consolidation_watermark: watermark,
            messages,
            suggestions,
        })
    }

    #[cfg(test)]
    pub(crate) fn set_consolidation_test_hook(&self, hook: Option<ConsolidationHook>) {
        *self
            .test_hooks
            .consolidation
            .lock()
            .expect("test hook mutex must not be poisoned") = hook;
    }

    #[cfg(test)]
    pub(crate) fn run_consolidation_test_hook(&self, point: ConsolidationHookPoint) {
        let hook = self
            .test_hooks
            .consolidation
            .lock()
            .expect("test hook mutex must not be poisoned")
            .clone();
        if let Some(hook) = hook {
            hook(point);
        }
    }

    #[cfg(test)]
    fn set_aggregate_audit_test_hook(&self, hook: Option<AggregateAuditHook>) {
        *self
            .test_hooks
            .aggregate_audit
            .lock()
            .expect("test hook mutex must not be poisoned") = hook;
    }

    #[cfg(test)]
    fn run_aggregate_audit_test_hook(&self, point: AggregateAuditHookPoint) {
        let hook = self
            .test_hooks
            .aggregate_audit
            .lock()
            .expect("test hook mutex must not be poisoned")
            .clone();
        if let Some(hook) = hook {
            hook(point);
        }
    }

    fn validate_aggregate_derived_integrity(&self, connection: &Connection) -> RetrievalResult<()> {
        #[cfg(test)]
        self.run_aggregate_audit_test_hook(AggregateAuditHookPoint::DerivedIntegrity);
        validate_full_derived_integrity(connection)
    }

    fn validate_episode_materialization_phase(
        &self,
        connection: &Connection,
        session_id: &str,
        spec: &VectorIndexSpec,
        fingerprint: &str,
        require_complete_message_embeddings: bool,
        phase: AggregateAuditPhase,
    ) -> RetrievalResult<AggregateSessionAudit> {
        #[cfg(test)]
        self.run_aggregate_audit_test_hook(AggregateAuditHookPoint::Materialization {
            session_id: session_id.to_owned(),
            phase,
        });
        #[cfg(not(test))]
        let _ = phase;
        let audit = validate_episode_materialization(
            connection,
            &self.root,
            session_id,
            spec,
            fingerprint,
            require_complete_message_embeddings,
        )?;
        if require_complete_message_embeddings && audit.readiness == AggregateReadiness::Ready {
            validate_existing_canonical_aggregate_embeddings(
                connection,
                session_id,
                spec,
                fingerprint,
                &audit.canonical_vector_blobs,
            )?;
        }
        Ok(audit)
    }

    fn aggregate_readiness_by_session(
        &self,
        connection: &Connection,
        sessions: &HashSet<String>,
        spec: &VectorIndexSpec,
        fingerprint: &str,
        require_complete_message_embeddings: bool,
        phase: AggregateAuditPhase,
    ) -> RetrievalResult<HashMap<String, AggregateSessionAudit>> {
        let mut ordered_sessions = sessions.iter().collect::<Vec<_>>();
        ordered_sessions.sort();
        ordered_sessions
            .into_iter()
            .map(|session_id| {
                self.validate_episode_materialization_phase(
                    connection,
                    session_id,
                    spec,
                    fingerprint,
                    require_complete_message_embeddings,
                    phase,
                )
                .map(|readiness| (session_id.clone(), readiness))
            })
            .collect()
    }

    pub fn get_session(&self, session_id: &str) -> RetrievalResult<IndexedSession> {
        let _guard = self.acquire_root_read()?;
        let state = self.replay_control_state_under_guard()?;
        Self::require_active_session(&state, session_id)?;
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|error| self.database_error(error))?;
        let session = transaction
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
        self.require_unchanged_control_state(&state)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(session)
    }

    pub fn get_event(&self, event_id: &str) -> RetrievalResult<StoredEvent> {
        let _guard = self.acquire_root_read()?;
        let state = self.replay_control_state_under_guard()?;
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|error| self.database_error(error))?;
        let event = self.get_event_from_connection(&transaction, event_id)?;
        Self::require_active_event(&state, &event)?;
        let session = self.get_session_from_connection(&transaction, &event.session_id)?;
        self.verify_fresh(&session)?;
        verify_event_hash(&event)?;
        self.require_unchanged_control_state(&state)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(event)
    }

    pub fn replay_session(&self, session_id: &str) -> RetrievalResult<Vec<StoredEvent>> {
        let _guard = self.acquire_root_read()?;
        let state = self.replay_control_state_under_guard()?;
        Self::require_active_session(&state, session_id)?;
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|error| self.database_error(error))?;
        let session = self.get_session_from_connection(&transaction, session_id)?;
        self.verify_fresh(&session)?;
        let mut statement = transaction
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
            if state.allows_event(&event.session_id, &event.id) {
                events.push(event);
            }
        }
        drop(statement);
        self.require_unchanged_control_state(&state)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(events)
    }

    pub(crate) fn replay_session_from_connection_with_state(
        &self,
        connection: &Connection,
        session_id: &str,
        state: &ControlState,
    ) -> RetrievalResult<Vec<StoredEvent>> {
        Self::require_active_session(state, session_id)?;
        let session = self.get_session_from_connection(connection, session_id)?;
        self.verify_fresh(&session)?;
        let mut statement = connection
            .prepare(
                "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                    content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error
             FROM events WHERE session_id=?1 ORDER BY sequence",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([session_id], map_event)
            .map_err(|error| self.database_error(error))?;
        let mut events = Vec::new();
        for row in rows {
            let event = row.map_err(|error| self.database_error(error))?;
            verify_event_hash(&event)?;
            events.push(event);
        }
        Ok(events)
    }

    pub fn resolve_span(&self, span: &SourceSpan) -> RetrievalResult<ResolvedSpan> {
        let _guard = self.acquire_root_read()?;
        let state = self.replay_control_state_under_guard()?;
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|error| self.database_error(error))?;
        let event = self.get_event_from_connection(&transaction, &span.event_id)?;
        Self::require_active_event(&state, &event)?;
        let session = self.get_session_from_connection(&transaction, &event.session_id)?;
        self.verify_fresh(&session)?;
        verify_event_hash(&event)?;
        let start_char =
            usize_to_i64(span.start_char).map_err(|source| self.database_error(source))?;
        let end_char = usize_to_i64(span.end_char).map_err(|source| self.database_error(source))?;
        let saved_hash = transaction
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
        let resolved = ResolvedSpan {
            span: span.clone(),
            content,
            content_sha256: actual_hash,
        };
        self.require_unchanged_control_state(&state)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(resolved)
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
        self.keyword_recall_scoped(
            raw_query,
            current_user_event_id,
            recent_event_ids,
            None,
            config,
        )
    }

    fn keyword_recall_scoped(
        &self,
        raw_query: &str,
        current_user_event_id: &str,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
        config: RetrievalConfig,
    ) -> RetrievalResult<RecallResult> {
        self.keyword_recall_scoped_with_origin(
            raw_query,
            current_user_event_id,
            recent_event_ids,
            session_filter,
            config,
            &RecallQueryOrigin::IndexedEvent,
        )
    }

    fn keyword_recall_scoped_with_origin(
        &self,
        raw_query: &str,
        current_user_event_id: &str,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
        config: RetrievalConfig,
        query_origin: &RecallQueryOrigin,
    ) -> RetrievalResult<RecallResult> {
        let _guard = self.acquire_root_read()?;
        let state = self.replay_control_state_under_guard()?;
        Self::require_active_session(&state, session_filter.unwrap_or_default()).or_else(
            |error| {
                if session_filter.is_none() {
                    Ok(())
                } else {
                    Err(error)
                }
            },
        )?;
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|error| self.database_error(error))?;
        if matches!(query_origin, RecallQueryOrigin::Synthetic { .. }) {
            self.validate_synthetic_sentinel(&transaction, current_user_event_id)?;
            if let Some(scope) = session_filter {
                let session = self.get_session_from_connection(&transaction, scope)?;
                self.verify_fresh(&session)?;
            }
        }
        let visibility = RecallVisibility {
            cutoff: match query_origin {
                RecallQueryOrigin::IndexedEvent => None,
                RecallQueryOrigin::Synthetic { reference_time } => Some(parse_retrieval_time(
                    reference_time,
                    "synthetic reference_time",
                )?),
            },
            allow_aggregate_documents: false,
        };
        let mut result = self.keyword_recall_core_from_connection(
            &transaction,
            raw_query,
            current_user_event_id,
            recent_event_ids,
            session_filter,
            config,
            &state,
            &visibility,
        )?;
        let recent = recent_event_ids.iter().map(String::as_str).collect();
        let mut used_events = result
            .evidence
            .iter()
            .map(|item| item.selected.span.event_id.clone())
            .collect();
        let mut used_hashes = result
            .evidence
            .iter()
            .map(|item| item.selected.content_sha256.clone())
            .collect();
        self.expand_context(
            &transaction,
            &mut result.trace,
            &mut result.evidence,
            current_user_event_id,
            &recent,
            &mut used_events,
            &mut used_hashes,
            session_filter,
            &state,
            &visibility,
        )?;
        self.validate_recall_evidence_active(&transaction, &state, &result, &visibility)?;
        self.require_unchanged_control_state(&state)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn keyword_recall_core_from_connection(
        &self,
        connection: &Connection,
        raw_query: &str,
        current_user_event_id: &str,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
        config: RetrievalConfig,
        control: &ControlState,
        visibility: &RecallVisibility,
    ) -> RetrievalResult<RecallResult> {
        self.validate_recall_input_id(connection, control, current_user_event_id, session_filter)?;
        for event_id in recent_event_ids {
            self.validate_recall_input_id(connection, control, event_id, session_filter)?;
        }
        config
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let terms = query_terms(raw_query);
        let query_kind = classify_query(raw_query);
        let mut trace = RetrievalTrace {
            status: "ok".into(),
            current_query_event_id: current_user_event_id.into(),
            query_terms: terms.clone(),
            config,
            query_kind,
            budget_allocation: BudgetAllocationTrace::for_query_kind(query_kind),
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
        let mut statement = connection.prepare(
            "SELECT d.document_id, d.granularity, d.event_id, d.start_char, d.end_char, d.content_sha256, d.exact_content,
                    e.role, e.session_id, e.created_at, bm25(retrieval_documents_fts) AS score
             FROM retrieval_documents_fts JOIN retrieval_documents d ON d.rowid = retrieval_documents_fts.rowid
             JOIN events e ON e.event_id = d.event_id
             WHERE retrieval_documents_fts MATCH ?1 AND (?2 IS NULL OR e.session_id=?2)
             ORDER BY score ASC, d.document_id ASC"
        ).map_err(|e| self.database_error(e))?;
        let rows = statement
            .query_map(params![expression, session_filter], |row| {
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
            })
            .map_err(|e| self.database_error(e))?;
        let recent: HashSet<&str> = recent_event_ids.iter().map(String::as_str).collect();
        let mut used_events = HashSet::new();
        let mut used_hashes = HashSet::new();
        let mut core_chars = 0usize;
        let mut active_rank = 0usize;
        // Keep the entire deterministic overfetch in the trace. Exclusions
        // must not consume the usable candidate pool.
        for row in rows {
            let row = row.map_err(|error| self.database_error(error))?;
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
            let active = control.allows_event(&session_id, &event_id_value);
            let visible = visibility.event_created_at_is_visible(&created_at)?;
            if active && visible {
                active_rank += 1;
            }
            let mut candidate = RankedCandidate {
                raw_rank: if active && visible { active_rank } else { 0 },
                document_id,
                granularity: if granularity == "fragment" {
                    RetrievalDocumentGranularity::Fragment
                } else {
                    RetrievalDocumentGranularity::Message
                },
                span: span.clone(),
                role,
                session_id: session_id.clone(),
                created_at,
                content_sha256: hash.clone(),
                bm25_score: score,
                selected: false,
                reason: String::new(),
            };
            if !active {
                candidate.reason = "control_excluded".into();
            } else if !visible {
                candidate.reason = "future_event".into();
            } else if event_id_value == current_user_event_id {
                candidate.reason = "current_message".into();
            } else if recent.contains(event_id_value.as_str()) {
                candidate.reason = "recent_context".into();
            } else if role == EventRole::System {
                candidate.reason = "system_message".into();
            } else if used_events.contains(&event_id_value) {
                candidate.reason = "duplicate_event".into();
            } else if used_hashes.contains(&hash) {
                candidate.reason = "duplicate_content".into();
            } else {
                let source_event = self.get_event_from_connection(connection, &span.event_id)?;
                let source_session =
                    self.get_session_from_connection(connection, &source_event.session_id)?;
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
                if final_hard_limit(trace.config.evidence_char_budget)
                    .is_some_and(|limit| span_content.chars().count() + core_chars > limit)
                {
                    candidate.reason = "evidence_budget".into();
                } else {
                    let selected_core_count = trace
                        .selected_evidence
                        .iter()
                        .filter(|e| e.kind == EvidenceKind::Core)
                        .count();
                    if selected_core_count >= trace.config.candidate_limit {
                        candidate.reason = "candidate_limit".into();
                    } else if selected_core_count
                        >= final_hard_limit(trace.config.max_selected).unwrap_or(usize::MAX)
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
            }
            trace.candidates.push(candidate);
            if final_hard_limit(trace.config.max_selected).is_none()
                && trace.selected_evidence.len() >= trace.config.candidate_limit
            {
                break;
            }
        }
        drop(statement);
        let mut evidence = Vec::new();
        for selected in trace.selected_evidence.clone() {
            let event = self.get_event_from_connection(connection, &selected.span.event_id)?;
            verify_event_hash(&event)?;
            let content = slice_chars(&event.content, &selected.span)?;
            evidence.push(RecalledEvidence { selected, content });
        }
        Ok(RecallResult { trace, evidence })
    }

    fn validate_recall_evidence_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        result: &RecallResult,
        visibility: &RecallVisibility,
    ) -> RetrievalResult<()> {
        for selected in &result.trace.selected_evidence {
            let event = self.get_event_from_connection(connection, &selected.span.event_id)?;
            Self::require_active_event(control, &event)?;
            if !visibility.event_created_at_is_visible(&event.created_at)? {
                return Err(RetrievalError::CorruptIndex(format!(
                    "召回证据 {} 超出 reference_time",
                    event.id
                )));
            }
        }
        for evidence in &result.evidence {
            let event =
                self.get_event_from_connection(connection, &evidence.selected.span.event_id)?;
            Self::require_active_event(control, &event)?;
            if !visibility.event_created_at_is_visible(&event.created_at)? {
                return Err(RetrievalError::CorruptIndex(format!(
                    "召回证据 {} 超出 reference_time",
                    event.id
                )));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_recall<B: ChatBackend>(
        &self,
        backend: &B,
        raw_query: &str,
        current_user_event_id: &str,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
        retrieval_config: RetrievalConfig,
        memory_config: &MemoryConfig,
    ) -> RetrievalResult<RecallResult> {
        let channels = if memory_config.enabled {
            RecallChannels::all()
        } else {
            RecallChannels::bm25_only()
        };
        self.hybrid_recall_with_options(
            backend,
            raw_query,
            current_user_event_id,
            recent_event_ids,
            session_filter,
            retrieval_config,
            memory_config,
            HybridRecallOptions {
                channels,
                query_origin: RecallQueryOrigin::IndexedEvent,
            },
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_recall_with_options<B: ChatBackend>(
        &self,
        backend: &B,
        raw_query: &str,
        current_user_event_id: &str,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
        retrieval_config: RetrievalConfig,
        memory_config: &MemoryConfig,
        options: HybridRecallOptions,
    ) -> RetrievalResult<RecallResult> {
        retrieval_config
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        memory_config
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        options
            .channels
            .validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.into()))?;
        let started = Instant::now();
        if let RecallQueryOrigin::Synthetic { reference_time } = &options.query_origin {
            parse_retrieval_time(reference_time, "synthetic reference_time")?;
        }
        if !memory_config.enabled {
            if options.channels != RecallChannels::bm25_only() {
                return Err(RetrievalError::CorruptIndex(
                    "memory.enabled=false only supports BM25-only recall".into(),
                ));
            }
            let channel_started = Instant::now();
            let mut result = self.keyword_recall_scoped_with_origin(
                raw_query,
                current_user_event_id,
                recent_event_ids,
                session_filter,
                retrieval_config,
                &options.query_origin,
            )?;
            let bm25_ms = elapsed_ms(channel_started);
            let query_kind = classify_query(raw_query);
            result.trace.query_kind = query_kind;
            result.trace.budget_allocation = memory_budget_trace(memory_config, query_kind);
            result.trace.channels = vec![
                channel_trace(
                    RetrievalChannel::Bm25,
                    "ok",
                    result.trace.candidates.len(),
                    bm25_ms,
                    None,
                ),
                channel_trace(RetrievalChannel::Vector, "disabled", 0, 0, None),
                channel_trace(RetrievalChannel::Entity, "disabled", 0, 0, None),
                channel_trace(RetrievalChannel::State, "disabled", 0, 0, None),
                channel_trace(RetrievalChannel::Episode, "disabled", 0, 0, None),
                channel_trace(RetrievalChannel::Graph, "disabled", 0, 0, None),
            ];
            result.trace.elapsed_ms = elapsed_ms(started);
            return Ok(result);
        }

        if options.channels == RecallChannels::bm25_only() {
            let channel_started = Instant::now();
            let mut result = self.keyword_recall_scoped_with_origin(
                raw_query,
                current_user_event_id,
                recent_event_ids,
                session_filter,
                retrieval_config,
                &options.query_origin,
            )?;
            let query_kind = classify_query(raw_query);
            result.trace.query_kind = query_kind;
            result.trace.budget_allocation = memory_budget_trace(memory_config, query_kind);
            result.trace.channels = requested_channel_traces(
                options.channels,
                "ok",
                result.trace.candidates.len(),
                elapsed_ms(channel_started),
            );
            result.trace.elapsed_ms = elapsed_ms(started);
            return Ok(result);
        }

        {
            let _guard = self.acquire_root_read()?;
            let control = self.replay_control_state_under_guard()?;
            if let Some(session_id) = session_filter {
                Self::require_active_session(&control, session_id)?;
            }
            let mut connection = self.open_connection()?;
            let transaction = connection
                .transaction_with_behavior(TransactionBehavior::Deferred)
                .map_err(|error| self.database_error(error))?;
            match &options.query_origin {
                RecallQueryOrigin::IndexedEvent => self.validate_recall_input_id(
                    &transaction,
                    &control,
                    current_user_event_id,
                    session_filter,
                )?,
                RecallQueryOrigin::Synthetic { reference_time } => {
                    parse_retrieval_time(reference_time, "synthetic reference_time")?;
                    self.validate_synthetic_sentinel(&transaction, current_user_event_id)?;
                    if let Some(scope) = session_filter {
                        let session = self.get_session_from_connection(&transaction, scope)?;
                        self.verify_fresh(&session)?;
                    }
                }
            }
            for event_id in recent_event_ids {
                self.validate_recall_input_id(&transaction, &control, event_id, session_filter)?;
            }
            transaction
                .commit()
                .map_err(|error| self.database_error(error))?;
            self.require_unchanged_control_state(&control)?;
        }

        let spec = VectorIndexSpec::from_config(memory_config)
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let query_kind = classify_query(raw_query);
        let decided_budget = memory_budget_trace(memory_config, query_kind);
        let embed_started = Instant::now();
        let embedded = tokio::time::timeout(
            Duration::from_secs(memory_config.embedding_timeout_secs),
            backend.embed(EmbeddingRequest {
                model: spec.model.clone(),
                input: vec![raw_query.to_owned()],
                dimensions: None,
                truncate: false,
            }),
        )
        .await;
        let vector_elapsed = elapsed_ms(embed_started);
        let query_vector = match validate_query_embedding(embedded, &spec) {
            Ok(input) => input,
            Err(error) => {
                if !options.channels.bm25 {
                    return Err(error);
                }
                let bm25_started = Instant::now();
                let mut fallback = self.keyword_recall_scoped_with_origin(
                    raw_query,
                    current_user_event_id,
                    recent_event_ids,
                    session_filter,
                    retrieval_config,
                    &options.query_origin,
                )?;
                let bm25_ms = elapsed_ms(bm25_started);
                fallback.trace.query_kind = query_kind;
                fallback.trace.budget_allocation = decided_budget;
                apply_vector_fallback(
                    &mut fallback,
                    bm25_ms,
                    vector_elapsed,
                    error.to_string(),
                    options.channels,
                );
                fallback.trace.elapsed_ms = elapsed_ms(started);
                return Ok(fallback);
            }
        };
        let fusion_store = self.clone();
        let fusion_query = raw_query.to_owned();
        let fusion_scope = session_filter.map(str::to_owned);
        let fusion_current = current_user_event_id.to_owned();
        let fusion_recent = recent_event_ids.to_vec();
        let retrieval = retrieval_config.clone();
        let memory = memory_config.clone();
        let fusion_spec = spec.clone();
        let fusion_options = options.clone();
        let fused_task = tokio::task::spawn_blocking(move || {
            fusion_store
                .fuse_vector_recall(
                    &fusion_query,
                    query_vector,
                    &fusion_spec,
                    &fusion_current,
                    &fusion_recent,
                    fusion_scope.as_deref(),
                    retrieval,
                    memory,
                    vector_elapsed,
                    fusion_options,
                )
                .map_err(|error| match error {
                    RetrievalError::HybridRecall(_) => error,
                    other => RetrievalError::HybridRecall(HybridRecallFailure::new(
                        HybridRecallStage::Vector,
                        other,
                    )),
                })
        });
        let mut result = match join_blocking(fused_task).await {
            Ok(result) => result,
            Err(error) => {
                if !options.channels.bm25 {
                    return Err(error);
                }
                let bm25_started = Instant::now();
                let mut fallback = self.keyword_recall_scoped_with_origin(
                    raw_query,
                    current_user_event_id,
                    recent_event_ids,
                    session_filter,
                    retrieval_config,
                    &options.query_origin,
                )?;
                let bm25_ms = elapsed_ms(bm25_started);
                fallback.trace.query_kind = query_kind;
                fallback.trace.budget_allocation = decided_budget;
                let message = error.to_string();
                let graph_elapsed = match &error {
                    RetrievalError::HybridRecall(failure) => failure.elapsed_ms,
                    _ => 0,
                };
                if matches!(&error, RetrievalError::HybridRecall(failure) if failure.stage == HybridRecallStage::Graph)
                {
                    apply_graph_fallback(
                        &mut fallback,
                        bm25_ms,
                        elapsed_ms(embed_started),
                        graph_elapsed,
                        message,
                        options.channels,
                    );
                } else if matches!(&error, RetrievalError::HybridRecall(failure) if failure.stage == HybridRecallStage::EntityState)
                {
                    apply_sidecar_fallback(
                        &mut fallback,
                        bm25_ms,
                        elapsed_ms(embed_started),
                        message,
                        options.channels,
                    );
                } else {
                    apply_vector_fallback(
                        &mut fallback,
                        bm25_ms,
                        elapsed_ms(embed_started),
                        message,
                        options.channels,
                    );
                }
                fallback.trace.elapsed_ms = elapsed_ms(started);
                return Ok(fallback);
            }
        };
        if let Some(channel) = result
            .trace
            .channels
            .iter_mut()
            .find(|channel| channel.channel == RetrievalChannel::Vector)
        {
            channel.elapsed_ms = elapsed_ms(embed_started);
        }
        result.trace.elapsed_ms = elapsed_ms(started);
        Ok(result)
    }

    fn load_vector_index_from_connection(
        &self,
        connection: &Connection,
        spec: &VectorIndexSpec,
        session_filter: Option<&str>,
        visibility: &RecallVisibility,
    ) -> RetrievalResult<(Arc<HnswVectorIndex>, Option<PendingVectorIndexCache>)> {
        spec.validate()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let control = self.replay_control_state_under_guard()?;
        let total = self.active_memory_document_count(connection, &control, session_filter)?;
        let fingerprint = spec
            .fingerprint()
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let rows = self.compatible_embeddings_from_connection(
            connection,
            spec,
            &fingerprint,
            session_filter,
        )?;
        if total != rows.len() {
            return Err(RetrievalError::CorruptIndex(format!(
                "向量目录不完整：文档总数 {total}，兼容 embedding 数 {}",
                rows.len()
            )));
        }
        let query_filtered = visibility.cutoff.is_some() || !visibility.allow_aggregate_documents;
        if query_filtered {
            let mut filtered = Vec::with_capacity(rows.len());
            for row in rows {
                if visibility.document_is_visible(
                    self,
                    connection,
                    &row.document_id,
                    row.granularity,
                )? {
                    filtered.push(row);
                }
            }
            return HnswVectorIndex::rebuild(spec.clone(), filtered)
                .map(Arc::new)
                .map(|index| (index, None))
                .map_err(|error| RetrievalError::CorruptIndex(error.to_string()));
        }
        if session_filter.is_some() {
            return HnswVectorIndex::rebuild(spec.clone(), rows)
                .map(Arc::new)
                .map(|index| (index, None))
                .map_err(|error| RetrievalError::CorruptIndex(error.to_string()));
        }
        let catalog_sha256 = embedding_catalog_identity(&rows);
        let observed_identity = {
            let cache = self
                .vector_cache
                .lock()
                .map_err(|_| RetrievalError::VectorCachePoisoned)?;
            if let Some(cache) = cache.as_ref()
                && cache.fingerprint == fingerprint
                && cache.catalog_sha256 == catalog_sha256
            {
                return Ok((Arc::clone(&cache.index), None));
            }
            cache
                .as_ref()
                .map(|cache| (cache.fingerprint.clone(), cache.catalog_sha256.clone()))
        };
        let index = Arc::new(
            HnswVectorIndex::rebuild(spec.clone(), rows)
                .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?,
        );
        let pending = PendingVectorIndexCache {
            fingerprint,
            catalog_sha256,
            index: Arc::clone(&index),
            observed_identity,
        };
        Ok((index, Some(pending)))
    }

    fn publish_vector_index_cache(&self, pending: PendingVectorIndexCache) -> RetrievalResult<()> {
        let mut cache = self
            .vector_cache
            .lock()
            .map_err(|_| RetrievalError::VectorCachePoisoned)?;
        let current_identity = cache
            .as_ref()
            .map(|cache| (cache.fingerprint.clone(), cache.catalog_sha256.clone()));
        let pending_identity = (pending.fingerprint.clone(), pending.catalog_sha256.clone());
        if current_identity.as_ref() == Some(&pending_identity) {
            return Ok(());
        }
        if current_identity != pending.observed_identity {
            return Ok(());
        }
        *cache = Some(VectorIndexCache {
            fingerprint: pending.fingerprint,
            catalog_sha256: pending.catalog_sha256,
            index: pending.index,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fuse_vector_recall(
        &self,
        raw_query: &str,
        query_vector: Vec<f32>,
        spec: &VectorIndexSpec,
        current_user_event_id: &str,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
        retrieval_config: RetrievalConfig,
        memory_config: MemoryConfig,
        vector_ms: u64,
        options: HybridRecallOptions,
    ) -> RetrievalResult<RecallResult> {
        let visibility = RecallVisibility::from_options(&options)?;
        let _guard = self.acquire_root_read()?;
        let control = self.replay_control_state_under_guard()?;
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|error| self.database_error(error))?;
        match &options.query_origin {
            RecallQueryOrigin::IndexedEvent => self.validate_recall_input_id(
                &transaction,
                &control,
                current_user_event_id,
                session_filter,
            )?,
            RecallQueryOrigin::Synthetic { .. } => {
                self.validate_synthetic_sentinel(&transaction, current_user_event_id)?;
                if let Some(scope) = session_filter {
                    let session = self.get_session_from_connection(&transaction, scope)?;
                    self.verify_fresh(&session)?;
                }
            }
        }
        for event_id in recent_event_ids {
            self.validate_recall_input_id(&transaction, &control, event_id, session_filter)?;
        }
        let bm25_started = Instant::now();
        let mut bm25 = if options.channels.bm25 {
            self.keyword_recall_core_from_connection(
                &transaction,
                raw_query,
                current_user_event_id,
                recent_event_ids,
                session_filter,
                retrieval_config.clone(),
                &control,
                &visibility,
            )?
        } else {
            empty_recall_base(raw_query, current_user_event_id, retrieval_config.clone())
        };
        let bm25_ms = if options.channels.bm25 {
            elapsed_ms(bm25_started)
        } else {
            0
        };
        self.require_current_control_projection(&transaction, &control)?;
        bm25.trace.budget_allocation = memory_budget_trace(&memory_config, bm25.trace.query_kind);
        let (index, pending_vector_cache) = self.load_vector_index_from_connection(
            &transaction,
            spec,
            session_filter,
            &visibility,
        )?;
        if index.is_empty() {
            return Err(RetrievalError::CorruptIndex("没有兼容的向量索引".into()));
        }
        let hits = index
            .search(&query_vector, memory_config.vector_candidate_limit)
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let vector_aggregate_sources = hits
            .iter()
            .filter(|hit| {
                matches!(
                    hit.granularity,
                    RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session
                )
            })
            .map(|hit| hit.document_id.clone())
            .collect::<BTreeSet<_>>();
        let mut bm25_document_ids = BTreeSet::new();
        for candidate in &bm25.trace.candidates {
            if candidate.document_id.is_empty()
                || !bm25_document_ids.insert(candidate.document_id.clone())
            {
                return Err(RetrievalError::CorruptIndex(
                    "BM25 candidate 文档 ID 为空或重复".into(),
                ));
            }
        }
        let mut bm25_seed_rows = Vec::new();
        for candidate in bm25.trace.candidates.iter().filter(|candidate| {
            matches!(
                candidate.reason.as_str(),
                "selected_core" | "evidence_budget" | "selection_limit"
            )
        }) {
            if candidate.span.event_id == current_user_event_id
                || recent_event_ids
                    .iter()
                    .any(|id| id == &candidate.span.event_id)
                || candidate.role == EventRole::System
                || session_filter.is_some_and(|scope| candidate.session_id != scope)
            {
                continue;
            }
            bm25_seed_rows.push(candidate);
        }
        let mut graph_seeds = bm25_seed_rows
            .into_iter()
            .enumerate()
            .map(|(index, candidate)| GraphRecallSeed {
                channel: RetrievalChannel::Bm25,
                source_id: candidate.document_id.clone(),
                document_id: Some(candidate.document_id.clone()),
                rank: index + 1,
                score: candidate.bm25_score,
            })
            .collect::<Vec<_>>();
        let recent_set = recent_event_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut projected = Vec::new();
        let mut vector_seed_rows = Vec::new();
        let mut vector_seed_documents = BTreeSet::new();
        for (rank_index, hit) in hits.iter().enumerate() {
            if hit.document_id.is_empty() || !vector_seed_documents.insert(hit.document_id.clone())
            {
                return Err(RetrievalError::CorruptIndex(
                    "Vector graph seed 文档 ID 为空或重复".into(),
                ));
            }
            let mut hit_projection = self.project_vector_hit(
                &transaction,
                &index,
                &query_vector,
                hit,
                rank_index + 1,
                usize::MAX,
            )?;
            if !options.channels.episode {
                for projection in &mut hit_projection {
                    projection.episode_id = None;
                }
            }
            let blocked = hit_projection.iter().any(|raw| {
                raw.span.event_id == current_user_event_id
                    || recent_set.contains(raw.span.event_id.as_str())
                    || raw.role == EventRole::System
                    || session_filter.is_some_and(|scope| raw.session_id != scope)
            });
            if !blocked {
                vector_seed_rows.push(hit);
            }
            hit_projection.truncate(memory_config.candidate_limit);
            projected.extend(hit_projection);
        }
        graph_seeds.extend(
            vector_seed_rows
                .into_iter()
                .enumerate()
                .map(|(index, hit)| GraphRecallSeed {
                    channel: RetrievalChannel::Vector,
                    source_id: hit.document_id.clone(),
                    document_id: Some(hit.document_id.clone()),
                    rank: index + 1,
                    score: f64::from(hit.cosine_similarity),
                }),
        );
        let mut fused = HashMap::<(String, usize, usize), FusedRawCandidate>::new();
        let mut excluded_vectors = Vec::new();
        let eligible = ["selected_core", "evidence_budget", "selection_limit"];
        let protected_span = bm25
            .trace
            .selected_evidence
            .iter()
            .find(|evidence| evidence.kind == EvidenceKind::Core)
            .map(|evidence| evidence.span.clone());
        for candidate in bm25
            .trace
            .candidates
            .iter()
            .filter(|candidate| eligible.contains(&candidate.reason.as_str()))
        {
            let event = self.get_event_from_connection(&transaction, &candidate.span.event_id)?;
            let content = slice_chars(&event.content, &candidate.span)?;
            let key = raw_candidate_key(&candidate.span);
            let is_protected = protected_span.as_ref() == Some(&candidate.span);
            let episode_id = if options.channels.episode {
                self.resolve_episode_id(&transaction, &candidate.span.event_id)?
            } else {
                None
            };
            let entry = fused.entry(key).or_insert_with(|| FusedRawCandidate {
                pre_cap_rank: 0,
                document_id: candidate.document_id.clone(),
                granularity: candidate.granularity,
                span: candidate.span.clone(),
                role: candidate.role,
                session_id: candidate.session_id.clone(),
                content_sha256: candidate.content_sha256.clone(),
                content,
                episode_id,
                source_document_ids: vec![candidate.document_id.clone()],
                bm25_rank: None,
                bm25_score: None,
                bm25_contribution: 0.0,
                vector_rank: None,
                vector_score: None,
                vector_contribution: 0.0,
                vector_source_document_id: None,
                rrf_score: 0.0,
                protected_exact: is_protected,
                selected: false,
                reason: String::new(),
                vector: index
                    .vector_for_document(&candidate.document_id)
                    .map(<[f32]>::to_vec),
            });
            entry
                .source_document_ids
                .push(candidate.document_id.clone());
            if entry.bm25_rank.is_none_or(|rank| candidate.raw_rank < rank) {
                entry.bm25_rank = Some(candidate.raw_rank);
                entry.bm25_score = Some(candidate.bm25_score);
                entry.bm25_contribution = rrf(memory_config.rrf_k, candidate.raw_rank);
            }
            entry.protected_exact |= is_protected;
        }
        for candidate in projected {
            let exclusion_reason = if candidate.span.event_id == current_user_event_id {
                Some("current_message")
            } else if recent_event_ids
                .iter()
                .any(|id| id == &candidate.span.event_id)
            {
                Some("recent_context")
            } else if candidate.role == EventRole::System {
                Some("system_message")
            } else {
                None
            };
            if let Some(reason) = exclusion_reason {
                excluded_vectors.push(FusionCandidateTrace {
                    fused_rank: 0,
                    document_id: candidate.document_id,
                    span: candidate.span,
                    session_id: candidate.session_id,
                    granularity: candidate.granularity,
                    source_document_ids: vec![candidate.source_document_id],
                    episode_id: candidate.episode_id,
                    bm25_rank: None,
                    bm25_score: None,
                    vector_rank: Some(candidate.vector_rank),
                    vector_score: Some(candidate.similarity),
                    rrf_score: 0.0,
                    protected_exact: false,
                    selected: false,
                    reason: reason.into(),
                });
                continue;
            }
            let key = raw_candidate_key(&candidate.span);
            let entry = fused.entry(key).or_insert_with(|| FusedRawCandidate {
                pre_cap_rank: 0,
                document_id: candidate.document_id.clone(),
                granularity: candidate.granularity,
                span: candidate.span.clone(),
                role: candidate.role,
                session_id: candidate.session_id.clone(),
                content_sha256: candidate.content_sha256.clone(),
                content: candidate.content.clone(),
                episode_id: candidate.episode_id.clone(),
                source_document_ids: Vec::new(),
                bm25_rank: None,
                bm25_score: None,
                bm25_contribution: 0.0,
                vector_rank: None,
                vector_score: None,
                vector_contribution: 0.0,
                vector_source_document_id: None,
                rrf_score: 0.0,
                protected_exact: false,
                selected: false,
                reason: String::new(),
                vector: Some(candidate.vector.clone()),
            });
            entry
                .source_document_ids
                .push(candidate.source_document_id.clone());
            let contribution = rrf(memory_config.rrf_k, candidate.vector_rank)
                / candidate.contribution_divisor as f64;
            let vector_is_better = contribution > entry.vector_contribution
                || (contribution == entry.vector_contribution
                    && entry.vector_rank.is_none_or(|rank| {
                        candidate.vector_rank < rank
                            || (candidate.vector_rank == rank
                                && entry.vector_score.is_none_or(|score| {
                                    candidate.similarity > score
                                        || (candidate.similarity == score
                                            && entry.vector_source_document_id.as_ref().is_none_or(
                                                |source| candidate.source_document_id < *source,
                                            ))
                                }))
                    }));
            if vector_is_better {
                entry.vector_rank = Some(candidate.vector_rank);
                entry.vector_score = Some(candidate.similarity);
                entry.vector_contribution = contribution;
                entry.vector_source_document_id = Some(candidate.source_document_id);
                entry.vector = Some(candidate.vector);
            }
            if entry.episode_id.is_none() {
                entry.episode_id = candidate.episode_id;
            }
        }
        let mut candidates = fused.into_values().collect::<Vec<_>>();
        for candidate in &mut candidates {
            candidate.source_document_ids.sort();
            candidate.source_document_ids.dedup();
            candidate.rrf_score = candidate.bm25_contribution + candidate.vector_contribution;
        }
        candidates.sort_by(|left, right| {
            right
                .rrf_score
                .total_cmp(&left.rrf_score)
                .then_with(|| left.document_id.cmp(&right.document_id))
        });
        for (index, candidate) in candidates.iter_mut().enumerate() {
            candidate.pre_cap_rank = index + 1;
        }
        let vector_candidate_count = candidates
            .iter()
            .filter(|candidate| candidate.vector_rank.is_some())
            .count()
            + excluded_vectors.len();
        let mut capped_candidates = Vec::new();
        if candidates.len() > memory_config.candidate_limit {
            capped_candidates = candidates.split_off(memory_config.candidate_limit);
            let protected = capped_candidates
                .iter()
                .position(|candidate| candidate.protected_exact)
                .map(|index| capped_candidates.remove(index));
            if let Some(protected_candidate) = protected {
                if let Some(displaced) = candidates.pop() {
                    capped_candidates.push(displaced);
                }
                candidates.push(protected_candidate);
                candidates.sort_by(|left, right| {
                    right
                        .rrf_score
                        .total_cmp(&left.rrf_score)
                        .then_with(|| left.document_id.cmp(&right.document_id))
                });
            }
        }
        for candidate in &mut capped_candidates {
            candidate.selected = false;
            candidate.reason = "candidate_limit".into();
        }
        capped_candidates.sort_by_key(|candidate| candidate.pre_cap_rank);
        excluded_vectors.sort_by(|left, right| {
            left.vector_rank
                .cmp(&right.vector_rank)
                .then_with(|| left.document_id.cmp(&right.document_id))
                .then_with(|| left.span.event_id.cmp(&right.span.event_id))
                .then_with(|| left.span.start_char.cmp(&right.span.start_char))
                .then_with(|| left.span.end_char.cmp(&right.span.end_char))
                .then_with(|| left.reason.cmp(&right.reason))
        });
        let mut sidecar = if options.channels.state {
            self.load_state_sidecar(
                &transaction,
                &control,
                raw_query,
                current_user_event_id,
                recent_event_ids,
                session_filter,
                bm25.trace.query_kind,
                memory_config.candidate_limit,
                &options.query_origin,
                options.channels.episode,
                &visibility,
            )
            .map_err(|error| HybridRecallFailure::new(HybridRecallStage::EntityState, error))?
        } else if options.channels.entity {
            let entity_started = Instant::now();
            let (entity_matches, _) = self
                .load_entity_matches(
                    &transaction,
                    &control,
                    raw_query,
                    session_filter,
                    &visibility,
                )
                .map_err(|error| HybridRecallFailure::new(HybridRecallStage::EntityState, error))?;
            StateSidecar {
                entity_matches,
                candidates: Vec::new(),
                warnings: Vec::new(),
                entity_ms: elapsed_ms(entity_started),
                state_ms: 0,
            }
        } else {
            empty_state_sidecar()
        };
        if !options.channels.state {
            sidecar.candidates.clear();
        }
        if !options.channels.episode {
            for candidate in &mut sidecar.candidates {
                candidate.episode_id = None;
            }
        }
        if options.channels.entity {
            graph_seeds.extend(entity_graph_seeds(&sidecar.entity_matches));
        } else {
            sidecar.entity_matches.clear();
            sidecar.entity_ms = 0;
        }
        let graph_started = Instant::now();
        let graph_seed_documents = graph_seeds
            .iter()
            .filter_map(|seed| seed.document_id.clone())
            .collect::<BTreeSet<_>>();
        let mut graph_sidecar = if options.channels.graph {
            let graph = crate::graph::recall_graph_from_connection(
                self,
                &transaction,
                &memory_config,
                &graph_seeds,
                session_filter,
                crate::graph::GraphRecallFilter {
                    cutoff: visibility.cutoff(),
                    allow_aggregate_documents: visibility.allows_aggregate_documents(),
                },
            )
            .map_err(|error| {
                HybridRecallFailure::timed(HybridRecallStage::Graph, error, graph_started)
            })?;
            let graph_ms = elapsed_ms(graph_started);
            bm25.trace.budget_allocation =
                memory_budget_trace(&memory_config, bm25.trace.query_kind);
            self.prepare_graph_candidates(
                &transaction,
                &index,
                &query_vector,
                current_user_event_id,
                recent_event_ids,
                session_filter,
                &graph_seed_documents,
                &vector_aggregate_sources,
                graph,
                graph_ms,
                options.channels.episode,
            )
            .map_err(|error| {
                HybridRecallFailure::timed(HybridRecallStage::Graph, error, graph_started)
            })?
        } else {
            empty_graph_sidecar()
        };
        if options.channels.episode && !options.channels.graph {
            graph_sidecar.aggregate_source_count = vector_aggregate_sources.len();
        }
        let mut result = self
            .select_fused_candidates(
                &transaction,
                candidates,
                bm25,
                retrieval_config.clone(),
                bm25_ms,
                vector_ms,
                vector_candidate_count,
                capped_candidates,
                excluded_vectors,
                sidecar,
                graph_sidecar,
                memory_config.graph_candidate_limit,
                options.channels,
            )
            .map_err(|error| HybridRecallFailure::new(HybridRecallStage::EntityState, error))?;
        let recent = recent_event_ids.iter().map(String::as_str).collect();
        let mut expansion_events = result
            .evidence
            .iter()
            .map(|item| item.selected.span.event_id.clone())
            .collect();
        let mut expansion_hashes = result
            .evidence
            .iter()
            .map(|item| item.selected.content_sha256.clone())
            .collect();
        self.expand_context(
            &transaction,
            &mut result.trace,
            &mut result.evidence,
            current_user_event_id,
            &recent,
            &mut expansion_events,
            &mut expansion_hashes,
            session_filter,
            &control,
            &visibility,
        )
        .map_err(|error| {
            HybridRecallFailure::timed(HybridRecallStage::Graph, error, graph_started)
        })?;
        self.validate_recall_evidence_active(&transaction, &control, &result, &visibility)?;
        self.require_unchanged_control_state(&control)?;
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        if let Some(pending) = pending_vector_cache {
            self.publish_vector_index_cache(pending)?;
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_graph_candidates(
        &self,
        connection: &Connection,
        index: &HnswVectorIndex,
        query_vector: &[f32],
        current_user_event_id: &str,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
        seed_documents: &BTreeSet<String>,
        vector_aggregate_sources: &BTreeSet<String>,
        mut graph: crate::graph::GraphRecallResult,
        graph_ms: u64,
        episode_enabled: bool,
    ) -> RetrievalResult<GraphSidecar> {
        let recent = recent_event_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut aggregate_sources = vector_aggregate_sources.clone();
        let mut candidates = Vec::new();
        for (path_index, path) in graph.paths.iter_mut().enumerate() {
            if seed_documents.contains(&path.target_document_id) {
                path.reason = "seed_document".into();
                continue;
            }
            let row = connection.query_row("SELECT session_id,granularity,source_sha256 FROM memory_documents WHERE document_id=?1", [path.target_document_id.as_str()], |row| Ok((row.get::<_,String>(0)?, row.get::<_,String>(1)?, row.get::<_,String>(2)?))).optional().map_err(|error| self.database_error(error))?.ok_or_else(|| RetrievalError::CorruptIndex(format!("图 target 文档缺失：{}", path.target_document_id)))?;
            let granularity = parse_granularity(&row.1)?;
            if !episode_enabled
                && matches!(
                    granularity,
                    RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session
                )
            {
                path.reason = "episode_channel_disabled".into();
                continue;
            }
            if matches!(
                granularity,
                RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session
            ) {
                aggregate_sources.insert(path.target_document_id.clone());
            }
            let hit = crate::vector::VectorSearchHit {
                document_id: path.target_document_id.clone(),
                session_id: row.0,
                granularity,
                source_sha256: row.2,
                cosine_similarity: 0.0,
                cosine_distance: 1.0,
            };
            let raw = self.project_graph_target(
                connection,
                index,
                query_vector,
                &hit,
                path.target_rank,
                current_user_event_id,
                &recent,
                session_filter,
            )?;
            let Some(raw) = raw else {
                path.reason = "no_eligible_raw_member".into();
                continue;
            };
            path.target_granularity = Some(granularity);
            path.target_session_id = raw.session_id.clone();
            path.span = Some(raw.span.clone());
            path.content_sha256 = raw.content_sha256.clone();
            path.role = Some(raw.role);
            path.reason = if raw.span.event_id == current_user_event_id {
                "current_message"
            } else if recent.contains(raw.span.event_id.as_str()) {
                "recent_context"
            } else if raw.role == EventRole::System {
                "system_message"
            } else if session_filter.is_some_and(|scope| raw.session_id != scope) {
                "scope_mismatch"
            } else {
                "eligible"
            }
            .into();
            if path.reason != "eligible" {
                continue;
            }
            candidates.push(GraphEvidenceCandidate { path_index, raw });
        }
        if !episode_enabled {
            graph
                .paths
                .retain(|path| path.reason != "episode_channel_disabled");
        }
        let candidate_count = graph.paths.len();
        Ok(GraphSidecar {
            paths: graph.paths,
            candidates,
            candidate_count,
            aggregate_source_count: aggregate_sources.len(),
            elapsed_ms: graph_ms,
            warning: graph.warning,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn project_graph_target(
        &self,
        connection: &Connection,
        index: &HnswVectorIndex,
        query_vector: &[f32],
        hit: &crate::vector::VectorSearchHit,
        rank: usize,
        current: &str,
        recent: &HashSet<&str>,
        session_filter: Option<&str>,
    ) -> RetrievalResult<Option<ProjectedVectorCandidate>> {
        if matches!(
            hit.granularity,
            RetrievalDocumentGranularity::Message | RetrievalDocumentGranularity::Fragment
        ) {
            return self
                .project_vector_hit(connection, index, query_vector, hit, rank, 1)
                .map(|mut rows| rows.pop());
        }
        validate_aggregate_document_source(
            connection,
            &hit.document_id,
            &hit.session_id,
            &hit.source_sha256,
        )?;
        let mut statement = connection.prepare(
            "SELECT m.ordinal,m.event_id,m.start_char,m.end_char,m.content_sha256 FROM memory_document_members m WHERE m.document_id=?1 ORDER BY m.ordinal"
        ).map_err(|error| self.database_error(error))?;
        let members = statement
            .query_map([hit.document_id.as_str()], |row| {
                Ok((
                    i64_to_usize(row.get(0)?)?,
                    row.get::<_, String>(1)?,
                    i64_to_usize(row.get(2)?)?,
                    i64_to_usize(row.get(3)?)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .map_err(|error| self.database_error(error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| self.database_error(error))?;
        let stored_count: usize = connection
            .query_row(
                "SELECT member_count FROM memory_documents WHERE document_id=?1",
                [hit.document_id.as_str()],
                |row| i64_to_usize(row.get(0)?),
            )
            .map_err(|error| self.database_error(error))?;
        if members.len() != stored_count {
            return Err(RetrievalError::CorruptIndex(format!(
                "聚合文档 {} member_count 不匹配",
                hit.document_id
            )));
        }
        let mut eligible = Vec::new();
        for (ordinal, event_id, start, end, hash) in members {
            let message_document_id = connection.query_row(
                "SELECT d.document_id FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id WHERE d.session_id=?1 AND d.granularity='message' AND d.member_count=1 AND m.ordinal=0 AND m.event_id=?2 AND m.start_char=?3 AND m.end_char=?4 AND m.content_sha256=?5",
                params![hit.session_id,event_id,usize_to_i64(start).map_err(|error| self.database_error(error))?,usize_to_i64(end).map_err(|error| self.database_error(error))?,hash], |row| row.get::<_,String>(0)
            ).optional().map_err(|error| self.database_error(error))?.ok_or_else(|| RetrievalError::CorruptIndex(format!("聚合文档 {} 成员缺少 message 文档", hit.document_id)))?;
            let vector = index
                .vector_for_document(&message_document_id)
                .ok_or_else(|| {
                    RetrievalError::CorruptIndex(format!(
                        "图聚合成员 {message_document_id} 缺少兼容向量"
                    ))
                })?;
            let similarity = exact_cosine_f64(query_vector, vector)?;
            let raw = self.load_projected_raw(
                connection,
                &message_document_id,
                &hit.document_id,
                RetrievalDocumentGranularity::Message,
                rank,
                similarity,
                stored_count,
                vector,
            )?;
            if raw.span.event_id != current
                && !recent.contains(raw.span.event_id.as_str())
                && raw.role != EventRole::System
                && session_filter.is_none_or(|scope| raw.session_id == scope)
            {
                eligible.push((
                    similarity,
                    ordinal,
                    raw.span.event_id.clone(),
                    message_document_id,
                    raw,
                ));
            }
        }
        eligible.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.3.cmp(&right.3))
        });
        Ok(eligible.into_iter().next().map(|row| row.4))
    }

    fn project_vector_hit(
        &self,
        connection: &Connection,
        index: &HnswVectorIndex,
        query_vector: &[f32],
        hit: &crate::vector::VectorSearchHit,
        vector_rank: usize,
        aggregate_limit: usize,
    ) -> RetrievalResult<Vec<ProjectedVectorCandidate>> {
        if matches!(
            hit.granularity,
            RetrievalDocumentGranularity::Message | RetrievalDocumentGranularity::Fragment
        ) {
            let vector = index
                .vector_for_document(&hit.document_id)
                .ok_or_else(|| RetrievalError::CorruptIndex("HNSW 文档缺少保留向量".into()))?;
            return self
                .load_projected_raw(
                    connection,
                    &hit.document_id,
                    &hit.document_id,
                    hit.granularity,
                    vector_rank,
                    f64::from(hit.cosine_similarity),
                    1,
                    vector,
                )
                .map(|candidate| vec![candidate]);
        }
        validate_aggregate_document_source(
            connection,
            &hit.document_id,
            &hit.session_id,
            &hit.source_sha256,
        )?;
        let member_count = connection
            .query_row(
                "SELECT member_count FROM memory_documents WHERE document_id=?1 AND session_id=?2 AND granularity=?3 AND source_sha256=?4",
                params![hit.document_id, hit.session_id, granularity_name(hit.granularity), hit.source_sha256],
                |row| i64_to_usize(row.get(0)?),
            )
            .optional()
            .map_err(|error| self.database_error(error))?
            .ok_or_else(|| RetrievalError::CorruptIndex(format!("聚合文档 {} 元数据不匹配", hit.document_id)))?;
        let mut statement = connection
            .prepare(
                "SELECT m.event_id,m.start_char,m.end_char,m.content_sha256
                 FROM memory_document_members m WHERE m.document_id=?1 ORDER BY m.ordinal",
            )
            .map_err(|error| self.database_error(error))?;
        let members = statement
            .query_map([hit.document_id.as_str()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    i64_to_usize(row.get(1)?)?,
                    i64_to_usize(row.get(2)?)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(|error| self.database_error(error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| self.database_error(error))?;
        if members.len() != member_count {
            return Err(RetrievalError::CorruptIndex(format!(
                "聚合文档 {} 的 member_count 不匹配",
                hit.document_id
            )));
        }
        let mut ranked = Vec::new();
        for (event_id, start, end, expected_hash) in members {
            let message_document_id = connection
                .query_row(
                    "SELECT d.document_id FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id
                     WHERE d.session_id=?1 AND d.granularity='message' AND d.member_count=1
                       AND m.ordinal=0 AND m.event_id=?2 AND m.start_char=?3 AND m.end_char=?4 AND m.content_sha256=?5",
                    params![hit.session_id, event_id, usize_to_i64(start).map_err(|error| self.database_error(error))?, usize_to_i64(end).map_err(|error| self.database_error(error))?, expected_hash],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(|error| self.database_error(error))?
                .ok_or_else(|| RetrievalError::CorruptIndex(format!("聚合文档 {} 的成员缺少直接 message 文档", hit.document_id)))?;
            let Some(vector) = index.vector_for_document(&message_document_id) else {
                return Err(RetrievalError::CorruptIndex(format!(
                    "聚合文档 {} 的成员 {} 缺少兼容 message 向量",
                    hit.document_id, message_document_id
                )));
            };
            ranked.push((
                exact_cosine_f64(query_vector, vector)?,
                message_document_id,
                vector.to_vec(),
            ));
        }
        ranked.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        ranked.truncate(aggregate_limit);
        ranked
            .into_iter()
            .map(|(similarity, message_document_id, vector)| {
                self.load_projected_raw(
                    connection,
                    &message_document_id,
                    &hit.document_id,
                    RetrievalDocumentGranularity::Message,
                    vector_rank,
                    similarity,
                    member_count,
                    &vector,
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn load_projected_raw(
        &self,
        connection: &Connection,
        document_id: &str,
        source_document_id: &str,
        granularity: RetrievalDocumentGranularity,
        vector_rank: usize,
        similarity: f64,
        contribution_divisor: usize,
        vector: &[f32],
    ) -> RetrievalResult<ProjectedVectorCandidate> {
        let row = connection
            .query_row(
                "SELECT d.session_id,d.source_sha256,d.start_sequence,d.end_sequence,d.member_count,
                        m.event_id,m.start_char,m.end_char,m.content_sha256,e.sequence,e.role,e.created_at,e.content,e.content_sha256,
                        r.exact_content,r.content_sha256,s.content_sha256
                 FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id
                 JOIN events e ON e.event_id=m.event_id
                 JOIN retrieval_documents r ON r.document_id=d.document_id
                 JOIN source_spans s ON s.event_id=m.event_id AND s.start_char=m.start_char AND s.end_char=m.end_char
                 WHERE d.document_id=?1 AND d.granularity=?2 AND d.member_count=1 AND m.ordinal=0",
                params![document_id, granularity_name(granularity)],
                |row| Ok((
                    row.get::<_, String>(0)?, row.get::<_, String>(1)?, i64_to_usize(row.get(2)?)?,
                    i64_to_usize(row.get(3)?)?, i64_to_usize(row.get(4)?)?, row.get::<_, String>(5)?,
                    i64_to_usize(row.get(6)?)?, i64_to_usize(row.get(7)?)?, row.get::<_, String>(8)?,
                    i64_to_usize(row.get(9)?)?, parse_role(&row.get::<_, String>(10)?)?, row.get::<_, String>(11)?,
                    row.get::<_, String>(12)?, row.get::<_, String>(13)?, row.get::<_, String>(14)?,
                    row.get::<_, String>(15)?, row.get::<_, String>(16)?,
                )),
            )
            .optional()
            .map_err(|error| self.database_error(error))?
            .ok_or_else(|| RetrievalError::CorruptIndex(format!("向量文档 {document_id} 的原文 provenance 不完整")))?;
        let (
            session_id,
            document_hash,
            start_sequence,
            end_sequence,
            member_count,
            event_id,
            start,
            end,
            member_hash,
            sequence,
            role,
            _created_at,
            event_content,
            event_hash,
            exact_content,
            retrieval_hash,
            span_hash,
        ) = row;
        let event = self.get_event_from_connection(connection, &event_id)?;
        if event.session_id != session_id {
            return Err(RetrievalError::CorruptIndex(format!(
                "向量文档 {document_id} 的成员事件会话与文档会话不匹配"
            )));
        }
        let session = self.get_session_from_connection(connection, &session_id)?;
        self.verify_fresh(&session)?;
        verify_event_hash(&event)?;
        let span = SourceSpan {
            event_id,
            start_char: start,
            end_char: end,
        };
        let content = slice_chars(&event_content, &span)?;
        let actual_hash = content_sha256(&content);
        if event.content_sha256 != event_hash
            || sequence != start_sequence
            || sequence != end_sequence
            || member_count != 1
            || document_hash != actual_hash
            || member_hash != actual_hash
            || retrieval_hash != actual_hash
            || span_hash != actual_hash
            || exact_content != content
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "向量文档 {document_id} 与原文不一致"
            )));
        }
        Ok(ProjectedVectorCandidate {
            document_id: document_id.to_owned(),
            source_document_id: source_document_id.to_owned(),
            granularity,
            span: span.clone(),
            role,
            session_id,
            content_sha256: actual_hash,
            content,
            episode_id: self.resolve_episode_id(connection, &span.event_id)?,
            vector_rank,
            similarity,
            contribution_divisor,
            vector: vector.to_vec(),
        })
    }

    fn resolve_episode_id(
        &self,
        connection: &Connection,
        event_id: &str,
    ) -> RetrievalResult<Option<String>> {
        let mut statement = connection
            .prepare(
                "SELECT DISTINCT d.document_id FROM memory_documents d
                 JOIN memory_document_members m ON m.document_id=d.document_id
                 WHERE d.granularity='episode' AND m.event_id=?1 ORDER BY d.document_id",
            )
            .map_err(|error| self.database_error(error))?;
        let ids = statement
            .query_map([event_id], |row| row.get::<_, String>(0))
            .map_err(|error| self.database_error(error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| self.database_error(error))?;
        match ids.as_slice() {
            [] => Ok(None),
            [id] => Ok(Some(id.clone())),
            _ => Err(RetrievalError::CorruptIndex(format!(
                "事件 {event_id} 同时属于多个 episode"
            ))),
        }
    }

    #[allow(clippy::type_complexity)]
    fn claim_snapshots_at(
        &self,
        connection: &Connection,
        visibility_upper: DateTime<Utc>,
    ) -> RetrievalResult<(
        BTreeMap<String, ClaimSnapshot>,
        BTreeMap<String, BTreeSet<String>>,
    )> {
        let original_valid_to = original_claim_valid_to_by_id(connection)?;
        let mut transitions = BTreeMap::<String, Vec<RecallTransition>>::new();
        let mut statement = connection
            .prepare(
                "SELECT claim_id,to_state,reason,related_claim_id,created_at
                 FROM memory_claim_transitions ORDER BY claim_id,ordinal,created_at,transition_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .map_err(|error| self.database_error(error))?;
        for row in rows {
            let (claim, to_state, reason, related_claim_id, created_at) =
                row.map_err(|error| self.database_error(error))?;
            if !matches!(
                to_state.as_str(),
                "active" | "superseded" | "conflicted" | "uncertain"
            ) || !matches!(
                reason.as_str(),
                "created"
                    | "confirmed"
                    | "certainty_upgraded"
                    | "conflicted"
                    | "corrected"
                    | "replaced"
            ) {
                return Err(RetrievalError::CorruptIndex(format!(
                    "声明 {claim} 的迁移状态损坏"
                )));
            }
            transitions
                .entry(claim)
                .or_default()
                .push(RecallTransition {
                    to_state,
                    reason,
                    related_claim_id,
                    created_at: parse_retrieval_time(&created_at, "transition.created_at")?,
                });
        }
        drop(statement);
        let mut claim_metadata = BTreeMap::<String, (String, String)>::new();
        let mut statement = connection
            .prepare(
                "SELECT c.claim_id,c.certainty,c.valid_from
                 FROM memory_claims c ORDER BY c.claim_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|error| self.database_error(error))?;
        for row in rows {
            let (claim, certainty, valid_from) = row.map_err(|error| self.database_error(error))?;
            claim_metadata.insert(claim, (certainty, valid_from));
        }
        drop(statement);
        if claim_metadata.len() != original_valid_to.len()
            || claim_metadata
                .keys()
                .any(|claim| !original_valid_to.contains_key(claim))
        {
            return Err(RetrievalError::CorruptIndex(
                "当前声明与 applied 创建账本不一致".into(),
            ));
        }
        let mut snapshots = BTreeMap::<String, ClaimSnapshot>::new();
        for (claim, (current_certainty, _)) in &claim_metadata {
            let original_end = original_valid_to.get(claim).ok_or_else(|| {
                RetrievalError::CorruptIndex(format!("声明 {claim} 缺少 applied 创建账本"))
            })?;
            let Some(history) = transitions.get(claim) else {
                continue;
            };
            let visible = history
                .iter()
                .filter(|transition| transition.created_at <= visibility_upper)
                .collect::<Vec<_>>();
            if !visible
                .iter()
                .any(|transition| transition.reason == "created")
            {
                continue;
            }
            let state = visible
                .last()
                .expect("visible created transition")
                .to_state
                .clone();
            let certainty = if history.iter().any(|transition| {
                transition.reason == "certainty_upgraded"
                    && transition.created_at > visibility_upper
            }) {
                "uncertain".into()
            } else {
                current_certainty.clone()
            };
            let valid_to = if original_end.is_some() {
                original_end.clone()
            } else {
                visible.iter().find_map(|transition| {
                    matches!(transition.reason.as_str(), "corrected" | "replaced")
                        .then(|| transition.related_claim_id.as_ref())
                        .flatten()
                        .filter(|related| {
                            transitions.get(*related).is_some_and(|history| {
                                history.iter().any(|transition| {
                                    transition.reason == "created"
                                        && transition.created_at <= visibility_upper
                                })
                            })
                        })
                        .and_then(|related| claim_metadata.get(related))
                        .map(|(_, related_from)| related_from.clone())
                })
            };
            let related_claim_ids = visible
                .iter()
                .filter_map(|transition| transition.related_claim_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            snapshots.insert(
                claim.clone(),
                ClaimSnapshot {
                    state,
                    certainty,
                    valid_to,
                    related_claim_ids,
                },
            );
        }
        let mut conflicts = BTreeMap::<String, BTreeSet<String>>::new();
        for (claim, history) in &transitions {
            for transition in history.iter().filter(|transition| {
                transition.created_at <= visibility_upper && transition.reason == "conflicted"
            }) {
                let Some(related) = transition.related_claim_id.as_ref() else {
                    continue;
                };
                if snapshots
                    .get(claim)
                    .is_some_and(|snapshot| snapshot.state == "conflicted")
                    && snapshots
                        .get(related)
                        .is_some_and(|snapshot| snapshot.state == "conflicted")
                {
                    conflicts
                        .entry(claim.clone())
                        .or_default()
                        .insert(related.clone());
                    conflicts
                        .entry(related.clone())
                        .or_default()
                        .insert(claim.clone());
                }
            }
        }
        Ok((snapshots, conflicts))
    }

    fn load_entity_matches(
        &self,
        connection: &Connection,
        control: &ControlState,
        raw_query: &str,
        session_filter: Option<&str>,
        visibility: &RecallVisibility,
    ) -> RetrievalResult<(Vec<EntityMatchTrace>, BTreeSet<String>)> {
        let normalized_query = normalize_match(raw_query);
        let identity_tokens = identity_token_ranges(&normalized_query);
        let mut groups = BTreeMap::<
            String,
            (
                BTreeSet<String>,
                BTreeSet<(u8, String)>,
                Vec<(usize, usize)>,
            ),
        >::new();
        let mut statement = connection
            .prepare(
                "SELECT e.entity_id,e.canonical_name,e.normalized_name
                 FROM memory_entities e ORDER BY e.entity_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|error| self.database_error(error))?;
        for row in rows {
            let (id, text, normalized) = row.map_err(|error| self.database_error(error))?;
            if !self.entity_is_visible(connection, control, &id, session_filter, visibility)? {
                continue;
            }
            let occurrences =
                identity_match_ranges(&normalized_query, &normalized, &identity_tokens);
            if id != "ent_self" && !is_generic_pronoun(&normalized) && !occurrences.is_empty() {
                let group = groups.entry(normalized).or_default();
                group.0.insert(id);
                group.1.insert((3, text));
                group.2.extend(occurrences);
            }
        }
        drop(statement);
        let mut statement = connection
            .prepare(
                "SELECT a.entity_id,a.alias_text,a.normalized_alias,a.alias_kind,a.session_id,
                        a.event_id,a.proof_event_id,a.identity_event_id
                 FROM memory_entity_aliases a JOIN memory_entities e ON e.entity_id=a.entity_id
                 WHERE (?1 IS NULL OR a.session_id=?1)
                 ORDER BY a.alias_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([session_filter], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<String>>(7)?,
                ))
            })
            .map_err(|error| self.database_error(error))?;
        for row in rows {
            let (
                id,
                text,
                normalized,
                basis,
                alias_session,
                event_id,
                proof_event_id,
                identity_event_id,
            ) = row.map_err(|error| self.database_error(error))?;
            if !self.alias_provenance_is_visible(
                connection,
                control,
                &alias_session,
                [
                    event_id.as_deref(),
                    proof_event_id.as_deref(),
                    identity_event_id.as_deref(),
                ],
                session_filter,
                visibility,
            )? || !self.entity_is_visible(
                connection,
                control,
                &id,
                session_filter,
                visibility,
            )? {
                continue;
            }
            let occurrences =
                identity_match_ranges(&normalized_query, &normalized, &identity_tokens);
            if !is_generic_pronoun(&normalized) && !occurrences.is_empty() {
                let priority = if basis == "stable_identifier" { 0 } else { 1 };
                let group = groups.entry(normalized).or_default();
                group.0.insert(id);
                group.1.insert((priority, text));
                group.2.extend(occurrences);
            }
        }
        drop(statement);
        let self_eligible = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM memory_entities WHERE entity_id='ent_self')",
                [],
                |row| row.get::<_, bool>(0),
            )
            .map_err(|error| self.database_error(error))?;
        if self_eligible
            && self.entity_is_visible(
                connection,
                control,
                "ent_self",
                session_filter,
                visibility,
            )?
        {
            for pronoun in ["我", "本人", "i", "me", "my"] {
                let occurrences =
                    identity_match_ranges(&normalized_query, pronoun, &identity_tokens);
                if !occurrences.is_empty() {
                    let group = groups.entry(pronoun.into()).or_default();
                    group.0.insert("ent_self".into());
                    group.1.insert((2, pronoun.into()));
                    group.2.extend(occurrences);
                }
            }
        }
        for (_, _, occurrences) in groups.values_mut() {
            occurrences.sort_unstable();
            occurrences.dedup();
        }
        let occurrence_catalog = groups
            .iter()
            .map(|(surface, (_, _, occurrences))| {
                (
                    surface.clone(),
                    surface.chars().count(),
                    occurrences.clone(),
                )
            })
            .collect::<Vec<_>>();
        let mut selected_entities = BTreeSet::new();
        let mut entity_matches = Vec::new();
        for (normalized, (owners, surfaces, occurrences)) in groups {
            let candidates = owners.into_iter().collect::<Vec<_>>();
            let surface_chars = normalized.chars().count();
            let shadowed = occurrences.iter().all(|&(start, end)| {
                occurrence_catalog
                    .iter()
                    .any(|(other, other_chars, ranges)| {
                        other != &normalized
                            && *other_chars > surface_chars
                            && ranges.iter().any(|&(other_start, other_end)| {
                                other_start <= start
                                    && end <= other_end
                                    && (other_start < start || end < other_end)
                            })
                    })
            });
            let selected = (!shadowed && candidates.len() == 1).then(|| candidates[0].clone());
            let (priority, matched_text) = surfaces.into_iter().next().unwrap_or_default();
            if let Some(id) = &selected {
                selected_entities.insert(id.clone());
            }
            entity_matches.push(EntityMatchTrace {
                matched_text,
                normalized_text: normalized,
                match_basis: match priority {
                    0 => "stable_identifier",
                    1 => "explicit_alias",
                    2 => "ent_self",
                    _ => "canonical_name",
                }
                .into(),
                candidate_entity_ids: candidates,
                selected_entity_id: selected.clone(),
                reason: if shadowed {
                    "shadowed_by_longer_match"
                } else if selected.is_some() {
                    "unique"
                } else {
                    "ambiguous"
                }
                .into(),
            });
        }
        Ok((entity_matches, selected_entities))
    }

    fn entity_is_visible(
        &self,
        connection: &Connection,
        control: &ControlState,
        entity_id: &str,
        session_filter: Option<&str>,
        visibility: &RecallVisibility,
    ) -> RetrievalResult<bool> {
        let created_event_id = connection
            .query_row(
                "SELECT created_event_id FROM memory_entities WHERE entity_id=?1",
                [entity_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| self.database_error(error))?
            .ok_or_else(|| RetrievalError::CorruptIndex(format!("实体 {entity_id} 缺失")))?;
        let event = self.authoritative_indexed_event(connection, &created_event_id)?;
        if !control.allows_session(&event.session_id)
            || !control.allows_event(&event.session_id, &event.id)
        {
            return Ok(false);
        }
        Ok(visibility.event_created_at_is_visible(&event.created_at)?
            && self.entity_has_visible_resolved_mention(
                connection,
                control,
                entity_id,
                session_filter,
                visibility,
            )?)
    }

    fn entity_has_visible_resolved_mention(
        &self,
        connection: &Connection,
        control: &ControlState,
        entity_id: &str,
        session_filter: Option<&str>,
        visibility: &RecallVisibility,
    ) -> RetrievalResult<bool> {
        let mut statement = connection
            .prepare(
                "SELECT event_id,session_id FROM memory_entity_mentions
                 WHERE entity_id=?1 AND entity_status='resolved'
                   AND (?2 IS NULL OR session_id=?2)
                 ORDER BY mention_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map(params![entity_id, session_filter], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| self.database_error(error))?;
        let mut visible = false;
        for row in rows {
            let (event_id, mention_session) = row.map_err(|error| self.database_error(error))?;
            let event = self.authoritative_indexed_event(connection, &event_id)?;
            if event.session_id != mention_session {
                return Err(RetrievalError::CorruptIndex(format!(
                    "实体 {entity_id} mention 会话绑定损坏"
                )));
            }
            if session_filter.is_some_and(|scope| scope != event.session_id) {
                return Err(RetrievalError::CorruptIndex(format!(
                    "实体 {entity_id} mention 超出会话范围"
                )));
            }
            if !control.allows_session(&event.session_id)
                || !control.allows_event(&event.session_id, &event.id)
            {
                continue;
            }
            visible |= visibility.event_created_at_is_visible(&event.created_at)?;
        }
        Ok(visible)
    }

    fn alias_provenance_is_visible(
        &self,
        connection: &Connection,
        control: &ControlState,
        alias_session: &str,
        event_ids: [Option<&str>; 3],
        session_filter: Option<&str>,
        visibility: &RecallVisibility,
    ) -> RetrievalResult<bool> {
        let mut visible = true;
        for event_id in event_ids.into_iter().flatten() {
            let event = self.authoritative_indexed_event(connection, event_id)?;
            if event.session_id != alias_session {
                return Err(RetrievalError::CorruptIndex(
                    "entity alias provenance 会话绑定损坏".into(),
                ));
            }
            if session_filter.is_some_and(|scope| scope != event.session_id) {
                return Err(RetrievalError::CorruptIndex(
                    "entity alias provenance 超出会话范围".into(),
                ));
            }
            if !control.allows_session(&event.session_id)
                || !control.allows_event(&event.session_id, &event.id)
            {
                return Ok(false);
            }
            visible &= visibility.event_created_at_is_visible(&event.created_at)?;
        }
        Ok(visible)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_state_sidecar(
        &self,
        connection: &Connection,
        control: &ControlState,
        raw_query: &str,
        current_user_event_id: &str,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
        query_kind: QueryKind,
        candidate_limit: usize,
        query_origin: &RecallQueryOrigin,
        episode_enabled: bool,
        visibility: &RecallVisibility,
    ) -> RetrievalResult<StateSidecar> {
        let entity_started = Instant::now();
        validate_full_derived_integrity(connection)?;
        let original_valid_to = original_claim_valid_to_by_id(connection)?;
        let normalized_query = normalize_match(raw_query);
        let (entity_matches, selected_entities) =
            self.load_entity_matches(connection, control, raw_query, session_filter, visibility)?;
        let entity_ms = elapsed_ms(entity_started);
        let state_started = Instant::now();
        let moment = match query_origin {
            RecallQueryOrigin::IndexedEvent => {
                let current = self.get_event_from_connection(connection, current_user_event_id)?;
                if current.role != EventRole::User
                    || session_filter.is_some_and(|session_id| current.session_id != session_id)
                {
                    return Err(RetrievalError::CorruptIndex(
                        "当前查询事件角色或会话范围不匹配".into(),
                    ));
                }
                let current_session =
                    self.get_session_from_connection(connection, &current.session_id)?;
                self.verify_fresh(&current_session)?;
                verify_event_hash(&current)?;
                RecallMoment::Indexed(Box::new(current))
            }
            RecallQueryOrigin::Synthetic { reference_time } => RecallMoment::Synthetic(
                parse_retrieval_time(reference_time, "synthetic reference_time")?,
            ),
        };
        let current_reference = match &moment {
            RecallMoment::Indexed(current) => {
                parse_retrieval_time(&current.created_at, "current query event")?
            }
            RecallMoment::Synthetic(reference) => *reference,
        };
        let (explicit_reference, invalid_date) = explicit_query_date(&normalized_query);
        let visibility_upper = explicit_reference
            .map(|value| value.min(current_reference))
            .unwrap_or(current_reference);
        let reference = visibility_upper;
        let historical = has_historical_cue(&normalized_query);
        let mut warnings = Vec::new();
        if invalid_date {
            warnings.push("查询包含无效日期，已使用时间提示模式".into());
        }
        let terms = query_terms(raw_query)
            .into_iter()
            .map(|term| normalize_match(&term))
            .filter(|term| !term.is_empty())
            .collect::<BTreeSet<_>>();
        let mut transitions = BTreeMap::<String, Vec<RecallTransition>>::new();
        let mut statement = connection
            .prepare(
                "SELECT claim_id,to_state,reason,related_claim_id,created_at
                 FROM memory_claim_transitions ORDER BY claim_id,ordinal,created_at,transition_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .map_err(|error| self.database_error(error))?;
        for row in rows {
            let (claim, to_state, reason, related_claim_id, created_at) =
                row.map_err(|error| self.database_error(error))?;
            if !matches!(
                to_state.as_str(),
                "active" | "superseded" | "conflicted" | "uncertain"
            ) || !matches!(
                reason.as_str(),
                "created"
                    | "confirmed"
                    | "certainty_upgraded"
                    | "conflicted"
                    | "corrected"
                    | "replaced"
            ) {
                return Err(RetrievalError::CorruptIndex(format!(
                    "声明 {claim} 的迁移状态损坏"
                )));
            }
            transitions
                .entry(claim)
                .or_default()
                .push(RecallTransition {
                    to_state,
                    reason,
                    related_claim_id,
                    created_at: parse_retrieval_time(&created_at, "transition.created_at")?,
                });
        }
        drop(statement);
        let mut claim_metadata = BTreeMap::<String, (String, String)>::new();
        let mut statement = connection
            .prepare(
                "SELECT c.claim_id,c.certainty,c.valid_from
                 FROM memory_claims c ORDER BY c.claim_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|error| self.database_error(error))?;
        for row in rows {
            let (claim, certainty, valid_from) = row.map_err(|error| self.database_error(error))?;
            claim_metadata.insert(claim, (certainty, valid_from));
        }
        drop(statement);
        if claim_metadata.len() != original_valid_to.len()
            || claim_metadata
                .keys()
                .any(|claim| !original_valid_to.contains_key(claim))
        {
            return Err(RetrievalError::CorruptIndex(
                "当前声明与 applied 创建账本不一致".into(),
            ));
        }
        let mut snapshots = BTreeMap::<String, ClaimSnapshot>::new();
        for (claim, (current_certainty, _)) in &claim_metadata {
            let original_end = original_valid_to.get(claim).ok_or_else(|| {
                RetrievalError::CorruptIndex(format!("声明 {claim} 缺少 applied 创建账本"))
            })?;
            let Some(history) = transitions.get(claim) else {
                continue;
            };
            let visible = history
                .iter()
                .filter(|transition| transition.created_at <= visibility_upper)
                .collect::<Vec<_>>();
            if !visible
                .iter()
                .any(|transition| transition.reason == "created")
            {
                continue;
            }
            let state = visible.last().unwrap().to_state.clone();
            let certainty = if history.iter().any(|transition| {
                transition.reason == "certainty_upgraded"
                    && transition.created_at > visibility_upper
            }) {
                "uncertain".into()
            } else {
                current_certainty.clone()
            };
            let valid_to = if original_end.is_some() {
                original_end.clone()
            } else {
                visible.iter().find_map(|transition| {
                    matches!(transition.reason.as_str(), "corrected" | "replaced")
                        .then(|| transition.related_claim_id.as_ref())
                        .flatten()
                        .filter(|related| {
                            transitions.get(*related).is_some_and(|history| {
                                history.iter().any(|transition| {
                                    transition.reason == "created"
                                        && transition.created_at <= visibility_upper
                                })
                            })
                        })
                        .and_then(|related| claim_metadata.get(related))
                        .map(|(_, related_from)| related_from.clone())
                })
            };
            let related_claim_ids = visible
                .iter()
                .filter_map(|transition| transition.related_claim_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            snapshots.insert(
                claim.clone(),
                ClaimSnapshot {
                    state,
                    certainty,
                    valid_to,
                    related_claim_ids,
                },
            );
        }
        let mut conflicts = BTreeMap::<String, BTreeSet<String>>::new();
        for (claim, history) in &transitions {
            for transition in history.iter().filter(|transition| {
                transition.created_at <= visibility_upper && transition.reason == "conflicted"
            }) {
                let Some(related) = transition.related_claim_id.as_ref() else {
                    continue;
                };
                if snapshots
                    .get(claim)
                    .is_some_and(|snapshot| snapshot.state == "conflicted")
                    && snapshots
                        .get(related)
                        .is_some_and(|snapshot| snapshot.state == "conflicted")
                {
                    conflicts
                        .entry(claim.clone())
                        .or_default()
                        .insert(related.clone());
                    conflicts
                        .entry(related.clone())
                        .or_default()
                        .insert(claim.clone());
                }
            }
        }
        let mut statement = connection
            .prepare(
                "SELECT c.claim_id,c.subject_entity_id,c.object_entity_id,c.predicate_key,
                        c.normalized_relation,c.normalized_object,c.object_kind,c.state,c.certainty,
                        c.asserted_at,c.event_time,c.valid_from,c.valid_to,c.reference_time
                 FROM memory_claims c ORDER BY c.claim_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, Option<String>>(10)?,
                    row.get::<_, String>(11)?,
                    row.get::<_, Option<String>>(12)?,
                    row.get::<_, String>(13)?,
                ))
            })
            .map_err(|error| self.database_error(error))?;
        let mut candidates = Vec::new();
        for row in rows {
            let (
                claim_id,
                subject,
                object,
                predicate,
                relation,
                object_text,
                object_kind,
                _current_state,
                _current_certainty,
                asserted,
                event_time,
                valid_from,
                _current_valid_to,
                reference_time,
            ) = row.map_err(|error| self.database_error(error))?;
            let Some(snapshot) = snapshots.get(&claim_id) else {
                continue;
            };
            let state = snapshot.state.clone();
            let certainty = snapshot.certainty.clone();
            let valid_to = snapshot.valid_to.clone();
            let mut reason = "eligible".to_owned();
            let lexical = claim_overlap(&terms, &predicate, &relation, &object_text);
            let from = parse_retrieval_time(&valid_from, "claim.valid_from")?;
            let asserted_time = parse_retrieval_time(&asserted, "claim.asserted_at")?;
            let claim_reference = parse_retrieval_time(&reference_time, "claim.reference_time")?;
            let claim_event_time = event_time
                .as_deref()
                .map(|value| parse_retrieval_time(value, "claim.event_time"))
                .transpose()?;
            let to = valid_to
                .as_deref()
                .map(|value| parse_retrieval_time(value, "claim.valid_to"))
                .transpose()?;
            let object_resolved = if object_kind == "entity" {
                object
                    .as_deref()
                    .map(|id| {
                        self.entity_is_visible(connection, control, id, session_filter, visibility)
                    })
                    .transpose()?
                    .unwrap_or(false)
            } else {
                true
            };
            if !selected_entities.contains(&subject) {
                reason = "subject_not_selected".into();
            } else if !object_resolved {
                reason = "pending_object".into();
            } else if asserted_time > visibility_upper
                || claim_reference > visibility_upper
                || from > visibility_upper
                || claim_event_time.is_some_and(|value| value > visibility_upper)
            {
                reason = "not_yet_visible".into();
            } else if query_kind != QueryKind::TemporalState && lexical == 0 {
                reason = "no_claim_overlap".into();
            } else if !historical
                && (!(from <= reference && to.is_none_or(|value| value >= reference))
                    || explicit_reference.is_none()
                        && !matches!(state.as_str(), "active" | "conflicted" | "uncertain"))
            {
                reason = "not_applicable".into();
            }
            let related_ids = snapshot.related_claim_ids.clone();
            let mut trace = StateSelectionTrace {
                claim_id: claim_id.clone(),
                subject_entity_id: subject,
                object_entity_id: object,
                predicate_key: predicate,
                state: state.clone(),
                certainty,
                asserted_at: asserted,
                event_time,
                valid_from,
                valid_to,
                reference_time,
                related_claim_ids: related_ids.clone(),
                reason: reason.clone(),
                ..Default::default()
            };
            let mut content = String::new();
            let mut hash = String::new();
            let mut episode = None;
            if reason == "eligible" {
                if let Some(evidence) = self.load_claim_evidence(
                    connection,
                    &claim_id,
                    query_kind,
                    &moment,
                    visibility_upper,
                    recent_event_ids,
                    session_filter,
                )? {
                    trace.evidence_id = Some(evidence.evidence_id);
                    trace.evidence_span = Some(evidence.span.clone());
                    trace.evidence_role = Some(evidence.role);
                    content = evidence.content;
                    hash = evidence.content_sha256;
                    if episode_enabled {
                        episode = self.resolve_episode_id(connection, &evidence.span.event_id)?;
                    }
                } else {
                    trace.reason = "no_eligible_user_evidence".into();
                }
            }
            candidates.push((
                lexical,
                from,
                StateEvidenceCandidate {
                    trace,
                    content,
                    content_sha256: hash,
                    episode_id: episode,
                    conflict_group: if state == "conflicted" {
                        complete_conflict_group(&conflicts, &claim_id)
                    } else {
                        vec![claim_id]
                    },
                },
            ));
        }
        candidates.sort_by(|left, right| {
            (right.0 > 0)
                .cmp(&(left.0 > 0))
                .then_with(|| right.0.cmp(&left.0))
                .then_with(|| {
                    state_priority(&left.2.trace.state).cmp(&state_priority(&right.2.trace.state))
                })
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| left.2.trace.claim_id.cmp(&right.2.trace.claim_id))
        });
        let mut eligible_groups = BTreeSet::new();
        for (index, (_, _, candidate)) in candidates.iter_mut().enumerate() {
            candidate.trace.rank = index + 1;
            if candidate.trace.reason == "eligible" {
                let group_key = candidate.conflict_group.join("\0");
                if !eligible_groups.contains(&group_key) && eligible_groups.len() >= candidate_limit
                {
                    candidate.trace.reason = "candidate_limit".into();
                } else {
                    eligible_groups.insert(group_key);
                }
            }
        }
        Ok(StateSidecar {
            entity_matches,
            candidates: candidates.into_iter().map(|(_, _, value)| value).collect(),
            warnings,
            entity_ms,
            state_ms: elapsed_ms(state_started),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_claim_evidence(
        &self,
        connection: &Connection,
        claim_id: &str,
        query_kind: QueryKind,
        moment: &RecallMoment,
        visibility_upper: DateTime<Utc>,
        recent_event_ids: &[String],
        session_filter: Option<&str>,
    ) -> RetrievalResult<Option<ClaimEvidenceSelection>> {
        let temporal_first = query_kind == QueryKind::TemporalState;
        let mut statement = connection
            .prepare(
                "SELECT v.evidence_id,v.event_id,v.start_char,v.end_char,v.content_sha256,e.role,
                        v.created_at,e.session_id,v.sequence
             FROM memory_claim_evidence v JOIN events e ON e.event_id=v.event_id
             WHERE v.claim_id=?1 AND v.role='user' AND (?2 IS NULL OR e.session_id=?2)
             ORDER BY CASE v.kind
                WHEN 'temporal' THEN ?3 WHEN 'correction' THEN ?4
                WHEN 'user_confirmation' THEN ?5 ELSE ?6 END,
                v.sequence DESC,v.start_char,v.end_char,v.evidence_id",
            )
            .map_err(|error| self.database_error(error))?;
        let priorities = if temporal_first {
            (0, 1, 2, 3)
        } else {
            (3, 0, 1, 2)
        };
        let rows = statement
            .query_map(
                params![
                    claim_id,
                    session_filter,
                    priorities.0,
                    priorities.1,
                    priorities.2,
                    priorities.3
                ],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, i64>(8)?,
                    ))
                },
            )
            .map_err(|error| self.database_error(error))?;
        for row in rows {
            let (id, event_id, start, end, stored_hash, role, created_at, event_session, sequence) =
                row.map_err(|error| self.database_error(error))?;
            let created = parse_retrieval_time(&created_at, "claim_evidence.created_at")?;
            if created > visibility_upper
                || matches!(moment, RecallMoment::Indexed(current)
                    if event_session == current.session_id
                    && i64_to_usize(sequence).map_err(|error| self.database_error(error))?
                        >= current.sequence)
                || matches!(moment, RecallMoment::Indexed(current) if event_id == current.id)
                || recent_event_ids.iter().any(|recent| recent == &event_id)
            {
                continue;
            }
            let event = self.get_event_from_connection(connection, &event_id)?;
            if parse_retrieval_time(&event.created_at, "event.created_at")? > visibility_upper {
                continue;
            }
            let session = self.get_session_from_connection(connection, &event.session_id)?;
            self.verify_fresh(&session)?;
            verify_event_hash(&event)?;
            if role != "user" || event.role != EventRole::User {
                continue;
            }
            let span = SourceSpan {
                event_id,
                start_char: i64_to_usize(start).map_err(|error| self.database_error(error))?,
                end_char: i64_to_usize(end).map_err(|error| self.database_error(error))?,
            };
            let content = slice_chars(&event.content, &span)?;
            let hash = content_sha256(&content);
            if hash != stored_hash {
                return Err(RetrievalError::CorruptIndex(format!(
                    "声明证据 {id} 与原文不一致"
                )));
            }
            return Ok(Some(ClaimEvidenceSelection {
                evidence_id: id,
                span,
                role: EventRole::User,
                content,
                content_sha256: hash,
            }));
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn select_fused_candidates(
        &self,
        connection: &Connection,
        mut candidates: Vec<FusedRawCandidate>,
        mut bm25: RecallResult,
        retrieval_config: RetrievalConfig,
        bm25_ms: u64,
        vector_ms: u64,
        vector_candidate_count: usize,
        capped_candidates: Vec<FusedRawCandidate>,
        excluded_vectors: Vec<FusionCandidateTrace>,
        mut sidecar: StateSidecar,
        mut graph: GraphSidecar,
        graph_candidate_limit: usize,
        channels: RecallChannels,
    ) -> RetrievalResult<RecallResult> {
        let mut processing_order = (0..candidates.len()).collect::<Vec<_>>();
        processing_order.sort_by_key(|&index| !candidates[index].protected_exact);
        let mut used_events = HashSet::new();
        let mut used_hashes = HashSet::new();
        let mut used_episodes = HashSet::new();
        let mut eligible = Vec::new();
        for index in processing_order {
            let candidate = &mut candidates[index];
            if !used_events.insert(candidate.span.event_id.clone()) {
                candidate.reason = "duplicate_event".into();
            } else if !used_hashes.insert(candidate.content_sha256.clone()) {
                candidate.reason = "duplicate_content".into();
            } else if candidate
                .episode_id
                .as_ref()
                .is_some_and(|episode| !used_episodes.insert(episode.clone()))
            {
                candidate.reason = "duplicate_episode".into();
            } else {
                candidate.reason = "eligible".into();
                eligible.push(index);
            }
        }
        eligible.sort_by(|&left, &right| {
            (!candidates[left].protected_exact)
                .cmp(&(!candidates[right].protected_exact))
                .then_with(|| {
                    candidates[right]
                        .rrf_score
                        .total_cmp(&candidates[left].rrf_score)
                })
                .then_with(|| {
                    candidates[left]
                        .document_id
                        .cmp(&candidates[right].document_id)
                })
        });
        let max_rrf = eligible
            .iter()
            .map(|&index| candidates[index].rrf_score)
            .fold(0.0_f64, f64::max)
            .max(f64::EPSILON);
        let mut selected = Vec::<usize>::new();
        let mut selected_chars = 0usize;
        if let Some(position) = eligible
            .iter()
            .position(|&index| candidates[index].protected_exact)
        {
            let index = eligible.remove(position);
            selected_chars += candidates[index].content.chars().count();
            candidates[index].selected = true;
            candidates[index].reason = "protected_exact".into();
            selected.push(index);
        }
        let mut state_only = Vec::new();
        let mut state_slots = 0usize;
        let target_state_slots =
            final_hard_limit(retrieval_config.max_selected).map_or(usize::MAX, |max_selected| {
                max_selected
                    .saturating_mul(usize::from(
                        bm25.trace.budget_allocation.exact_or_state_percent,
                    ))
                    .div_ceil(100)
                    .max(1)
                    .min(max_selected.saturating_sub(selected.len()))
            });
        let mut selected_events = selected
            .iter()
            .map(|&index| candidates[index].span.event_id.clone())
            .collect::<HashSet<_>>();
        let mut selected_hashes = selected
            .iter()
            .map(|&index| candidates[index].content_sha256.clone())
            .collect::<HashSet<_>>();
        let mut selected_episodes = selected
            .iter()
            .filter_map(|&index| candidates[index].episode_id.clone())
            .collect::<HashSet<_>>();
        let mut handled_groups = BTreeSet::new();
        for index in 0..sidecar.candidates.len() {
            if sidecar.candidates[index].trace.reason != "eligible" {
                continue;
            }
            let group = sidecar.candidates[index].conflict_group.clone();
            let group_key = group.join("\0");
            if !handled_groups.insert(group_key) {
                continue;
            }
            let member_indexes = group
                .iter()
                .filter_map(|claim| {
                    sidecar
                        .candidates
                        .iter()
                        .position(|candidate| candidate.trace.claim_id == *claim)
                })
                .collect::<Vec<_>>();
            if member_indexes.len() != group.len()
                || member_indexes.iter().any(|&member| {
                    sidecar.candidates[member].trace.reason != "eligible"
                        || sidecar.candidates[member].trace.evidence_span.is_none()
                })
            {
                for member in member_indexes {
                    sidecar.candidates[member].trace.reason = "incomplete_conflict_group".into();
                }
                continue;
            }
            let mut group_events = selected_events.clone();
            let mut group_hashes = selected_hashes.clone();
            let mut group_episodes = selected_episodes.clone();
            let mut actions = Vec::new();
            let mut additional = 0usize;
            let mut chars = 0usize;
            let mut duplicate_reason = None;
            for &member in &member_indexes {
                let item = &sidecar.candidates[member];
                let span = item.trace.evidence_span.as_ref().unwrap();
                let exact = candidates.iter().position(|candidate| {
                    candidate.span == *span
                        && (candidate.reason == "eligible"
                            || candidate.protected_exact && candidate.selected)
                });
                let already_selected_exact = exact.is_some_and(|candidate_index| {
                    candidates[candidate_index].protected_exact
                        && candidates[candidate_index].selected
                });
                if !already_selected_exact {
                    duplicate_reason = if group_events.contains(&span.event_id) {
                        Some("duplicate_state_event")
                    } else if group_hashes.contains(&item.content_sha256) {
                        Some("duplicate_state_content")
                    } else if item
                        .episode_id
                        .as_ref()
                        .is_some_and(|episode| group_episodes.contains(episode))
                    {
                        Some("duplicate_state_episode")
                    } else {
                        None
                    };
                    if duplicate_reason.is_some() {
                        break;
                    }
                    group_events.insert(span.event_id.clone());
                    group_hashes.insert(item.content_sha256.clone());
                    if let Some(episode) = &item.episode_id {
                        group_episodes.insert(episode.clone());
                    }
                    additional += 1;
                    chars += item.content.chars().count();
                }
                actions.push((member, exact, already_selected_exact));
            }
            if let Some(reason) = duplicate_reason {
                for member in member_indexes {
                    sidecar.candidates[member].trace.selected = false;
                    sidecar.candidates[member].trace.reason = reason.into();
                }
                continue;
            }
            if state_slots + additional > target_state_slots
                || final_hard_limit(retrieval_config.max_selected)
                    .is_some_and(|limit| selected.len() + state_only.len() + additional > limit)
            {
                for member in member_indexes {
                    sidecar.candidates[member].trace.reason = "state_slot_limit".into();
                }
                continue;
            }
            if final_hard_limit(retrieval_config.evidence_char_budget)
                .is_some_and(|limit| selected_chars + chars > limit)
            {
                for member in member_indexes {
                    sidecar.candidates[member].trace.reason = "evidence_budget".into();
                }
                continue;
            }
            for (member, exact, already_selected_exact) in actions {
                let item = &mut sidecar.candidates[member];
                let span = item.trace.evidence_span.as_ref().unwrap();
                if let Some(candidate_index) = exact {
                    candidates[candidate_index].selected = true;
                    candidates[candidate_index].reason = "selected_state".into();
                    if !selected.contains(&candidate_index) {
                        selected.push(candidate_index);
                        if let Some(position) =
                            eligible.iter().position(|value| *value == candidate_index)
                        {
                            eligible.remove(position);
                        }
                    }
                    item.trace.selected = true;
                    item.trace.reason = format!("selected_state:{}", item.trace.claim_id);
                } else {
                    state_only.push(member);
                    item.trace.selected = true;
                    item.trace.reason = format!("selected_state:{}", item.trace.claim_id);
                }
                if !already_selected_exact {
                    selected_events.insert(span.event_id.clone());
                    selected_hashes.insert(item.content_sha256.clone());
                    if let Some(episode) = &item.episode_id {
                        selected_episodes.insert(episode.clone());
                    }
                    selected_chars += item.content.chars().count();
                    state_slots += 1;
                }
            }
        }
        let graph_slot_budget = if final_hard_limit(retrieval_config.max_selected).is_none() {
            usize::MAX
        } else if bm25.trace.budget_allocation.graph_percent == 0 {
            0
        } else {
            (retrieval_config.max_selected
                * usize::from(bm25.trace.budget_allocation.graph_percent))
            .div_ceil(100)
            .max(1)
        };
        let graph_char_budget = final_hard_limit(retrieval_config.evidence_char_budget).map_or(
            usize::MAX,
            |char_budget| {
                (char_budget * usize::from(bm25.trace.budget_allocation.graph_percent))
                    .div_ceil(100)
            },
        );
        let mut graph_only = Vec::new();
        let mut graph_slots = 0usize;
        let mut graph_chars = 0usize;
        let mut graph_candidates_selected = 0usize;
        for graph_index in 0..graph.candidates.len() {
            let item = &graph.candidates[graph_index];
            let path = &mut graph.paths[item.path_index];
            let raw = &item.raw;
            let raw_chars = raw.content.chars().count();
            if graph_slots >= graph_slot_budget {
                path.reason = "graph_slot_limit".into();
                continue;
            }
            if graph_chars + raw_chars > graph_char_budget {
                path.reason = "graph_evidence_budget".into();
                continue;
            }
            if sidecar.candidates.iter().any(|state| {
                state.trace.selected && state.trace.evidence_span.as_ref() == Some(&raw.span)
            }) {
                path.reason = "duplicate_state".into();
                continue;
            }
            if let Some(candidate_index) = candidates.iter().position(|candidate| {
                candidate.span == raw.span
                    && candidate.selected
                    && candidate.reason != "selected_state"
                    && candidate.reason != "selected_graph"
            }) {
                if graph_candidates_selected >= graph_candidate_limit {
                    path.reason = "candidate_limit".into();
                    continue;
                }
                path.selected = true;
                path.reason = "selected_coalesced".into();
                candidates[candidate_index].selected = true;
                graph_slots += 1;
                graph_chars += raw_chars;
                graph_candidates_selected += 1;
                continue;
            }
            let duplicate_reason = if selected_events.contains(&raw.span.event_id) {
                Some("duplicate_event")
            } else if selected_hashes.contains(&raw.content_sha256) {
                Some("duplicate_content")
            } else if raw
                .episode_id
                .as_ref()
                .is_some_and(|episode| selected_episodes.contains(episode))
            {
                Some("duplicate_episode")
            } else {
                None
            };
            if let Some(reason) = duplicate_reason {
                path.reason = reason.into();
                continue;
            }
            if final_hard_limit(retrieval_config.max_selected)
                .is_some_and(|limit| selected.len() + state_only.len() + graph_only.len() >= limit)
            {
                path.reason = "selection_limit".into();
                continue;
            }
            if final_hard_limit(retrieval_config.evidence_char_budget)
                .is_some_and(|limit| selected_chars + raw_chars > limit)
            {
                path.reason = "evidence_budget".into();
                continue;
            }
            if graph_candidates_selected >= graph_candidate_limit {
                path.reason = "candidate_limit".into();
                continue;
            }
            if let Some(candidate_index) = candidates
                .iter()
                .position(|candidate| candidate.span == raw.span && candidate.reason == "eligible")
            {
                candidates[candidate_index].selected = true;
                candidates[candidate_index].reason = "selected_graph".into();
                selected.push(candidate_index);
                if let Some(position) = eligible.iter().position(|value| *value == candidate_index)
                {
                    eligible.remove(position);
                }
            } else {
                graph_only.push(graph_index);
            }
            selected_events.insert(raw.span.event_id.clone());
            selected_hashes.insert(raw.content_sha256.clone());
            if let Some(episode) = &raw.episode_id {
                selected_episodes.insert(episode.clone());
            }
            selected_chars += raw_chars;
            graph_slots += 1;
            graph_chars += raw_chars;
            graph_candidates_selected += 1;
            path.selected = true;
            path.reason = "selected_graph".into();
        }
        eligible.retain(|&index| {
            let candidate = &mut candidates[index];
            if selected_events.contains(&candidate.span.event_id) {
                candidate.reason = "duplicate_state_event".into();
                false
            } else if selected_hashes.contains(&candidate.content_sha256) {
                candidate.reason = "duplicate_state_content".into();
                false
            } else if candidate
                .episode_id
                .as_ref()
                .is_some_and(|episode| selected_episodes.contains(episode))
            {
                candidate.reason = "duplicate_state_episode".into();
                false
            } else {
                true
            }
        });
        while final_hard_limit(retrieval_config.max_selected)
            .is_none_or(|limit| selected.len() + state_only.len() + graph_only.len() < limit)
            && !eligible.is_empty()
        {
            let best_position = eligible
                .iter()
                .enumerate()
                .max_by(|left_entry, right_entry| {
                    let left = *left_entry.1;
                    let right = *right_entry.1;
                    let left_score = mmr_score(&candidates[left], &selected, &candidates, max_rrf);
                    let right_score =
                        mmr_score(&candidates[right], &selected, &candidates, max_rrf);
                    left_score
                        .total_cmp(&right_score)
                        .then_with(|| {
                            candidates[left]
                                .rrf_score
                                .total_cmp(&candidates[right].rrf_score)
                        })
                        .then_with(|| {
                            candidates[right]
                                .document_id
                                .cmp(&candidates[left].document_id)
                        })
                })
                .map(|(position, _)| position)
                .expect("non-empty eligible pool has a best candidate");
            let index = eligible.remove(best_position);
            let chars = candidates[index].content.chars().count();
            if final_hard_limit(retrieval_config.evidence_char_budget)
                .is_some_and(|limit| selected_chars + chars > limit)
            {
                candidates[index].reason = "evidence_budget".into();
                continue;
            }
            selected_chars += chars;
            candidates[index].selected = true;
            candidates[index].reason = "selected_mmr".into();
            selected.push(index);
        }
        for index in eligible {
            candidates[index].reason = "selection_limit".into();
        }
        let mut evidence = Vec::new();
        let mut selected_evidence = Vec::new();
        let mut expansion_events = selected_events;
        let mut expansion_hashes = selected_hashes;
        for &index in &selected {
            let candidate = &candidates[index];
            if candidate.reason == "selected_state" && !candidate.protected_exact {
                continue;
            }
            let rank = candidate.pre_cap_rank;
            let evidence_reason = if candidate.reason == "selected_state" {
                sidecar
                    .candidates
                    .iter()
                    .find(|state| state.trace.evidence_span.as_ref() == Some(&candidate.span))
                    .map(|state| state.trace.reason.clone())
                    .unwrap_or_else(|| candidate.reason.clone())
            } else {
                candidate.reason.clone()
            };
            let selected_item = SelectedEvidence {
                span: candidate.span.clone(),
                content_sha256: candidate.content_sha256.clone(),
                role: candidate.role,
                kind: EvidenceKind::Core,
                originating_candidate_rank: Some(rank),
                reason: evidence_reason,
            };
            expansion_events.insert(candidate.span.event_id.clone());
            expansion_hashes.insert(candidate.content_sha256.clone());
            selected_evidence.push(selected_item.clone());
            evidence.push(RecalledEvidence {
                selected: selected_item,
                content: candidate.content.clone(),
            });
        }
        for graph_index in graph_only {
            let item = &graph.candidates[graph_index];
            let raw = &item.raw;
            let selected_item = SelectedEvidence {
                span: raw.span.clone(),
                content_sha256: raw.content_sha256.clone(),
                role: raw.role,
                kind: EvidenceKind::Core,
                originating_candidate_rank: None,
                reason: "selected_graph".into(),
            };
            expansion_events.insert(raw.span.event_id.clone());
            expansion_hashes.insert(raw.content_sha256.clone());
            selected_evidence.push(selected_item.clone());
            evidence.push(RecalledEvidence {
                selected: selected_item,
                content: raw.content.clone(),
            });
        }
        bm25.trace.status = "ok".into();
        bm25.trace.config = retrieval_config;
        bm25.trace.selected_evidence = selected_evidence;
        bm25.trace.entity_matches = sidecar.entity_matches;
        bm25.trace.warnings.append(&mut sidecar.warnings);
        bm25.trace.fusion_candidates = candidates
            .iter()
            .chain(capped_candidates.iter())
            .map(|candidate| FusionCandidateTrace {
                fused_rank: candidate.pre_cap_rank,
                document_id: candidate.document_id.clone(),
                span: candidate.span.clone(),
                session_id: candidate.session_id.clone(),
                granularity: candidate.granularity,
                source_document_ids: candidate.source_document_ids.clone(),
                episode_id: candidate.episode_id.clone(),
                bm25_rank: candidate.bm25_rank,
                bm25_score: candidate.bm25_score,
                vector_rank: candidate.vector_rank,
                vector_score: candidate.vector_score,
                rrf_score: candidate.rrf_score,
                protected_exact: candidate.protected_exact,
                selected: candidate.selected,
                reason: candidate.reason.clone(),
            })
            .collect();
        bm25.trace
            .fusion_candidates
            .sort_by_key(|candidate| candidate.fused_rank);
        bm25.trace
            .fusion_candidates
            .extend(excluded_vectors.iter().cloned());
        bm25.trace.channels = vec![
            channel_trace(
                RetrievalChannel::Bm25,
                if channels.bm25 { "ok" } else { "disabled" },
                bm25.trace.candidates.len(),
                bm25_ms,
                None,
            ),
            channel_trace(
                RetrievalChannel::Vector,
                if channels.vector { "ok" } else { "disabled" },
                vector_candidate_count,
                vector_ms,
                None,
            ),
            channel_trace(
                RetrievalChannel::Entity,
                if !channels.entity {
                    "disabled"
                } else if bm25.trace.entity_matches.is_empty() {
                    "empty"
                } else {
                    "ok"
                },
                bm25.trace.entity_matches.len(),
                sidecar.entity_ms,
                None,
            ),
            channel_trace(
                RetrievalChannel::State,
                if !channels.state {
                    "disabled"
                } else if sidecar.candidates.is_empty() {
                    "empty"
                } else {
                    "ok"
                },
                sidecar.candidates.len(),
                sidecar.state_ms,
                None,
            ),
            channel_trace(
                RetrievalChannel::Episode,
                if !channels.episode {
                    "disabled"
                } else if graph.aggregate_source_count == 0 {
                    "empty"
                } else {
                    "ok"
                },
                graph.aggregate_source_count,
                0,
                None,
            ),
            channel_trace(
                RetrievalChannel::Graph,
                if !channels.graph {
                    "disabled"
                } else if graph.candidate_count == 0 {
                    "empty"
                } else {
                    "ok"
                },
                graph.candidate_count,
                graph.elapsed_ms,
                None,
            ),
        ];
        if let Some(warning) = graph.warning {
            bm25.trace.warnings.push(warning);
        }
        bm25.trace.graph_paths = graph.paths;
        let mut emitted_events = evidence
            .iter()
            .map(|item| item.selected.span.event_id.clone())
            .collect::<HashSet<_>>();
        let mut emitted_hashes = evidence
            .iter()
            .map(|item| item.selected.content_sha256.clone())
            .collect::<HashSet<_>>();
        let mut emitted_episodes = HashSet::new();
        if channels.episode {
            for item in &evidence {
                if let Some(episode) =
                    self.resolve_episode_id(connection, &item.selected.span.event_id)?
                {
                    emitted_episodes.insert(episode);
                }
            }
        }
        let mut emitted_core_slots = evidence
            .iter()
            .filter(|item| item.selected.kind == EvidenceKind::Core)
            .count();
        let mut emitted_core_chars = evidence
            .iter()
            .filter(|item| item.selected.kind == EvidenceKind::Core)
            .map(|item| item.content.chars().count())
            .sum::<usize>();
        let mut final_groups = sidecar
            .candidates
            .iter()
            .filter(|candidate| candidate.trace.selected)
            .map(|candidate| (candidate.trace.rank, candidate.conflict_group.clone()))
            .collect::<Vec<_>>();
        final_groups.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let mut final_handled = BTreeSet::new();
        for (_, group) in final_groups {
            if !final_handled.insert(group.clone()) {
                continue;
            }
            let member_indexes = group
                .iter()
                .filter_map(|claim| {
                    sidecar
                        .candidates
                        .iter()
                        .position(|candidate| candidate.trace.claim_id == *claim)
                })
                .collect::<Vec<_>>();
            if member_indexes.len() != group.len()
                || member_indexes
                    .iter()
                    .any(|&member| !sidecar.candidates[member].trace.selected)
            {
                continue;
            }
            let pending = member_indexes
                .iter()
                .copied()
                .filter(|&member| {
                    let span = sidecar.candidates[member]
                        .trace
                        .evidence_span
                        .as_ref()
                        .unwrap();
                    !evidence.iter().any(|item| item.selected.span == *span)
                })
                .collect::<Vec<_>>();
            let mut group_events = emitted_events.clone();
            let mut group_hashes = emitted_hashes.clone();
            let mut group_episodes = emitted_episodes.clone();
            let mut exclusion = None;
            for &member in &pending {
                let candidate = &sidecar.candidates[member];
                let span = candidate.trace.evidence_span.as_ref().unwrap();
                exclusion = if !group_events.insert(span.event_id.clone()) {
                    Some("final_duplicate_state_event")
                } else if !group_hashes.insert(candidate.content_sha256.clone()) {
                    Some("final_duplicate_state_content")
                } else if candidate
                    .episode_id
                    .as_ref()
                    .is_some_and(|episode| !group_episodes.insert(episode.clone()))
                {
                    Some("final_duplicate_state_episode")
                } else {
                    None
                };
                if exclusion.is_some() {
                    break;
                }
            }
            let pending_chars = pending
                .iter()
                .map(|&member| sidecar.candidates[member].content.chars().count())
                .sum::<usize>();
            if exclusion.is_none()
                && final_hard_limit(bm25.trace.config.max_selected)
                    .is_some_and(|limit| emitted_core_slots + pending.len() > limit)
            {
                exclusion = Some("final_state_slot_limit");
            }
            if exclusion.is_none()
                && final_hard_limit(bm25.trace.config.evidence_char_budget)
                    .is_some_and(|limit| emitted_core_chars + pending_chars > limit)
            {
                exclusion = Some("final_state_evidence_budget");
            }
            if let Some(reason) = exclusion {
                for member in member_indexes {
                    let state = &mut sidecar.candidates[member];
                    state.trace.selected = false;
                    state.trace.reason = reason.into();
                    if let Some(span) = &state.trace.evidence_span
                        && let Some(fused) = bm25
                            .trace
                            .fusion_candidates
                            .iter_mut()
                            .find(|fused| fused.span == *span && !fused.protected_exact)
                    {
                        fused.selected = false;
                        fused.reason = reason.into();
                    }
                }
                continue;
            }
            for &member in &pending {
                let candidate = &sidecar.candidates[member];
                let span = candidate.trace.evidence_span.clone().unwrap();
                let fused_rank = candidates
                    .iter()
                    .find(|fused| fused.span == span && fused.reason == "selected_state")
                    .map(|fused| fused.pre_cap_rank);
                let selected = SelectedEvidence {
                    span,
                    content_sha256: candidate.content_sha256.clone(),
                    role: EventRole::User,
                    kind: EvidenceKind::Core,
                    originating_candidate_rank: fused_rank,
                    reason: candidate.trace.reason.clone(),
                };
                bm25.trace.selected_evidence.push(selected.clone());
                expansion_events.insert(selected.span.event_id.clone());
                expansion_hashes.insert(selected.content_sha256.clone());
                evidence.push(RecalledEvidence {
                    selected,
                    content: candidate.content.clone(),
                });
            }
            emitted_events = group_events;
            emitted_hashes = group_hashes;
            emitted_episodes = group_episodes;
            emitted_core_slots += pending.len();
            emitted_core_chars += pending_chars;
        }
        bm25.trace.state_selections = sidecar
            .candidates
            .into_iter()
            .map(|candidate| candidate.trace)
            .collect();
        bm25.evidence = evidence;
        Ok(bm25)
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
        session_filter: Option<&str>,
        control: &ControlState,
        visibility: &RecallVisibility,
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
                if !control.allows_event(&adjacent.session_id, &adjacent.id) {
                    continue;
                }
                if !visibility.event_created_at_is_visible(&adjacent.created_at)? {
                    continue;
                }
                if session_filter.is_some_and(|scope| adjacent.session_id != scope) {
                    continue;
                }
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
                if final_hard_limit(trace.config.expansion_char_budget)
                    .is_some_and(|limit| budget + chars > limit)
                {
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
        let _guard = self.acquire_root_read()?;
        let state = self.replay_control_state_under_guard()?;
        let mut database = self.open_connection()?;
        let transaction = database
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|error| self.database_error(error))?;
        let connection: &Connection = &transaction;
        let event = self.get_event_from_connection(connection, answer_event_id)?;
        let mut answer_active = state.allows_event(&event.session_id, &event.id);
        let session = self.get_session_from_connection(connection, &event.session_id)?;
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
        let retrieval_trace_json = connection
            .query_row(
                "SELECT trace_json FROM retrieval_runs WHERE answer_event_id=?1",
                [answer_event_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| self.database_error(e))?;
        let source = self.read_source(&self.root.join(&session.source_file))?;
        let event_turn_id = event
            .turn_id
            .as_deref()
            .ok_or_else(|| RetrievalError::CorruptIndex("回答事件缺少轮次标识".into()))?;
        if event.session_id != source.session.id
            || event.role != EventRole::Assistant
            || event.id
                != event_id(
                    &source.session.id,
                    Some(event_turn_id),
                    EventRole::Assistant,
                )
        {
            return Err(RetrievalError::CorruptIndex(
                "回答事件不是来源会话的规范 Assistant 事件".into(),
            ));
        }
        let turn = source
            .session
            .turns
            .iter()
            .find(|turn| turn.id == event_turn_id)
            .ok_or_else(|| {
                RetrievalError::CorruptIndex(format!(
                    "回答 {answer_event_id} 在原始会话中缺少对应轮次"
                ))
            })?;
        let structural_user_id = event_id(&source.session.id, Some(&turn.id), EventRole::User);
        answer_active &= state.allows_event(&source.session.id, &structural_user_id);
        answer.retrieval_trace = if let Some(retrieval_trace_json) = retrieval_trace_json {
            let canonical_retrieval_trace = serde_json::to_string(&turn.context_trace.retrieval)
                .map_err(|error| {
                    RetrievalError::CorruptIndex(format!(
                        "回答 {answer_event_id} 的来源 retrieval trace 无法规范序列化：{error}"
                    ))
                })?;
            if retrieval_trace_json != canonical_retrieval_trace {
                return Err(RetrievalError::CorruptIndex(format!(
                    "回答 {answer_event_id} 的 retrieval run 与来源 trace 原始字节不一致"
                )));
            }
            let mut trace =
                serde_json::from_str::<RetrievalTrace>(&retrieval_trace_json).map_err(|error| {
                    RetrievalError::CorruptIndex(format!(
                        "回答 {answer_event_id} 的 retrieval run JSON 无效：{error}"
                    ))
                })?;
            trace.normalize_usage();
            trace
        } else if turn.context_trace.retrieval == RetrievalTrace::default() {
            RetrievalTrace::default()
        } else {
            return Err(RetrievalError::CorruptIndex(format!(
                "回答 {answer_event_id} 缺少 retrieval run"
            )));
        };
        let source_events = derive_events(&source.session);
        let source_event_by_id = source_events
            .iter()
            .map(|event| (event.id.clone(), event))
            .collect::<HashMap<_, _>>();
        if source_event_by_id.get(answer_event_id).copied() != Some(&event) {
            return Err(RetrievalError::CorruptIndex(
                "回答事件索引与来源会话不匹配".into(),
            ));
        }
        let authoritative = derive_context(
            &source.session,
            turn,
            &source_event_by_id,
            source.legacy,
            &source.path,
        )
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        if answer.turn_id != turn.id
            || answer.context_sha256 != authoritative.context_sha256
            || answer.estimated_upper_tokens != turn.context_trace.estimated_upper_tokens
            || answer.exact_input_tokens != turn.context_trace.exact_input_tokens
            || answer.input_budget != turn.context_trace.input_budget
            || answer.decision != turn.context_trace.decision
            || answer.provenance_quality != authoritative.provenance_quality
            || answer.request != authoritative.request
            || answer.identity_instruction != authoritative.identity_instruction
            || answer.retrieval_trace != turn.context_trace.retrieval
        {
            return Err(RetrievalError::CorruptIndex(
                "回答上下文索引元数据与来源 trace 不匹配".into(),
            ));
        }
        answer_active &=
            self.retrieval_trace_is_active(connection, &state, &answer.retrieval_trace)?;
        if turn.context_trace.untrusted_history_wrapped {
            let expected_identity = wrapped_history_identity(
                &source.session.ai_name,
                &turn.context_trace.retrieval.selected_evidence,
            );
            if turn.context_trace.identity_instruction.as_deref()
                != Some(expected_identity.as_str())
                || answer.identity_instruction.as_deref() != Some(expected_identity.as_str())
            {
                return Err(RetrievalError::CorruptIndex(
                    "不可信历史身份指令与规范元数据不匹配".into(),
                ));
            }
        }
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
        let mut wrapped_history = WrappedHistoryCursor::new(
            turn.context_trace.untrusted_history_wrapped,
            &turn.context_trace.retrieval.selected_evidence,
        )
        .map_err(|message| RetrievalError::CorruptIndex(message.into()))?;
        let session_system_event_id = event_id(&session.id, None, EventRole::System);
        let session_system_end = source.session.system_prompt.chars().count();
        let session_system_hash = content_sha256(&source.session.system_prompt);
        for row in rows {
            let (ordinal, role, span, expected_hash, content) =
                row.map_err(|source| self.database_error(source))?;
            let authoritative_item = authoritative
                .items
                .get(answer.items.len())
                .ok_or_else(|| RetrievalError::CorruptIndex("回答上下文索引包含多余片段".into()))?;
            if ordinal != answer.items.len()
                || role != authoritative_item.role
                || span != authoritative_item.span
                || expected_hash != authoritative_item.content_sha256
            {
                return Err(RetrievalError::CorruptIndex(
                    "回答上下文索引片段与来源 trace 不匹配".into(),
                ));
            }
            let source_event = self.authoritative_indexed_event(connection, &span.event_id)?;
            let is_session_system_prompt = !source.session.system_prompt.is_empty()
                && ordinal == 0
                && span.event_id == session_system_event_id
                && span.start_char == 0
                && span.end_char == session_system_end
                && expected_hash == session_system_hash;
            let wrapped_source_role = wrapped_history
                .consume(
                    &ContextItemTrace {
                        role,
                        span: span.clone(),
                        content_sha256: expected_hash.clone(),
                    },
                    is_session_system_prompt,
                )
                .map_err(|message| RetrievalError::CorruptIndex(message.into()))?;
            let role_matches = match wrapped_source_role {
                Some(selected_role) => selected_role == source_event.role,
                None => role == source_event.role,
            };
            if !role_matches {
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
            answer_active &= state.allows_event(&source_event.session_id, &source_event.id);
            let generated_before_item = wrapped_source_role.is_some();
            if !inserted_generated && (role != EventRole::System || generated_before_item) {
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
            if !inserted_generated && role == EventRole::System && !generated_before_item {
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
        wrapped_history
            .finish()
            .map_err(|message| RetrievalError::CorruptIndex(message.into()))?;
        if answer.items.len() != authoritative.items.len() {
            return Err(RetrievalError::CorruptIndex(
                "回答上下文索引缺少来源 trace 片段".into(),
            ));
        }
        if context_sha256(&messages) != answer.context_sha256 {
            return Err(RetrievalError::CorruptIndex(format!(
                "回答 {answer_event_id} 的整体上下文哈希不匹配"
            )));
        }
        drop(statement);
        self.require_unchanged_control_state(&state)?;
        if !answer_active {
            return Err(RetrievalError::ExcludedEvent(answer_event_id.to_owned()));
        }
        transaction
            .commit()
            .map_err(|error| self.database_error(error))?;
        Ok(answer)
    }

    fn authoritative_source_span_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        local_events: &HashMap<String, &StoredEvent>,
        span: &SourceSpan,
        expected_hash: &str,
        expected_role: EventRole,
    ) -> RetrievalResult<Option<bool>> {
        if span.event_id.is_empty() {
            return Err(RetrievalError::CorruptIndex(
                "回答 provenance 缺少权威事件".into(),
            ));
        }
        let indexed = connection
            .query_row(
                "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                        content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error
                 FROM events WHERE event_id=?1",
                [&span.event_id],
                map_event,
            )
            .optional()
            .map_err(|error| self.database_error(error))?;
        let has_indexed = indexed.is_some();
        let local = local_events.get(&span.event_id).copied();
        let authoritative = if let Some(indexed) = indexed {
            let authoritative = self.authoritative_indexed_event(connection, &span.event_id)?;
            if indexed != authoritative || local.is_some_and(|event| event != &authoritative) {
                return Err(RetrievalError::CorruptIndex(
                    "回答 provenance 的 indexed/raw 事件不一致".into(),
                ));
            }
            Some(authoritative)
        } else if let Some(local) = local {
            Some(local.clone())
        } else {
            let mut matches = self
                .load_all_sources()?
                .into_iter()
                .flat_map(|source| derive_events(&source.session))
                .filter(|event| event.id == span.event_id);
            let event = matches.next();
            if matches.next().is_some() {
                return Err(RetrievalError::CorruptIndex(
                    "回答 provenance 事件未唯一绑定权威来源".into(),
                ));
            }
            event
        };
        let Some(authoritative) = authoritative else {
            return Ok(None);
        };
        verify_event_hash(&authoritative)?;
        let selected = slice_chars(&authoritative.content, span)?;
        if authoritative.role != expected_role || content_sha256(&selected) != expected_hash {
            return Err(RetrievalError::CorruptIndex(
                "回答 provenance 与权威事件的角色或片段哈希不一致".into(),
            ));
        }
        if has_indexed {
            let stored_hash = connection
                .query_row(
                    "SELECT content_sha256 FROM source_spans
                     WHERE event_id=?1 AND start_char=?2 AND end_char=?3",
                    params![
                        span.event_id,
                        usize_to_i64(span.start_char).map_err(|error| self.database_error(error))?,
                        usize_to_i64(span.end_char).map_err(|error| self.database_error(error))?,
                    ],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(|error| self.database_error(error))?;
            if stored_hash.is_some_and(|hash| hash != expected_hash) {
                return Err(RetrievalError::CorruptIndex(
                    "回答 provenance 的持久化片段哈希不一致".into(),
                ));
            }
        }
        Ok(Some(control.allows_event(
            &authoritative.session_id,
            &authoritative.id,
        )))
    }

    fn trace_span_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        expected_session: Option<&str>,
        span: &SourceSpan,
        expected_hash: Option<&str>,
        expected_role: Option<EventRole>,
    ) -> RetrievalResult<bool> {
        if span.event_id.is_empty() {
            return Ok(true);
        }
        let event = self.authoritative_indexed_event(connection, &span.event_id)?;
        if expected_session.is_some_and(|session| session != event.session_id) {
            return Err(RetrievalError::CorruptIndex(
                "trace 来源事件属于错误会话".into(),
            ));
        }
        let content = slice_chars(&event.content, span)?;
        let actual_hash = content_sha256(&content);
        let stored_hash = connection
            .query_row(
                "SELECT content_sha256 FROM source_spans
                 WHERE event_id=?1 AND start_char=?2 AND end_char=?3",
                params![
                    span.event_id,
                    usize_to_i64(span.start_char).map_err(|error| self.database_error(error))?,
                    usize_to_i64(span.end_char).map_err(|error| self.database_error(error))?
                ],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| self.database_error(error))?;
        if stored_hash.as_deref() != Some(actual_hash.as_str())
            || expected_hash.is_some_and(|hash| hash != actual_hash)
            || expected_role.is_some_and(|role| role != event.role)
        {
            return Err(RetrievalError::CorruptIndex(
                "trace 来源片段与权威原文不一致".into(),
            ));
        }
        Ok(control.allows_event(&event.session_id, &event.id)
            && expected_session.is_none_or(|session| control.allows_session(session)))
    }

    fn validate_entity_trace_provenance(
        &self,
        connection: &Connection,
        control: &ControlState,
        trace: &RetrievalTrace,
        current: &StoredEvent,
    ) -> RetrievalResult<()> {
        if trace
            .entity_matches
            .iter()
            .any(|matched| matched == &EntityMatchTrace::default())
        {
            return Err(RetrievalError::CorruptIndex(
                "entity trace 混入 default-shaped provenance".into(),
            ));
        }
        if trace.entity_matches.windows(2).any(|pair| {
            pair[0].normalized_text.is_empty() || pair[0].normalized_text >= pair[1].normalized_text
        }) || trace
            .entity_matches
            .last()
            .is_some_and(|matched| matched.normalized_text.is_empty())
        {
            return Err(RetrievalError::CorruptIndex(
                "entity trace 的 canonical order 或唯一性损坏".into(),
            ));
        }

        let indexed_visibility = RecallVisibility {
            cutoff: None,
            allow_aggregate_documents: true,
        };
        let (global_matches, _) = self.load_entity_matches(
            connection,
            control,
            &current.content,
            None,
            &indexed_visibility,
        )?;
        let (session_matches, _) = self.load_entity_matches(
            connection,
            control,
            &current.content,
            Some(&current.session_id),
            &indexed_visibility,
        )?;
        let all_stored_entries_match_global_scope = trace
            .entity_matches
            .iter()
            .all(|stored| global_matches.iter().any(|current| current == stored));
        let all_stored_entries_match_session_scope = trace
            .entity_matches
            .iter()
            .all(|stored| session_matches.iter().any(|current| current == stored));
        if !all_stored_entries_match_global_scope && !all_stored_entries_match_session_scope {
            return Err(RetrievalError::CorruptIndex(
                "entity trace 未整体匹配同一 current safe projection scope".into(),
            ));
        }

        for matched in &trace.entity_matches {
            let Some(entity_id) = matched.selected_entity_id.as_ref() else {
                continue;
            };
            if matched.reason != "unique"
                || matched.candidate_entity_ids.as_slice() != std::slice::from_ref(entity_id)
            {
                return Err(RetrievalError::CorruptIndex(
                    "selected entity trace 未绑定唯一 resolved owner".into(),
                ));
            }
            let resolved_entity = connection
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM memory_entities
                         WHERE entity_id=?1 AND disambiguation='resolved')",
                    [entity_id],
                    |row| row.get::<_, bool>(0),
                )
                .map_err(|error| self.database_error(error))?;
            if !resolved_entity
                || !self.entity_has_visible_resolved_mention(
                    connection,
                    control,
                    entity_id,
                    None,
                    &indexed_visibility,
                )?
            {
                return Err(RetrievalError::CorruptIndex(
                    "selected entity trace 不属于 resolved projection".into(),
                ));
            }
        }
        let expected_seeds = entity_graph_seeds(&trace.entity_matches)
            .into_iter()
            .map(|seed| {
                let GraphRecallSeed {
                    source_id,
                    rank,
                    score,
                    ..
                } = seed;
                (source_id, (rank, score))
            })
            .collect::<BTreeMap<_, _>>();
        for path in trace
            .graph_paths
            .iter()
            .filter(|path| path.seed_channel == RetrievalChannel::Entity)
        {
            let (rank, score) = expected_seeds.get(&path.seed_source_id).ok_or_else(|| {
                RetrievalError::CorruptIndex(
                    "graph entity seed 未绑定 selected entity match".into(),
                )
            })?;
            let expected_node = trace_graph_node_id("entity", &path.seed_source_id);
            if !path.seed_document_id.is_empty()
                || path.seed_rank != *rank
                || path.seed_score.to_bits() != score.to_bits()
                || path.seed_node_id != expected_node
                || path.node_ids.first() != Some(&expected_node)
            {
                return Err(RetrievalError::CorruptIndex(
                    "graph entity seed 与 canonical entity producer 不一致".into(),
                ));
            }
        }
        Ok(())
    }

    fn state_selection_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        selection: &StateSelectionTrace,
        snapshots: &BTreeMap<String, ClaimSnapshot>,
        conflicts: &BTreeMap<String, BTreeSet<String>>,
    ) -> RetrievalResult<bool> {
        if selection.claim_id.is_empty() || selection.rank == 0 || selection.reason.is_empty() {
            return Err(RetrievalError::CorruptIndex(
                "state trace 缺少 canonical claim selection".into(),
            ));
        }
        let claim = connection
            .query_row(
                "SELECT subject_entity_id,object_entity_id,predicate_key,asserted_at,event_time,
                        valid_from,reference_time
                 FROM memory_claims WHERE claim_id=?1",
                [&selection.claim_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| self.database_error(error))?
            .ok_or_else(|| RetrievalError::CorruptIndex("state trace claim 不存在".into()))?;
        let snapshot = snapshots.get(&selection.claim_id).ok_or_else(|| {
            RetrievalError::CorruptIndex("state trace claim 在查询时间不可见".into())
        })?;
        if selection.subject_entity_id != claim.0
            || selection.object_entity_id != claim.1
            || selection.predicate_key != claim.2
            || selection.asserted_at != claim.3
            || selection.event_time != claim.4
            || selection.valid_from != claim.5
            || selection.reference_time != claim.6
            || selection.state != snapshot.state
            || selection.certainty != snapshot.certainty
            || selection.valid_to != snapshot.valid_to
            || selection.related_claim_ids != snapshot.related_claim_ids
        {
            return Err(RetrievalError::CorruptIndex(
                "state trace 与 authoritative as-of claim 不一致".into(),
            ));
        }
        let conflict_group = if snapshot.state == "conflicted" {
            complete_conflict_group(conflicts, &selection.claim_id)
        } else {
            vec![selection.claim_id.clone()]
        };
        if !conflict_group
            .iter()
            .any(|claim_id| claim_id == &selection.claim_id)
        {
            return Err(RetrievalError::CorruptIndex(
                "state trace claim 不属于 authoritative conflict group".into(),
            ));
        }
        if selection.selected
            && selection.reason != format!("selected_state:{}", selection.claim_id)
        {
            return Err(RetrievalError::CorruptIndex(
                "state trace selected reason 未绑定 claim".into(),
            ));
        }
        match (
            selection.evidence_id.as_deref(),
            selection.evidence_span.as_ref(),
            selection.evidence_role,
        ) {
            (None, None, None) => {
                if selection.selected {
                    return Err(RetrievalError::CorruptIndex(
                        "selected state trace 缺少 evidence".into(),
                    ));
                }
                Ok(true)
            }
            (Some(evidence_id), Some(span), Some(evidence_role)) => {
                let evidence = connection
                    .query_row(
                        "SELECT claim_id,session_id,batch_key,event_id,sequence,role,
                                start_char,end_char,content_sha256
                         FROM memory_claim_evidence WHERE evidence_id=?1",
                        [evidence_id],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, String>(3)?,
                                i64_to_usize(row.get(4)?)?,
                                parse_role(&row.get::<_, String>(5)?)?,
                                i64_to_usize(row.get(6)?)?,
                                i64_to_usize(row.get(7)?)?,
                                row.get::<_, String>(8)?,
                            ))
                        },
                    )
                    .optional()
                    .map_err(|error| self.database_error(error))?
                    .ok_or_else(|| {
                        RetrievalError::CorruptIndex("state trace evidence_id 不存在".into())
                    })?;
                let event = self.authoritative_indexed_event(connection, &evidence.3)?;
                let expected_span = SourceSpan {
                    event_id: evidence.3.clone(),
                    start_char: evidence.6,
                    end_char: evidence.7,
                };
                let content = slice_chars(&event.content, &expected_span)?;
                if evidence.0 != selection.claim_id
                    || evidence.1 != event.session_id
                    || evidence.2.is_empty()
                    || evidence.4 != event.sequence
                    || evidence.5 != event.role
                    || evidence_role != EventRole::User
                    || evidence.5 != evidence_role
                    || expected_span != *span
                    || content_sha256(&content) != evidence.8
                {
                    return Err(RetrievalError::CorruptIndex(
                        "state trace evidence 与 authoritative claim/raw span 不一致".into(),
                    ));
                }
                self.trace_span_is_active(
                    connection,
                    control,
                    Some(&evidence.1),
                    span,
                    Some(&evidence.8),
                    Some(evidence_role),
                )
            }
            _ => Err(RetrievalError::CorruptIndex(
                "state trace evidence tuple 不完整".into(),
            )),
        }
    }

    fn graph_path_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        path: &crate::model::GraphPathTrace,
    ) -> RetrievalResult<bool> {
        if path == &crate::model::GraphPathTrace::default() {
            return Ok(true);
        }
        let target_granularity = path.target_granularity.ok_or_else(|| {
            RetrievalError::CorruptIndex("graph trace 缺少 target granularity".into())
        })?;
        if path.seed_node_id.is_empty()
            || path.seed_source_id.is_empty()
            || path.target_document_id.is_empty()
            || path.target_session_id.is_empty()
            || path.reason.is_empty()
            || path.seed_rank == 0
            || path.target_rank == 0
            || !path.score.is_finite()
            || path.score < 0.0
            || !path.path_quality.is_finite()
            || path.path_quality <= 0.0
            || !path.seed_score.is_finite()
            || !path.seed_mass.is_finite()
            || path.seed_mass <= 0.0
            || path.edge_ids.len() > 2
            || path.edge_types.len() != path.edge_ids.len()
            || path.node_ids.len() != path.edge_ids.len() + 1
            || path.node_ids.iter().any(String::is_empty)
            || path.edge_ids.iter().any(String::is_empty)
            || path.edge_types.iter().any(String::is_empty)
            || path
                .node_ids
                .iter()
                .any(|node_id| !trace_graph_id_is_valid(node_id, "gnode_"))
            || path
                .edge_ids
                .iter()
                .any(|edge_id| !trace_graph_id_is_valid(edge_id, "gedge_"))
            || path.node_ids.iter().collect::<HashSet<_>>().len() != path.node_ids.len()
            || path.edge_ids.iter().collect::<HashSet<_>>().len() != path.edge_ids.len()
        {
            return Err(RetrievalError::CorruptIndex(
                "graph trace 的自包含 path provenance 不规范".into(),
            ));
        }
        let expected_seed_node = match path.seed_channel {
            RetrievalChannel::Bm25 | RetrievalChannel::Vector => {
                if path.seed_document_id.is_empty() || path.seed_document_id != path.seed_source_id
                {
                    return Err(RetrievalError::CorruptIndex(
                        "graph document seed 字段不一致".into(),
                    ));
                }
                trace_graph_node_id("document", &path.seed_source_id)
            }
            RetrievalChannel::Entity => {
                if !path.seed_document_id.is_empty() {
                    return Err(RetrievalError::CorruptIndex(
                        "graph entity seed 不应包含 document_id".into(),
                    ));
                }
                trace_graph_node_id("entity", &path.seed_source_id)
            }
            _ => {
                return Err(RetrievalError::CorruptIndex(
                    "graph trace 包含未生产的 seed channel".into(),
                ));
            }
        };
        let expected_target_node = trace_graph_node_id("document", &path.target_document_id);
        if path.seed_node_id != expected_seed_node
            || path.node_ids.first() != Some(&expected_seed_node)
            || path.node_ids.last() != Some(&expected_target_node)
        {
            return Err(RetrievalError::CorruptIndex(
                "graph trace 的 seed/target node 绑定不一致".into(),
            ));
        }
        for index in 0..path.edge_ids.len() {
            if !trace_graph_edge_id_matches(
                &path.edge_types[index],
                &path.node_ids[index],
                &path.node_ids[index + 1],
                &path.edge_ids[index],
            ) {
                return Err(RetrievalError::CorruptIndex(
                    "graph trace 的 edge ID/type/path 绑定不一致".into(),
                ));
            }
        }
        let (target_active, target_session, stored_granularity) =
            self.document_metadata_is_active(connection, control, &path.target_document_id)?;
        if target_session != path.target_session_id || stored_granularity != target_granularity {
            return Err(RetrievalError::CorruptIndex(
                "graph target document/session/granularity 不一致".into(),
            ));
        }
        let mut active = target_active;
        if !path.seed_document_id.is_empty() {
            let (seed_active, _, _) =
                self.document_metadata_is_active(connection, control, &path.seed_document_id)?;
            active &= seed_active;
        }
        match (&path.span, path.role) {
            (Some(span), Some(role)) => {
                if path.content_sha256.is_empty() {
                    return Err(RetrievalError::CorruptIndex(
                        "graph target span 缺少 content hash".into(),
                    ));
                }
                let span_active = self.trace_span_is_active(
                    connection,
                    control,
                    Some(&path.target_session_id),
                    span,
                    Some(&path.content_sha256),
                    Some(role),
                )?;
                let document_active = self.document_span_binding_is_active(
                    connection,
                    control,
                    &path.target_document_id,
                    &path.target_session_id,
                    span,
                    Some(target_granularity),
                    false,
                )?;
                active &= span_active && document_active;
            }
            (None, None) => {
                if !path.content_sha256.is_empty()
                    || path.selected
                    || path.reason == "no_eligible_raw_member"
                        && !matches!(
                            target_granularity,
                            RetrievalDocumentGranularity::Episode
                                | RetrievalDocumentGranularity::Session
                        )
                    || !matches!(
                        path.reason.as_str(),
                        "seed_document" | "no_eligible_raw_member"
                    )
                {
                    return Err(RetrievalError::CorruptIndex(
                        "graph no-span path 字段不规范".into(),
                    ));
                }
            }
            _ => {
                return Err(RetrievalError::CorruptIndex(
                    "graph target span/role tuple 不完整".into(),
                ));
            }
        }
        Ok(active)
    }

    fn retrieval_trace_is_active(
        &self,
        connection: &Connection,
        control: &ControlState,
        trace: &RetrievalTrace,
    ) -> RetrievalResult<bool> {
        let mut active = true;
        if !trace.current_query_event_id.is_empty() {
            if control.event_is_excluded(&trace.current_query_event_id) {
                active = false;
            }
            if let Some(event) = connection.query_row(
                "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                        content_sha256, reply_to_event_id, token_count, turn_status, done_reason, error
                 FROM events WHERE event_id=?1",
                [&trace.current_query_event_id], map_event,
            ).optional().map_err(|error| self.database_error(error))? {
                verify_event_hash(&event)?;
                active &= control.allows_event(&event.session_id, &event.id);
            }
        }
        for candidate in &trace.candidates {
            if candidate.span.event_id.is_empty() || candidate.session_id.is_empty() {
                return Err(RetrievalError::CorruptIndex(
                    "BM25 trace 缺少权威来源".into(),
                ));
            }
            let span_active = self.trace_span_is_active(
                connection,
                control,
                Some(&candidate.session_id),
                &candidate.span,
                Some(&candidate.content_sha256),
                Some(candidate.role),
            )?;
            let document_active =
                self.ranked_candidate_document_is_active(connection, control, candidate)?;
            active &= span_active && document_active;
        }
        for candidate in &trace.fusion_candidates {
            if candidate == &FusionCandidateTrace::default() {
                continue;
            }
            if candidate.span.event_id.is_empty()
                || candidate.session_id.is_empty()
                || candidate.document_id.is_empty()
                || candidate.source_document_ids.is_empty()
                || candidate.source_document_ids.iter().any(String::is_empty)
                || candidate
                    .source_document_ids
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || candidate.bm25_rank.is_some() != candidate.bm25_score.is_some()
                || candidate.vector_rank.is_some() != candidate.vector_score.is_some()
                || candidate.bm25_rank.is_some_and(|rank| rank == 0)
                || candidate.vector_rank.is_some_and(|rank| rank == 0)
                || candidate.bm25_score.is_some_and(|score| !score.is_finite())
                || candidate
                    .vector_score
                    .is_some_and(|score| !score.is_finite())
                || !candidate.rrf_score.is_finite()
                || candidate.rrf_score < 0.0
                || candidate.bm25_rank.is_none() && candidate.vector_rank.is_none()
                || candidate.protected_exact && candidate.bm25_rank.is_none()
                || candidate.reason.is_empty()
                || candidate.episode_id.as_ref().is_some_and(String::is_empty)
                || candidate.fused_rank == 0
                    && (candidate.bm25_rank.is_some()
                        || candidate.vector_rank.is_none()
                        || candidate.rrf_score != 0.0
                        || candidate.protected_exact
                        || candidate.selected
                        || !matches!(
                            candidate.reason.as_str(),
                            "current_message" | "recent_context" | "system_message"
                        ))
                || candidate.fused_rank > 0 && candidate.rrf_score <= 0.0
            {
                return Err(RetrievalError::CorruptIndex(
                    "fusion trace 的 canonical producer 字段不规范".into(),
                ));
            }
            if candidate.bm25_rank.is_some()
                && !candidate
                    .source_document_ids
                    .iter()
                    .any(|source| source == &candidate.document_id)
            {
                return Err(RetrievalError::CorruptIndex(
                    "fusion BM25 source 集合缺少 primary document".into(),
                ));
            }
            if let Some(bm25_rank) = candidate.bm25_rank {
                let mut matches = trace
                    .candidates
                    .iter()
                    .filter(|ranked| ranked.raw_rank == bm25_rank);
                let ranked = matches.next().ok_or_else(|| {
                    RetrievalError::CorruptIndex(
                        "fusion BM25 rank 缺少 canonical ranked candidate".into(),
                    )
                })?;
                if matches.next().is_some()
                    || ranked.document_id != candidate.document_id
                    || ranked.span != candidate.span
                    || ranked.session_id != candidate.session_id
                    || ranked.granularity != candidate.granularity
                    || candidate.bm25_score != Some(ranked.bm25_score)
                {
                    return Err(RetrievalError::CorruptIndex(
                        "fusion BM25 candidate 与 ranked raw provenance 不一致".into(),
                    ));
                }
            }
            if candidate.vector_rank.is_none()
                && candidate.source_document_ids.as_slice()
                    != std::slice::from_ref(&candidate.document_id)
            {
                return Err(RetrievalError::CorruptIndex(
                    "fusion BM25-only source 集合不精确".into(),
                ));
            }
            let span_active = self.trace_span_is_active(
                connection,
                control,
                Some(&candidate.session_id),
                &candidate.span,
                None,
                None,
            )?;
            let primary_active = self.document_span_binding_is_active(
                connection,
                control,
                &candidate.document_id,
                &candidate.session_id,
                &candidate.span,
                Some(candidate.granularity),
                true,
            )?;
            let mut sources_active = true;
            for document_id in &candidate.source_document_ids {
                sources_active &= self.document_span_binding_is_active(
                    connection,
                    control,
                    document_id,
                    &candidate.session_id,
                    &candidate.span,
                    None,
                    false,
                )?;
            }
            let expected_episode = self.resolve_episode_id(connection, &candidate.span.event_id)?;
            if candidate.episode_id != expected_episode {
                return Err(RetrievalError::CorruptIndex(
                    "fusion episode_id 与 raw event membership 不一致".into(),
                ));
            }
            let episode_active = if let Some(episode_id) = &candidate.episode_id {
                self.episode_reference_is_active(
                    connection,
                    control,
                    episode_id,
                    &candidate.session_id,
                    &candidate.span.event_id,
                )?
            } else {
                true
            };
            active &= span_active && primary_active && sources_active && episode_active;
        }
        let has_entity_matches = trace
            .entity_matches
            .iter()
            .any(|matched| matched != &EntityMatchTrace::default());
        let has_entity_seed = trace
            .graph_paths
            .iter()
            .any(|path| path.seed_channel == RetrievalChannel::Entity);
        let has_graph_paths = trace
            .graph_paths
            .iter()
            .any(|path| path != &crate::model::GraphPathTrace::default());
        let has_state_selections = trace
            .state_selections
            .iter()
            .any(|selection| selection != &StateSelectionTrace::default());
        if has_entity_matches || has_graph_paths || has_state_selections {
            self.require_current_control_projection(connection, control)?;
            validate_full_derived_integrity(connection)?;
        }
        if has_entity_matches || has_entity_seed {
            if trace.current_query_event_id.is_empty() {
                return Err(RetrievalError::CorruptIndex(
                    "entity trace 缺少 current query event".into(),
                ));
            }
            let current = self.authoritative_trace_query_event(
                connection,
                control,
                &trace.current_query_event_id,
            )?;
            if current.role != EventRole::User {
                return Err(RetrievalError::CorruptIndex(
                    "entity trace current query 不是 user event".into(),
                ));
            }
            self.validate_entity_trace_provenance(connection, control, trace, &current)?;
            active &= control.allows_event(&current.session_id, &current.id);
        }
        if has_state_selections {
            if trace.current_query_event_id.is_empty() {
                return Err(RetrievalError::CorruptIndex(
                    "state trace 缺少 current query event".into(),
                ));
            }
            let current =
                self.authoritative_indexed_event(connection, &trace.current_query_event_id)?;
            if current.role != EventRole::User {
                return Err(RetrievalError::CorruptIndex(
                    "state trace current query 不是 user event".into(),
                ));
            }
            let current_reference = DateTime::parse_from_rfc3339(&current.created_at)
                .map_err(|_| RetrievalError::CorruptIndex("当前查询事件时间损坏".into()))?
                .with_timezone(&Utc);
            let normalized_query = normalize_match(&current.content);
            let visibility_upper = explicit_query_date(&normalized_query)
                .0
                .map(|value| value.min(current_reference))
                .unwrap_or(current_reference);
            let (snapshots, conflicts) = self.claim_snapshots_at(connection, visibility_upper)?;
            let mut selections_by_claim = HashMap::<&str, &StateSelectionTrace>::new();
            let mut selection_ranks = HashSet::new();
            for selection in &trace.state_selections {
                if selection == &StateSelectionTrace::default() {
                    continue;
                }
                if selections_by_claim
                    .insert(&selection.claim_id, selection)
                    .is_some()
                    || !selection_ranks.insert(selection.rank)
                {
                    return Err(RetrievalError::CorruptIndex(
                        "state trace 包含重复 claim 或 rank".into(),
                    ));
                }
                active &= self.state_selection_is_active(
                    connection, control, selection, &snapshots, &conflicts,
                )?;
            }
            for selection in selections_by_claim.values() {
                let snapshot = snapshots.get(&selection.claim_id).ok_or_else(|| {
                    RetrievalError::CorruptIndex("state trace claim 在查询时间不可见".into())
                })?;
                let conflict_group = if snapshot.state == "conflicted" {
                    complete_conflict_group(&conflicts, &selection.claim_id)
                } else {
                    vec![selection.claim_id.clone()]
                };
                if conflict_group
                    .iter()
                    .any(|claim_id| !selections_by_claim.contains_key(claim_id.as_str()))
                    || selection.selected
                        && conflict_group
                            .iter()
                            .any(|claim_id| !selections_by_claim[claim_id.as_str()].selected)
                {
                    return Err(RetrievalError::CorruptIndex(
                        "state trace 未完整绑定 authoritative conflict group".into(),
                    ));
                }
            }
        }
        for path in &trace.graph_paths {
            active &= self.graph_path_is_active(connection, control, path)?;
        }
        for selected in &trace.selected_evidence {
            if selected.span.event_id.is_empty() {
                return Err(RetrievalError::CorruptIndex(
                    "selected evidence 缺少权威来源".into(),
                ));
            }
            active &= self.trace_span_is_active(
                connection,
                control,
                None,
                &selected.span,
                Some(&selected.content_sha256),
                Some(selected.role),
            )?;
        }
        Ok(active)
    }

    fn write_session(
        &self,
        transaction: &Transaction<'_>,
        source: &SessionSource,
        materialize_answers: bool,
        control: &ControlState,
    ) -> RetrievalResult<SyncReport> {
        Self::require_active_session(control, &source.session.id)?;
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
        transaction
            .execute(
                "DELETE FROM memory_episode_materializations WHERE session_id=?1",
                [&source.session.id],
            )
            .map_err(|error| self.database_error(error))?;

        let all_events = derive_events(&source.session);
        let all_event_by_id = all_events
            .iter()
            .map(|event| (event.id.clone(), event))
            .collect::<HashMap<_, _>>();
        let events = all_events
            .iter()
            .filter(|event| control.allows_event(&event.session_id, &event.id))
            .cloned()
            .collect::<Vec<_>>();
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
        if existing_ids.iter().any(|id| {
            !expected_ids.contains(id.as_str())
                && !all_events.iter().any(|event| {
                    event.id == *id && !control.allows_event(&event.session_id, &event.id)
                })
        }) {
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
                    insert_memory_document(transaction, event, &span, granularity, &text)
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
            let user_id = event_id(&source.session.id, Some(&turn.id), EventRole::User);
            if !all_event_by_id.contains_key(&answer_id) {
                continue;
            }
            let mut answer_active = control.allows_event(&source.session.id, &answer_id)
                && control.allows_event(&source.session.id, &user_id);
            answer_active &= self.retrieval_trace_is_active(
                transaction,
                control,
                &turn.context_trace.retrieval,
            )?;
            let mut wrapped_history = WrappedHistoryCursor::new(
                turn.context_trace.untrusted_history_wrapped,
                &turn.context_trace.retrieval.selected_evidence,
            )
            .map_err(|message| RetrievalError::InvalidSource {
                path: source.path.clone(),
                message: message.into(),
            })?;
            let session_system_event_id = event_id(&source.session.id, None, EventRole::System);
            let session_system_end = source.session.system_prompt.chars().count();
            let session_system_hash = content_sha256(&source.session.system_prompt);
            for (ordinal, item) in turn.context_trace.context_items.iter().enumerate() {
                let is_session_system_prompt = !source.session.system_prompt.is_empty()
                    && ordinal == 0
                    && item.span.event_id == session_system_event_id
                    && item.span.start_char == 0
                    && item.span.end_char == session_system_end
                    && item.content_sha256 == session_system_hash;
                let expected_role = wrapped_history
                    .consume(item, is_session_system_prompt)
                    .map_err(|message| RetrievalError::InvalidSource {
                        path: source.path.clone(),
                        message: message.into(),
                    })?
                    .unwrap_or(item.role);
                match self.authoritative_source_span_is_active(
                    transaction,
                    control,
                    &all_event_by_id,
                    &item.span,
                    &item.content_sha256,
                    expected_role,
                )? {
                    Some(item_active) => answer_active &= item_active,
                    None => {
                        return Err(RetrievalError::InvalidSource {
                            path: source.path.clone(),
                            message: format!("回答 {} 引用了不存在的权威来源事件", turn.id),
                        });
                    }
                }
            }
            wrapped_history
                .finish()
                .map_err(|message| RetrievalError::InvalidSource {
                    path: source.path.clone(),
                    message: message.into(),
                })?;
            for selected in &turn.context_trace.retrieval.selected_evidence {
                match self.authoritative_source_span_is_active(
                    transaction,
                    control,
                    &all_event_by_id,
                    &selected.span,
                    &selected.content_sha256,
                    selected.role,
                )? {
                    Some(item_active) => answer_active &= item_active,
                    None => {
                        return Err(RetrievalError::InvalidSource {
                            path: source.path.clone(),
                            message: format!("回答 {} 引用了不存在的权威来源事件", turn.id),
                        });
                    }
                }
            }
            if !answer_active {
                continue;
            }
            let derived = derive_context(
                &source.session,
                turn,
                &all_event_by_id,
                source.legacy,
                &source.path,
            )?;
            if derived.untrusted_history_wrapped {
                let expected_identity = wrapped_history_identity(
                    &source.session.ai_name,
                    &turn.context_trace.retrieval.selected_evidence,
                );
                if derived.identity_instruction.as_deref() != Some(expected_identity.as_str()) {
                    return Err(RetrievalError::InvalidSource {
                        path: source.path.clone(),
                        message: format!("回答 {} 的不可信历史身份指令与规范元数据不匹配", turn.id),
                    });
                }
            }
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
            let mut wrapped_history = WrappedHistoryCursor::new(
                derived.untrusted_history_wrapped,
                &turn.context_trace.retrieval.selected_evidence,
            )
            .map_err(|message| RetrievalError::InvalidSource {
                path: source.path.clone(),
                message: message.into(),
            })?;
            let session_system_event_id = event_id(&source.session.id, None, EventRole::System);
            let session_system_end = source.session.system_prompt.chars().count();
            let session_system_hash = content_sha256(&source.session.system_prompt);
            for (ordinal, item) in derived.items.iter().enumerate() {
                let local = all_event_by_id.get(&item.span.event_id).copied();
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
                if !control.allows_event(&event.session_id, &event.id) {
                    return Err(RetrievalError::ExcludedEvent(event.id.clone()));
                }
                let indexed = transaction.query_row("SELECT session_id, title, created_at, updated_at, source_file, source_sha256, source_schema_version FROM indexed_sessions WHERE session_id=?1", [&event.session_id], map_session).map_err(|e| self.database_error(e))?;
                self.verify_fresh(&indexed)?;
                verify_event_hash(event)?;
                let is_session_system_prompt = !source.session.system_prompt.is_empty()
                    && ordinal == 0
                    && item.span.event_id == session_system_event_id
                    && item.span.start_char == 0
                    && item.span.end_char == session_system_end
                    && item.content_sha256 == session_system_hash;
                let wrapped_source_role = wrapped_history
                    .consume(item, is_session_system_prompt)
                    .map_err(|message| RetrievalError::InvalidSource {
                    path: source.path.clone(),
                    message: message.into(),
                })?;
                let role_matches = match wrapped_source_role {
                    Some(selected_role) => selected_role == event.role,
                    None => item.role == event.role,
                };
                if !role_matches {
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
                let generated_before_item = wrapped_source_role.is_some();
                if !inserted_generated && (item.role != EventRole::System || generated_before_item)
                {
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
                if !inserted_generated && item.role == EventRole::System && !generated_before_item {
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
            wrapped_history
                .finish()
                .map_err(|message| RetrievalError::InvalidSource {
                    path: source.path.clone(),
                    message: message.into(),
                })?;
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

    fn open_status_connection(&self) -> RetrievalResult<Connection> {
        let connection =
            Connection::open_with_flags(&self.index_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|source| self.database_error(source))?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|source| self.database_error(source))?;

        let application_id = connection
            .query_row("PRAGMA application_id", [], |row| row.get::<_, i64>(0))
            .map_err(|source| self.database_error(source))?;
        if application_id != 0 {
            return Err(RetrievalError::CorruptIndex(format!(
                "派生索引 application_id 不受支持：{application_id}"
            )));
        }
        let schema_version = connection
            .query_row("PRAGMA schema_version", [], |row| row.get::<_, i64>(0))
            .map_err(|source| self.database_error(source))?;
        if schema_version <= 0 {
            return Err(RetrievalError::CorruptIndex(format!(
                "派生索引 schema_version 无效：{schema_version}"
            )));
        }
        let user_version = connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .map_err(|source| self.database_error(source))?;
        if user_version != INDEX_SCHEMA_VERSION {
            return Err(RetrievalError::UnsupportedIndexVersion(user_version));
        }
        match read_existing_memory_state_version(&connection)
            .map_err(|source| self.database_error(source))?
        {
            Some(MEMORY_STATE_SCHEMA_VERSION) => {}
            Some(version) => return Err(RetrievalError::UnsupportedMemoryStateVersion(version)),
            None => {
                return Err(RetrievalError::CorruptIndex(
                    "派生索引缺少 memory state schema version".into(),
                ));
            }
        }
        match read_existing_graph_schema_version(&connection)
            .map_err(|source| self.database_error(source))?
        {
            Some(crate::graph::GRAPH_SCHEMA_VERSION) => {}
            Some(version) => return Err(RetrievalError::UnsupportedGraphSchemaVersion(version)),
            None => {
                return Err(RetrievalError::CorruptIndex(
                    "派生索引缺少 graph schema version".into(),
                ));
            }
        }

        const REQUIRED_STATUS_TABLES: &[&str] = &[
            "indexed_sessions",
            "events",
            "source_spans",
            "answer_contexts",
            "answer_context_items",
            "retrieval_runs",
            "retrieval_documents",
            "retrieval_documents_fts",
            "memory_documents",
            "memory_document_members",
            "memory_embeddings",
            "memory_episode_boundaries",
            "memory_episode_materializations",
            "consolidation_watermarks",
            "consolidation_batches",
            "memory_schema_meta",
            "memory_entities",
            "memory_entity_mentions",
            "memory_entity_aliases",
            "memory_claims",
            "memory_claim_evidence",
            "memory_claim_transitions",
            "memory_boundary_suggestions",
            "memory_graph_nodes",
            "memory_graph_edges",
            "memory_graph_materializations",
        ];
        let mut table_statement = connection
            .prepare("SELECT type FROM sqlite_schema WHERE name=?1")
            .map_err(|source| self.database_error(source))?;
        for table in REQUIRED_STATUS_TABLES {
            let object_type = table_statement
                .query_row([table], |row| row.get::<_, String>(0))
                .optional()
                .map_err(|source| self.database_error(source))?;
            if object_type.as_deref() != Some("table") {
                return Err(RetrievalError::CorruptIndex(format!(
                    "派生索引缺少必要表 {table}"
                )));
            }
        }
        drop(table_statement);

        const REQUIRED_STATUS_COLUMNS: &[&str] = &[
            "SELECT session_id,title,created_at,updated_at,source_file,source_sha256,source_schema_version,indexed_at FROM indexed_sessions LIMIT 0",
            "SELECT event_id,session_id,turn_id,sequence,role,created_at,content,content_sha256,reply_to_event_id,token_count,turn_status,done_reason,error FROM events LIMIT 0",
            "SELECT event_id,start_char,end_char,content_sha256 FROM source_spans LIMIT 0",
            "SELECT answer_event_id,turn_id,context_sha256,estimated_upper_tokens,exact_input_tokens,input_budget,decision,provenance_quality,request_model,request_think,request_context_window,request_max_output_tokens,identity_instruction FROM answer_contexts LIMIT 0",
            "SELECT answer_event_id,ordinal,role,event_id,start_char,end_char,content_sha256 FROM answer_context_items LIMIT 0",
            "SELECT answer_event_id,trace_json FROM retrieval_runs LIMIT 0",
            "SELECT document_id,event_id,start_char,end_char,granularity,content_sha256,exact_content,lexical_content,ngram_content FROM retrieval_documents LIMIT 0",
            "SELECT rowid,lexical_content,ngram_content FROM retrieval_documents_fts LIMIT 0",
            "SELECT document_id,session_id,granularity,source_sha256,start_sequence,end_sequence,member_count FROM memory_documents LIMIT 0",
            "SELECT document_id,ordinal,event_id,start_char,end_char,content_sha256 FROM memory_document_members LIMIT 0",
            "SELECT document_id,model,dimensions,source_sha256,index_fingerprint,vector_blob,embedded_at FROM memory_embeddings LIMIT 0",
            "SELECT session_id,before_event_id,decision_json,input_sha256 FROM memory_episode_boundaries LIMIT 0",
            "SELECT session_id,source_session_sha256,ledger_snapshot_sha256,vector_index_fingerprint,plan_input_sha256,algorithm_version,gap_minutes,topic_similarity_threshold,episode_count,boundary_count,materialized_at FROM memory_episode_materializations LIMIT 0",
            "SELECT session_id,through_sequence,through_event_id,through_event_sha256,updated_at FROM consolidation_watermarks LIMIT 0",
            "SELECT attempt_id,batch_key,session_id,from_sequence,through_sequence,trigger,model,request_json,request_sha256,input_event_ids,input_event_hashes,response_json,response_sha256,status,input_tokens,output_tokens,latency_ms,started_at,completed_at,validation_json,error_json,projection_schema_version FROM consolidation_batches LIMIT 0",
            "SELECT key,value FROM memory_schema_meta LIMIT 0",
            "SELECT entity_id,kind,canonical_name,normalized_name,disambiguation,created_session_id,created_batch_key,created_event_id,created_start,created_end,created_hash,created_at,updated_at FROM memory_entities LIMIT 0",
            "SELECT mention_id,session_id,batch_key,mention_kind,source_record_id,entity_id,entity_status,event_id,sequence,role,start_char,end_char,content_sha256,created_at FROM memory_entity_mentions LIMIT 0",
            "SELECT alias_id,entity_id,alias_text,normalized_alias,alias_kind,stable_identifier_kind,session_id,batch_key,event_id,start_char,end_char,content_sha256,proof_event_id,proof_start_char,proof_end_char,proof_sha256,identity_event_id,identity_start_char,identity_end_char,identity_sha256,created_at FROM memory_entity_aliases LIMIT 0",
            "SELECT claim_id,session_id,subject_entity_id,predicate_key,normalized_relation,object_kind,object_text,object_entity_id,normalized_object,polarity,cardinality,certainty,state,asserted_at,event_time,valid_from,valid_to,reference_time,created_batch_key,updated_batch_key,created_at,updated_at FROM memory_claims LIMIT 0",
            "SELECT evidence_id,claim_id,session_id,batch_key,event_id,sequence,role,kind,start_char,end_char,content_sha256,subject_start_char,subject_end_char,subject_sha256,relation_start_char,relation_end_char,relation_sha256,object_start_char,object_end_char,object_sha256,speech_act_event_id,speech_act_start_char,speech_act_end_char,speech_act_sha256,created_at FROM memory_claim_evidence LIMIT 0",
            "SELECT transition_id,claim_id,ordinal,from_state,to_state,reason,related_claim_id,session_id,batch_key,created_at FROM memory_claim_transitions LIMIT 0",
            "SELECT boundary_id,session_id,batch_key,before_event_id,reason,evidence_json,created_at FROM memory_boundary_suggestions LIMIT 0",
            "SELECT node_id,node_kind,source_id,session_id,granularity,source_sha256 FROM memory_graph_nodes LIMIT 0",
            "SELECT edge_id,edge_type,source_node_id,target_node_id,weight,directed,provenance_json,provenance_sha256 FROM memory_graph_edges LIMIT 0",
            "SELECT singleton,algorithm_version,vector_index_fingerprint,config_sha256,source_sha256,catalog_sha256,node_count,edge_count,materialized_at FROM memory_graph_materializations LIMIT 0",
        ];
        for query in REQUIRED_STATUS_COLUMNS {
            connection
                .prepare(query)
                .map_err(|source| self.database_error(source))?;
        }
        Ok(connection)
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
        if !matches!(version, 0 | 1 | 2 | 3 | 4 | 5 | 6 | INDEX_SCHEMA_VERSION) {
            return Err(RetrievalError::UnsupportedIndexVersion(version));
        }
        let existing_memory_version = read_existing_memory_state_version(&connection)
            .map_err(|source| self.database_error(source))?;
        if existing_memory_version.is_some_and(|value| !matches!(value, 1..=4)) {
            return Err(RetrievalError::UnsupportedMemoryStateVersion(
                existing_memory_version.expect("checked as some"),
            ));
        }
        let existing_graph_version = read_existing_graph_schema_version(&connection)
            .map_err(|source| self.database_error(source))?;
        if existing_graph_version.is_some_and(|value| value != crate::graph::GRAPH_SCHEMA_VERSION) {
            return Err(RetrievalError::UnsupportedGraphSchemaVersion(
                existing_graph_version.expect("checked as some"),
            ));
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
            prepare_memory_state_schema(&transaction, existing_memory_version)
                .map_err(|e| self.database_error(e))?;
            prepare_graph_schema(&transaction, existing_graph_version)
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
                        "DELETE FROM memory_documents;
                         DELETE FROM retrieval_documents_fts; DELETE FROM retrieval_documents;",
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
                            insert_memory_document(&transaction, event, &span, granularity, &text)
                                .map_err(|e| self.database_error(e))?;
                        }
                    }
                }
            }
            if version == 2 {
                backfill_memory_documents(&transaction).map_err(|e| self.database_error(e))?;
            }
            transaction
                .pragma_update(None, "user_version", INDEX_SCHEMA_VERSION)
                .map_err(|e| self.database_error(e))?;
            transaction.commit().map_err(|e| self.database_error(e))?;
        } else if matches!(version, 0 | 3 | 4 | 5) {
            let transaction = connection
                .transaction()
                .map_err(|e| self.database_error(e))?;
            if version == 5 {
                transaction
                    .execute_batch(
                        "DROP TRIGGER IF EXISTS memory_documents_before_source_span_delete;",
                    )
                    .map_err(|e| self.database_error(e))?;
            }
            transaction
                .execute_batch(SCHEMA_SQL)
                .map_err(|e| self.database_error(e))?;
            prepare_memory_state_schema(&transaction, existing_memory_version)
                .map_err(|e| self.database_error(e))?;
            prepare_graph_schema(&transaction, existing_graph_version)
                .map_err(|e| self.database_error(e))?;
            if matches!(version, 3 | 4) {
                backfill_memory_documents(&transaction).map_err(|e| self.database_error(e))?;
            }
            transaction
                .pragma_update(None, "user_version", INDEX_SCHEMA_VERSION)
                .map_err(|e| self.database_error(e))?;
            transaction.commit().map_err(|e| self.database_error(e))?;
        } else {
            let transaction = connection
                .transaction()
                .map_err(|e| self.database_error(e))?;
            transaction
                .execute_batch(SCHEMA_SQL)
                .map_err(|source| self.database_error(source))?;
            prepare_memory_state_schema(&transaction, existing_memory_version)
                .map_err(|e| self.database_error(e))?;
            prepare_graph_schema(&transaction, existing_graph_version)
                .map_err(|e| self.database_error(e))?;
            transaction
                .pragma_update(None, "user_version", INDEX_SCHEMA_VERSION)
                .map_err(|e| self.database_error(e))?;
            transaction.commit().map_err(|e| self.database_error(e))?;
        }
        Ok(connection)
    }

    pub(crate) fn get_session_from_connection(
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

    pub(crate) fn get_event_from_connection(
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

    pub(crate) fn verify_fresh(&self, session: &IndexedSession) -> RetrievalResult<()> {
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

    pub(crate) fn verify_indexed_session_source_projection(
        &self,
        connection: &Connection,
        session_id: &str,
    ) -> RetrievalResult<()> {
        let control = self.replay_control_state_under_guard()?;
        self.verify_indexed_session_source_projection_with_control(connection, session_id, &control)
    }

    pub(crate) fn verify_indexed_session_source_projection_with_control(
        &self,
        connection: &Connection,
        session_id: &str,
        control: &ControlState,
    ) -> RetrievalResult<()> {
        let indexed = self.get_session_from_connection(connection, session_id)?;
        if indexed.source_file != format!("{session_id}.json")
            || indexed.source_sha256.len() != 64
            || !indexed
                .source_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "会话 {session_id} 的源文件绑定不精确"
            )));
        }
        let path = self.root.join(&indexed.source_file);
        let bytes = fs::read(path).map_err(|_| RetrievalError::StaleIndex {
            session_id: session_id.to_owned(),
        })?;
        if bytes_sha256(&bytes) != indexed.source_sha256 {
            return Err(RetrievalError::StaleIndex {
                session_id: session_id.to_owned(),
            });
        }
        let mut source: Session =
            serde_json::from_slice(&bytes).map_err(|error| RetrievalError::InvalidSource {
                path: self.root.join(&indexed.source_file),
                message: error.to_string(),
            })?;
        source.normalize_legacy_provenance();
        source
            .validate()
            .map_err(|error| RetrievalError::InvalidSource {
                path: self.root.join(&indexed.source_file),
                message: error.to_string(),
            })?;
        if source.id != session_id {
            return Err(RetrievalError::StaleIndex {
                session_id: session_id.to_owned(),
            });
        }
        let indexed_events = connection.prepare(
            "SELECT event_id, session_id, turn_id, sequence, role, created_at, content, content_sha256,
                    reply_to_event_id, token_count, turn_status, done_reason, error
             FROM events WHERE session_id=?1 ORDER BY sequence",
        ).and_then(|mut statement| statement.query_map([session_id], map_event)?.collect::<rusqlite::Result<Vec<_>>>() )
            .map_err(|error| self.database_error(error))?
            .into_iter()
            .filter(|event| control.allows_event(&event.session_id, &event.id))
            .collect::<Vec<_>>();
        let expected = derive_events(&source)
            .into_iter()
            .filter(|event| control.allows_event(&event.session_id, &event.id))
            .collect::<Vec<_>>();
        if indexed_events != expected {
            return Err(RetrievalError::StaleIndex {
                session_id: session_id.to_owned(),
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

    pub(crate) fn validated_retrieval_runs_from_connection(
        &self,
        connection: &Connection,
    ) -> RetrievalResult<Vec<(String, RetrievalTrace)>> {
        let control = self.replay_control_state_under_guard()?;
        self.validated_retrieval_runs_from_connection_with_control(connection, &control)
    }

    pub(crate) fn validated_retrieval_runs_from_connection_with_control(
        &self,
        connection: &Connection,
        control: &ControlState,
    ) -> RetrievalResult<Vec<(String, RetrievalTrace)>> {
        let mut statement = connection
            .prepare(
                "SELECT answer_event_id,trace_json FROM retrieval_runs ORDER BY answer_event_id",
            )
            .map_err(|error| self.database_error(error))?;
        let mut stored = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| self.database_error(error))?
            .collect::<rusqlite::Result<BTreeMap<_, _>>>()
            .map_err(|error| self.database_error(error))?;
        drop(statement);
        let mut validated = Vec::new();
        for source in self.load_all_sources()? {
            if !control.allows_session(&source.session.id) {
                continue;
            }
            self.verify_indexed_session_source_projection_with_control(
                connection,
                &source.session.id,
                control,
            )?;
            for turn in &source.session.turns {
                if !has_assistant_event(&source.session, turn) {
                    continue;
                }
                let answer_event_id =
                    event_id(&source.session.id, Some(&turn.id), EventRole::Assistant);
                let user_event_id = event_id(&source.session.id, Some(&turn.id), EventRole::User);
                if !control.allows_event(&source.session.id, &answer_event_id)
                    || !control.allows_event(&source.session.id, &user_event_id)
                {
                    continue;
                }
                let canonical_json =
                    serde_json::to_string(&turn.context_trace.retrieval).map_err(|error| {
                        RetrievalError::CorruptIndex(format!(
                            "原始 retrieval run {answer_event_id} 无法规范序列化：{error}"
                        ))
                    })?;
                let json = stored.get(&answer_event_id);
                if json.is_some_and(|json| json != &canonical_json) {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "retrieval run {answer_event_id} 与原始 trace 不一致"
                    )));
                }
                if !self.retrieval_trace_is_active(
                    connection,
                    control,
                    &turn.context_trace.retrieval,
                )? {
                    if json.is_some() {
                        return Err(RetrievalError::CorruptIndex(
                            "受控 retrieval run 不应被物化".into(),
                        ));
                    }
                    continue;
                }
                let json = stored.remove(&answer_event_id).ok_or_else(|| {
                    RetrievalError::CorruptIndex(format!(
                        "回答 {answer_event_id} 缺少 retrieval run"
                    ))
                })?;
                let mut actual: RetrievalTrace = serde_json::from_str(&json).map_err(|error| {
                    RetrievalError::CorruptIndex(format!(
                        "retrieval run {answer_event_id} JSON 无效：{error}"
                    ))
                })?;
                actual.normalize_usage();
                validated.push((answer_event_id, actual));
            }
        }
        if !stored.is_empty() {
            return Err(RetrievalError::CorruptIndex(
                "retrieval_runs 包含未绑定的回答".into(),
            ));
        }
        Ok(validated)
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

fn insert_memory_document(
    transaction: &Transaction<'_>,
    event: &StoredEvent,
    span: &SourceSpan,
    granularity: RetrievalDocumentGranularity,
    content: &str,
) -> rusqlite::Result<()> {
    let granularity = match granularity {
        RetrievalDocumentGranularity::Message => "message",
        RetrievalDocumentGranularity::Fragment => "fragment",
        RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session => {
            return Err(rusqlite::Error::ToSqlConversionFailure(Box::new(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "lexical document must be message or fragment",
                ),
            )));
        }
    };
    let document_id = format!("{}:{}:{}", event.id, span.start_char, span.end_char);
    let source_sha256 = content_sha256(content);
    transaction.execute(
        "INSERT INTO memory_documents
         (document_id, session_id, granularity, source_sha256, start_sequence, end_sequence, member_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1)
         ON CONFLICT(document_id) DO UPDATE SET
         session_id=excluded.session_id, granularity=excluded.granularity,
         source_sha256=excluded.source_sha256, start_sequence=excluded.start_sequence,
         end_sequence=excluded.end_sequence, member_count=excluded.member_count",
        params![
            document_id,
            event.session_id,
            granularity,
            source_sha256,
            usize_to_i64(event.sequence)?,
            usize_to_i64(event.sequence)?,
        ],
    )?;
    transaction.execute(
        "DELETE FROM memory_document_members WHERE document_id=?1",
        [&document_id],
    )?;
    transaction.execute(
        "INSERT INTO memory_document_members
         (document_id, ordinal, event_id, start_char, end_char, content_sha256)
         VALUES (?1, 0, ?2, ?3, ?4, ?5)",
        params![
            document_id,
            event.id,
            usize_to_i64(span.start_char)?,
            usize_to_i64(span.end_char)?,
            source_sha256,
        ],
    )?;
    Ok(())
}

type EpisodeSnapshot = (
    Vec<EpisodeInputMessage>,
    Option<u64>,
    Vec<EpisodeBoundarySuggestion>,
    String,
);

fn load_episode_snapshot(
    transaction: &Connection,
    session_id: &str,
    spec: &VectorIndexSpec,
    fingerprint: &str,
) -> RetrievalResult<EpisodeSnapshot> {
    let watermark = load_validated_consolidation_watermark(transaction, session_id)?;
    let mut entities = HashMap::<String, std::collections::BTreeSet<String>>::new();
    // Historical resolution is immutable: later entity promotion must never alter
    // the entity set that was observed for an older episode input.
    for query in ["SELECT event_id, entity_id FROM memory_entity_mentions
         WHERE session_id=?1 AND entity_status='resolved' ORDER BY event_id, entity_id, mention_id"]
    {
        let mut statement = transaction
            .prepare(query)
            .map_err(|error| RetrievalError::CorruptIndex(format!("读取实体证据失败：{error}")))?;
        let rows = statement
            .query_map([session_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| RetrievalError::CorruptIndex(format!("读取实体证据失败：{error}")))?;
        for row in rows {
            let (event, entity) = row.map_err(|error| {
                RetrievalError::CorruptIndex(format!("读取实体证据失败：{error}"))
            })?;
            entities.entry(event).or_default().insert(entity);
        }
    }
    let mut embeddings = HashMap::<String, Vec<f32>>::new();
    let mut embedding_statement = transaction.prepare(
        "SELECT d.document_id, e.vector_blob FROM memory_documents d JOIN memory_embeddings e ON e.document_id=d.document_id
         WHERE d.session_id=?1 AND d.granularity='message' AND e.model=?2 AND e.dimensions=?3
           AND e.source_sha256=d.source_sha256 AND e.index_fingerprint=?4",
    ).map_err(|error| RetrievalError::CorruptIndex(format!("读取消息向量失败：{error}")))?;
    let embedding_rows = embedding_statement
        .query_map(
            params![
                session_id,
                spec.model,
                usize_to_i64(spec.dimensions)
                    .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?,
                fingerprint
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .map_err(|error| RetrievalError::CorruptIndex(format!("读取消息向量失败：{error}")))?;
    for row in embedding_rows {
        let (id, bytes) = row
            .map_err(|error| RetrievalError::CorruptIndex(format!("读取消息向量失败：{error}")))?;
        embeddings.insert(
            id,
            decode_f32_le(&bytes, spec.dimensions)
                .map_err(|error| RetrievalError::CorruptIndex(format!("消息向量损坏：{error}")))?,
        );
    }
    let mut statement = transaction
        .prepare(
            "SELECT event_id, sequence, role, created_at, content, content_sha256
         FROM events WHERE session_id=?1 AND role IN ('user','assistant') ORDER BY sequence",
        )
        .map_err(|error| RetrievalError::CorruptIndex(format!("读取原文事件失败：{error}")))?;
    let rows = statement
        .query_map([session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })
        .map_err(|error| RetrievalError::CorruptIndex(format!("读取原文事件失败：{error}")))?;
    let mut messages = Vec::new();
    for row in rows {
        let (event_id, sequence, role, created_at, content, hash) = row
            .map_err(|error| RetrievalError::CorruptIndex(format!("读取原文事件失败：{error}")))?;
        if content.trim().is_empty() {
            continue;
        }
        let actual_hash = content_sha256(&content);
        if actual_hash != hash {
            return Err(RetrievalError::CorruptIndex(format!(
                "事件 {event_id} 的内容哈希不匹配"
            )));
        }
        let sequence = i64_to_u64(sequence)
            .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let expected_end = content.chars().count();
        let source_spans = transaction
            .prepare("SELECT content_sha256 FROM source_spans WHERE event_id=?1 AND start_char=0 AND end_char=?2")
            .map_err(|error| RetrievalError::CorruptIndex(format!("读取原文 span 失败：{error}")))?
            .query_map(params![event_id, usize_to_i64(expected_end).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?], |row| row.get::<_, String>(0))
            .map_err(|error| RetrievalError::CorruptIndex(format!("读取原文 span 失败：{error}")))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| RetrievalError::CorruptIndex(format!("读取原文 span 失败：{error}")))?;
        if source_spans.as_slice() != [actual_hash.clone()] {
            return Err(RetrievalError::CorruptIndex(format!(
                "事件 {event_id} 缺少唯一完整原文 span"
            )));
        }
        let document_id = format!("{event_id}:0:{expected_end}");
        let members = transaction.prepare("SELECT d.source_sha256,d.start_sequence,d.end_sequence,d.member_count,m.ordinal,m.event_id,m.start_char,m.end_char,m.content_sha256 FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id WHERE d.document_id=?1 AND d.session_id=?2 AND d.granularity='message'")
            .map_err(|e| RetrievalError::CorruptIndex(format!("读取 message document 失败：{e}")))?
            .query_map(params![document_id, session_id], |row| Ok((row.get::<_,String>(0)?,row.get::<_,i64>(1)?,row.get::<_,i64>(2)?,row.get::<_,i64>(3)?,row.get::<_,i64>(4)?,row.get::<_,String>(5)?,row.get::<_,i64>(6)?,row.get::<_,i64>(7)?,row.get::<_,String>(8)?)))
            .map_err(|e| RetrievalError::CorruptIndex(format!("读取 message document 失败：{e}")))?.collect::<Result<Vec<_>,_>>().map_err(|e| RetrievalError::CorruptIndex(format!("读取 message document 失败：{e}")))?;
        if members.len() != 1 {
            return Err(RetrievalError::CorruptIndex(format!(
                "事件 {event_id} 未恰好对应一个完整 message document"
            )));
        }
        let (
            document_hash,
            start_seq,
            end_seq,
            count,
            ordinal,
            member_event,
            start,
            end,
            member_hash,
        ) = members.into_iter().next().expect("checked");
        let start =
            i64_to_usize(start).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let end =
            i64_to_usize(end).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let sequence_i64 = i64::try_from(sequence)
            .map_err(|_| RetrievalError::CorruptIndex(format!("事件 {event_id} 序号溢出")))?;
        if document_hash != actual_hash
            || start_seq != sequence_i64
            || end_seq != start_seq
            || count != 1
            || ordinal != 0
            || member_event != event_id
            || start != 0
            || end != expected_end
            || member_hash != actual_hash
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "事件 {event_id} 缺少唯一完整 message 成员"
            )));
        }
        let role = match role.as_str() {
            "user" => EventRole::User,
            "assistant" => EventRole::Assistant,
            _ => {
                return Err(RetrievalError::CorruptIndex(format!(
                    "事件 {event_id} 角色损坏"
                )));
            }
        };
        messages.push(EpisodeInputMessage {
            member: EpisodeMember {
                document_id: document_id.clone(),
                event_id: event_id.clone(),
                sequence,
                role,
                span: SourceSpan {
                    event_id: event_id.clone(),
                    start_char: 0,
                    end_char: expected_end,
                },
                content_sha256: actual_hash,
            },
            created_at,
            resolved_entity_ids: entities.remove(&event_id).unwrap_or_default(),
            embedding: embeddings.remove(&document_id),
        });
    }
    let mut suggestions = Vec::new();
    let mut statement = transaction.prepare("SELECT DISTINCT before_event_id, reason FROM memory_boundary_suggestions WHERE session_id=?1 ORDER BY before_event_id, reason")
        .map_err(|error| RetrievalError::CorruptIndex(format!("读取边界建议失败：{error}")))?;
    let rows = statement
        .query_map([session_id], |row| {
            Ok(EpisodeBoundarySuggestion {
                before_event_id: row.get(0)?,
                reason: row.get(1)?,
            })
        })
        .map_err(|error| RetrievalError::CorruptIndex(format!("读取边界建议失败：{error}")))?;
    for row in rows {
        suggestions.push(
            row.map_err(|error| {
                RetrievalError::CorruptIndex(format!("读取边界建议失败：{error}"))
            })?,
        );
    }
    let snapshot = ledger_snapshot_hash(session_id, watermark, &messages, &suggestions);
    Ok((messages, watermark, suggestions, snapshot))
}

fn persist_episode_plan(
    transaction: &Transaction<'_>,
    plan: &EpisodeMaterializationReport,
    ledger_snapshot_sha256: String,
    fingerprint: &str,
    config: &MemoryConfig,
) -> RetrievalResult<()> {
    transaction.execute(
        "DELETE FROM memory_embeddings WHERE document_id IN (SELECT document_id FROM memory_documents WHERE session_id=?1 AND granularity IN ('episode','session'))",
        [&plan.session_id],
    ).map_err(|error| RetrievalError::CorruptIndex(format!("清理旧 aggregate 向量失败：{error}")))?;
    let documents = aggregate_documents_for_plan(plan);
    let ids = documents
        .iter()
        .map(|document| document.document_id.as_str())
        .collect::<Vec<_>>();
    if ids.is_empty() { transaction.execute("DELETE FROM memory_documents WHERE session_id=?1 AND granularity IN ('episode','session')", [plan.session_id.as_str()]) }
    else { let placeholders = std::iter::repeat_n("?", ids.len()).collect::<Vec<_>>().join(","); let mut values: Vec<&dyn rusqlite::ToSql> = vec![&plan.session_id]; for id in &ids { values.push(id); } transaction.execute(&format!("DELETE FROM memory_documents WHERE session_id=? AND granularity IN ('episode','session') AND document_id NOT IN ({placeholders})"), rusqlite::params_from_iter(values)) }.map_err(|error| RetrievalError::CorruptIndex(format!("清理旧 episode 失败：{error}")))?;
    for document in &documents {
        upsert_aggregate_document(transaction, document)?;
    }
    transaction
        .execute(
            "DELETE FROM memory_episode_boundaries WHERE session_id=?1",
            [&plan.session_id],
        )
        .map_err(|error| RetrievalError::CorruptIndex(format!("清理边界审计失败：{error}")))?;
    for decision in &plan.boundary_decisions {
        let json = serde_json::to_string(decision).map_err(|error| {
            RetrievalError::CorruptIndex(format!("边界审计序列化失败：{error}"))
        })?;
        transaction.execute("INSERT INTO memory_episode_boundaries(session_id,before_event_id,decision_json,input_sha256) VALUES(?1,?2,?3,?4)", params![plan.session_id, decision.before_event_id, json, decision.input_sha256]).map_err(|error| RetrievalError::CorruptIndex(format!("写入边界审计失败：{error}")))?;
    }
    transaction.execute("INSERT INTO memory_episode_materializations(session_id,source_session_sha256,ledger_snapshot_sha256,vector_index_fingerprint,plan_input_sha256,algorithm_version,gap_minutes,topic_similarity_threshold,episode_count,boundary_count,materialized_at) VALUES(?1,?2,?3,?4,?5,1,?6,0.60,?7,?8,?9) ON CONFLICT(session_id) DO UPDATE SET source_session_sha256=excluded.source_session_sha256,ledger_snapshot_sha256=excluded.ledger_snapshot_sha256,vector_index_fingerprint=excluded.vector_index_fingerprint,plan_input_sha256=excluded.plan_input_sha256,algorithm_version=excluded.algorithm_version,gap_minutes=excluded.gap_minutes,topic_similarity_threshold=excluded.topic_similarity_threshold,episode_count=excluded.episode_count,boundary_count=excluded.boundary_count,materialized_at=excluded.materialized_at", params![plan.session_id, plan.source_session_sha256, ledger_snapshot_sha256, fingerprint, plan.plan_input_sha256, i64::try_from(config.episode_gap_minutes).map_err(|_| RetrievalError::CorruptIndex("gap minutes overflow".into()))?, usize_to_i64(plan.episode_documents.len()).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?, usize_to_i64(plan.boundary_decisions.len()).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?, Utc::now().to_rfc3339()]).map_err(|error| RetrievalError::CorruptIndex(format!("写入 episode freshness 失败：{error}")))?;
    Ok(())
}

fn aggregate_documents_for_plan(plan: &EpisodeMaterializationReport) -> Vec<EpisodeDocument> {
    let mut documents = plan.episode_documents.clone();
    if !documents.is_empty() {
        let members = documents
            .iter()
            .flat_map(|document| document.members.clone())
            .collect::<Vec<_>>();
        documents.push(EpisodeDocument {
            document_id: session_document_id(&plan.session_id),
            session_id: plan.session_id.clone(),
            granularity: "session".into(),
            source_sha256: aggregate_members_hash("session", &plan.session_id, &members),
            start_sequence: members.first().expect("documents nonempty").sequence,
            end_sequence: members.last().expect("documents nonempty").sequence,
            members,
        });
    }
    documents
}

fn upsert_aggregate_document(
    transaction: &Transaction<'_>,
    document: &EpisodeDocument,
) -> RetrievalResult<()> {
    transaction.execute("INSERT INTO memory_documents(document_id,session_id,granularity,source_sha256,start_sequence,end_sequence,member_count) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(document_id) DO UPDATE SET session_id=excluded.session_id,granularity=excluded.granularity,source_sha256=excluded.source_sha256,start_sequence=excluded.start_sequence,end_sequence=excluded.end_sequence,member_count=excluded.member_count", params![document.document_id, document.session_id, document.granularity, document.source_sha256, i64::try_from(document.start_sequence).map_err(|_| RetrievalError::CorruptIndex("aggregate start sequence overflow".into()))?, i64::try_from(document.end_sequence).map_err(|_| RetrievalError::CorruptIndex("aggregate end sequence overflow".into()))?, usize_to_i64(document.members.len()).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?]).map_err(|error| RetrievalError::CorruptIndex(format!("写入 aggregate document 失败：{error}")))?;
    transaction
        .execute(
            "DELETE FROM memory_document_members WHERE document_id=?1",
            [&document.document_id],
        )
        .map_err(|error| {
            RetrievalError::CorruptIndex(format!("替换 aggregate members 失败：{error}"))
        })?;
    for (ordinal, member) in document.members.iter().enumerate() {
        if member.span.start_char != 0 {
            return Err(RetrievalError::CorruptIndex(
                "aggregate member 不是完整原文 message".into(),
            ));
        }
        transaction.execute("INSERT INTO memory_document_members(document_id,ordinal,event_id,start_char,end_char,content_sha256) VALUES(?1,?2,?3,?4,?5,?6)", params![document.document_id, usize_to_i64(ordinal).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?, member.event_id, usize_to_i64(member.span.start_char).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?, usize_to_i64(member.span.end_char).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?, member.content_sha256]).map_err(|error| RetrievalError::CorruptIndex(format!("写入 aggregate member 失败：{error}")))?;
    }
    verify_aggregate_document(transaction, document)?;
    Ok(())
}

fn verify_aggregate_document(
    transaction: &Transaction<'_>,
    document: &EpisodeDocument,
) -> RetrievalResult<()> {
    let mut statement = transaction.prepare("SELECT m.ordinal,m.event_id,m.start_char,m.end_char,m.content_sha256,e.sequence,e.role FROM memory_document_members m JOIN events e ON e.event_id=m.event_id WHERE m.document_id=?1 ORDER BY m.ordinal")
        .map_err(|error| RetrievalError::CorruptIndex(format!("验证 aggregate members 失败：{error}")))?;
    let rows = statement
        .query_map([&document.document_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, String>(6)?,
            ))
        })
        .map_err(|error| {
            RetrievalError::CorruptIndex(format!("验证 aggregate members 失败：{error}"))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            RetrievalError::CorruptIndex(format!("验证 aggregate members 失败：{error}"))
        })?;
    if rows.len() != document.members.len() {
        return Err(RetrievalError::CorruptIndex(
            "aggregate member_count 不匹配".into(),
        ));
    }
    let mut members = Vec::new();
    for (ordinal, (stored_ordinal, event, start, end, hash, sequence, role)) in
        rows.into_iter().enumerate()
    {
        if stored_ordinal
            != usize_to_i64(ordinal)
                .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?
        {
            return Err(RetrievalError::CorruptIndex(
                "aggregate member ordinal 不连续".into(),
            ));
        }
        let role = match role.as_str() {
            "user" => EventRole::User,
            "assistant" => EventRole::Assistant,
            _ => {
                return Err(RetrievalError::CorruptIndex(
                    "aggregate member 角色无效".into(),
                ));
            }
        };
        members.push(EpisodeMember {
            document_id: format!("{event}:0:{end}"),
            event_id: event.clone(),
            sequence: i64_to_u64(sequence)
                .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?,
            role,
            span: SourceSpan {
                event_id: event,
                start_char: i64_to_usize(start)
                    .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?,
                end_char: i64_to_usize(end)
                    .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?,
            },
            content_sha256: hash,
        });
    }
    if members.first().map(|member| member.sequence) != Some(document.start_sequence)
        || members.last().map(|member| member.sequence) != Some(document.end_sequence)
        || aggregate_members_hash(&document.granularity, &document.session_id, &members)
            != document.source_sha256
    {
        return Err(RetrievalError::CorruptIndex(
            "aggregate range 或 source hash 不匹配".into(),
        ));
    }
    Ok(())
}

fn backfill_memory_documents(transaction: &Transaction<'_>) -> rusqlite::Result<()> {
    transaction.execute_batch(
        "INSERT INTO memory_documents
         (document_id, session_id, granularity, source_sha256, start_sequence, end_sequence, member_count)
         SELECT d.document_id, e.session_id, d.granularity, d.content_sha256,
                e.sequence, e.sequence, 1
         FROM retrieval_documents d JOIN events e ON e.event_id=d.event_id
         WHERE d.granularity IN ('message', 'fragment')
         ORDER BY d.document_id
         ON CONFLICT(document_id) DO UPDATE SET
         session_id=excluded.session_id, granularity=excluded.granularity,
         source_sha256=excluded.source_sha256, start_sequence=excluded.start_sequence,
         end_sequence=excluded.end_sequence, member_count=excluded.member_count;
         DELETE FROM memory_document_members
         WHERE document_id IN (
             SELECT document_id FROM retrieval_documents
             WHERE granularity IN ('message', 'fragment')
         );
         INSERT INTO memory_document_members
         (document_id, ordinal, event_id, start_char, end_char, content_sha256)
         SELECT d.document_id, 0, d.event_id, d.start_char, d.end_char, d.content_sha256
         FROM retrieval_documents d
         WHERE d.granularity IN ('message', 'fragment')
         ORDER BY d.document_id
         ON CONFLICT(document_id, ordinal) DO UPDATE SET
         event_id=excluded.event_id, start_char=excluded.start_char,
         end_char=excluded.end_char, content_sha256=excluded.content_sha256;",
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregateReadiness {
    Ready,
    Stale,
}

#[derive(Debug)]
struct AggregateSessionAudit {
    readiness: AggregateReadiness,
    canonical_vector_blobs: HashMap<String, Vec<u8>>,
}

impl AggregateSessionAudit {
    fn stale() -> Self {
        Self {
            readiness: AggregateReadiness::Stale,
            canonical_vector_blobs: HashMap::new(),
        }
    }

    fn ready(canonical_vector_blobs: HashMap<String, Vec<u8>>) -> Self {
        Self {
            readiness: AggregateReadiness::Ready,
            canonical_vector_blobs,
        }
    }
}

#[derive(Debug)]
struct PreparedEmbedding {
    document_id: String,
    session_id: String,
    granularity: String,
    source_sha256: String,
    vector_blob: Vec<u8>,
}

const EMBEDDING_CATALOG_ALGORITHM_VERSION: u32 = 1;

#[derive(Debug)]
struct PreparedCatalogEmbedding {
    document_id: String,
    source_sha256: String,
    vector_blob: Vec<u8>,
}

#[derive(Debug)]
struct RawEmbedding {
    model: String,
    dimensions: i64,
    source_sha256: String,
    fingerprint: String,
    vector_blob: Vec<u8>,
}

fn validate_embedding_spec(spec: &VectorIndexSpec) -> RetrievalResult<()> {
    spec.validate()
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))
}

fn embedding_fingerprint(spec: &VectorIndexSpec) -> RetrievalResult<String> {
    spec.fingerprint()
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))
}

fn is_unit_vector(vector: &[f32]) -> bool {
    let norm = vector.iter().try_fold(0.0_f64, |sum, value| {
        value
            .is_finite()
            .then_some(sum + f64::from(*value) * f64::from(*value))
    });
    norm.is_some_and(|value| value.is_finite() && (value.sqrt() - 1.0).abs() <= 1e-5)
}

fn raw_embedding(
    connection: &Connection,
    document_id: &str,
) -> RetrievalResult<Option<RawEmbedding>> {
    connection
        .query_row(
            "SELECT model,dimensions,source_sha256,index_fingerprint,vector_blob
         FROM memory_embeddings WHERE document_id=?1",
            [document_id],
            |row| {
                Ok(RawEmbedding {
                    model: row.get(0)?,
                    dimensions: row.get(1)?,
                    source_sha256: row.get(2)?,
                    fingerprint: row.get(3)?,
                    vector_blob: row.get(4)?,
                })
            },
        )
        .optional()
        .map_err(|error| RetrievalError::CorruptIndex(format!("读取 embedding row 失败：{error}")))
}

fn embedding_equals(
    existing: Option<&RawEmbedding>,
    spec: &VectorIndexSpec,
    fingerprint: &str,
    source: &str,
    blob: &[u8],
) -> bool {
    existing.is_some_and(|row| {
        row.model == spec.model
            && row.dimensions == i64::try_from(spec.dimensions).unwrap_or(-1)
            && row.source_sha256 == source
            && row.fingerprint == fingerprint
            && row.vector_blob == blob
    })
}

fn prepare_complete_catalog_writes<'a>(
    spec: &VectorIndexSpec,
    expected: impl Iterator<Item = (&'a str, &'a str)>,
    writes: &[EmbeddingWrite],
) -> RetrievalResult<Vec<PreparedCatalogEmbedding>> {
    let expected = expected
        .map(|(id, source)| (id.to_owned(), source.to_owned()))
        .collect::<Vec<_>>();
    let mut supplied = HashMap::with_capacity(writes.len());
    for write in writes {
        if supplied.insert(write.document_id.as_str(), write).is_some() {
            return Err(RetrievalError::CorruptIndex(format!(
                "embedding catalog 包含重复文档 {}",
                write.document_id
            )));
        }
    }
    if supplied.len() != expected.len()
        || expected
            .iter()
            .any(|(id, _)| !supplied.contains_key(id.as_str()))
    {
        return Err(RetrievalError::CorruptIndex(
            "embedding publication 必须精确覆盖完整 catalog".into(),
        ));
    }
    expected
        .into_iter()
        .map(|(document_id, source_sha256)| {
            let write = supplied[document_id.as_str()];
            if write.expected_source_sha256 != source_sha256 {
                return Err(RetrievalError::CorruptIndex(format!(
                    "文档 {document_id} 的源哈希不匹配"
                )));
            }
            if write.vector.len() != spec.dimensions {
                return Err(RetrievalError::CorruptIndex(format!(
                    "文档 {document_id} 的向量维度不匹配"
                )));
            }
            if !is_unit_vector(&write.vector) {
                return Err(RetrievalError::CorruptIndex(format!(
                    "文档 {document_id} 的向量必须为有限非零单位向量"
                )));
            }
            let vector_blob = encode_f32_le(&write.vector)
                .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
            Ok(PreparedCatalogEmbedding {
                document_id,
                source_sha256,
                vector_blob,
            })
        })
        .collect()
}

fn verify_published_catalog(
    connection: &Connection,
    spec: &VectorIndexSpec,
    fingerprint: &str,
    prepared: &[PreparedCatalogEmbedding],
    granularities: &str,
) -> RetrievalResult<()> {
    for write in prepared {
        let row = raw_embedding(connection, &write.document_id)?.ok_or_else(|| {
            RetrievalError::CorruptIndex(format!("文档 {} 写后缺少 embedding", write.document_id))
        })?;
        if !embedding_equals(
            Some(&row),
            spec,
            fingerprint,
            &write.source_sha256,
            &write.vector_blob,
        ) {
            return Err(RetrievalError::CorruptIndex(format!(
                "文档 {} 写后 embedding 不精确",
                write.document_id
            )));
        }
    }
    let sql = format!(
        "SELECT e.document_id FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id WHERE d.granularity IN ({granularities}) AND e.model=?1 AND e.dimensions=?2 AND e.index_fingerprint=?3 AND e.source_sha256=d.source_sha256 ORDER BY e.document_id"
    );
    let actual = connection
        .prepare(&sql)
        .and_then(|mut statement| {
            statement
                .query_map(
                    params![
                        spec.model,
                        i64::try_from(spec.dimensions).unwrap_or(-1),
                        fingerprint
                    ],
                    |row| row.get::<_, String>(0),
                )?
                .collect::<rusqlite::Result<Vec<_>>>()
        })
        .map_err(|error| {
            RetrievalError::CorruptIndex(format!("读取写后 compatible embedding 集合失败：{error}"))
        })?;
    let mut expected = prepared
        .iter()
        .map(|write| write.document_id.clone())
        .collect::<Vec<_>>();
    expected.sort();
    if actual != expected {
        return Err(RetrievalError::CorruptIndex(
            "写后 compatible embedding 集合不精确".into(),
        ));
    }
    Ok(())
}

fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hash_field(hasher, value.as_bytes());
}
fn hash_usize(hasher: &mut Sha256, value: usize) {
    hash_field(hasher, &(value as u64).to_be_bytes());
}

fn hash_session_catalog(
    store: &RetrievalStore,
    connection: &Connection,
    hasher: &mut Sha256,
) -> RetrievalResult<Vec<String>> {
    let mut statement = connection.prepare("SELECT session_id,title,created_at,updated_at,source_file,source_sha256,source_schema_version FROM indexed_sessions ORDER BY session_id")
        .map_err(|error| store.database_error(error))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
            ))
        })
        .map_err(|error| store.database_error(error))?;
    let mut ids = Vec::new();
    for row in rows {
        let (id, title, created, updated, file, source, version) =
            row.map_err(|error| store.database_error(error))?;
        for value in [&id, &title, &created, &updated, &file, &source] {
            hash_string(hasher, value);
        }
        hash_field(hasher, &version.to_be_bytes());
        ids.push(id);
    }
    Ok(ids)
}

fn hash_embedding_rows(
    store: &RetrievalStore,
    connection: &Connection,
    hasher: &mut Sha256,
    granularities: &str,
) -> RetrievalResult<()> {
    let sql = format!("SELECT e.document_id,e.model,e.dimensions,e.source_sha256,e.index_fingerprint,e.vector_blob,e.embedded_at
        FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id
        WHERE d.granularity IN ({granularities}) ORDER BY e.document_id");
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| store.database_error(error))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Vec<u8>>(5)?,
                row.get::<_, String>(6)?,
            ))
        })
        .map_err(|error| store.database_error(error))?;
    for row in rows {
        let (id, model, dim, source, fingerprint, blob, time) =
            row.map_err(|error| store.database_error(error))?;
        for value in [&id, &model, &source, &fingerprint] {
            hash_string(hasher, value);
        }
        hash_field(hasher, &dim.to_be_bytes());
        hash_field(hasher, &blob);
        hash_string(hasher, &time);
    }
    Ok(())
}

pub(crate) fn load_leaf_embedding_snapshot(
    store: &RetrievalStore,
    connection: &Connection,
    spec: &VectorIndexSpec,
) -> RetrievalResult<LeafEmbeddingSnapshot> {
    let control = store.replay_control_state_under_guard()?;
    load_leaf_embedding_snapshot_with_control(store, connection, spec, &control)
}

pub(crate) fn load_leaf_embedding_snapshot_with_control(
    store: &RetrievalStore,
    connection: &Connection,
    spec: &VectorIndexSpec,
    control: &ControlState,
) -> RetrievalResult<LeafEmbeddingSnapshot> {
    let fingerprint = embedding_fingerprint(spec)?;
    let mut hasher = Sha256::new();
    hash_string(&mut hasher, "hippocampus.embedding.catalog.leaf");
    hash_field(
        &mut hasher,
        &EMBEDDING_CATALOG_ALGORITHM_VERSION.to_be_bytes(),
    );
    hash_string(&mut hasher, &fingerprint);
    let control_generation_sha256 = control.generation_sha256();
    hash_string(&mut hasher, &control_generation_sha256);
    let session_ids = hash_session_catalog(store, connection, &mut hasher)?
        .into_iter()
        .filter(|id| control.allows_session(id))
        .collect::<Vec<_>>();
    for session_id in &session_ids {
        store.verify_indexed_session_source_projection_with_control(
            connection, session_id, control,
        )?;
    }
    let mut expected = Vec::new();
    let mut events = connection.prepare("SELECT event_id,session_id,sequence,role,content,content_sha256 FROM events ORDER BY session_id,sequence,event_id")
        .map_err(|error| store.database_error(error))?;
    let rows = events
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })
        .map_err(|error| store.database_error(error))?;
    for row in rows {
        let (event_id, session_id, sequence, role, content, event_hash) =
            row.map_err(|error| store.database_error(error))?;
        if !control.allows_event(&session_id, &event_id) {
            continue;
        }
        if role == "system" || content.trim().is_empty() {
            continue;
        }
        if content_sha256(&content) != event_hash {
            return Err(RetrievalError::CorruptIndex(format!(
                "事件 {event_id} 内容哈希损坏"
            )));
        }
        let event = StoredEvent {
            id: event_id.clone(),
            session_id: session_id.clone(),
            turn_id: None,
            sequence: i64_to_usize(sequence)
                .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?,
            role: EventRole::User,
            created_at: String::new(),
            content: content.clone(),
            content_sha256: event_hash,
            reply_to_event_id: None,
            token_count: None,
            turn_status: None,
            done_reason: None,
            error: None,
        };
        let message_id = format!("{}:0:{}", event_id, content.chars().count());
        for (granularity, span) in document_spans(&event) {
            let part = slice_chars(&content, &span)?;
            let source = content_sha256(&part);
            let document_id = format!("{}:{}:{}", event_id, span.start_char, span.end_char);
            expected.push((
                document_id,
                session_id.clone(),
                granularity,
                source,
                part,
                event_id.clone(),
                span.start_char,
                span.end_char,
                message_id.clone(),
                sequence,
            ));
        }
    }
    expected.sort_by(|a, b| a.0.cmp(&b.0));
    let all_actual_ids = connection
        .prepare(
            "SELECT r.document_id,e.session_id,r.event_id
         FROM retrieval_documents r LEFT JOIN events e ON e.event_id=r.event_id
         WHERE r.granularity IN ('message','fragment') ORDER BY r.document_id",
        )
        .and_then(|mut statement| {
            statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
        })
        .map_err(|error| store.database_error(error))?;
    let mut actual_ids = Vec::new();
    for (id, session_id, event_id) in all_actual_ids {
        if control.event_is_excluded(&event_id)
            || session_id
                .as_deref()
                .is_some_and(|session_id| !control.allows_session(session_id))
        {
            continue;
        }
        let session_id = session_id.ok_or_else(|| {
            RetrievalError::CorruptIndex(format!("retrieval leaf {id} 缺少权威事件"))
        })?;
        if !control.allows_event(&session_id, &event_id) {
            continue;
        }
        if store.memory_document_is_active(connection, control, &id)? {
            actual_ids.push(id);
        }
    }
    if actual_ids != expected.iter().map(|row| row.0.clone()).collect::<Vec<_>>() {
        return Err(RetrievalError::CorruptIndex(
            "retrieval leaf catalog 集合不规范".into(),
        ));
    }
    let all_memory_ids = connection.prepare("SELECT document_id FROM memory_documents WHERE granularity IN ('message','fragment') ORDER BY document_id")
        .and_then(|mut statement| statement.query_map([], |row| row.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>())
        .map_err(|error| store.database_error(error))?;
    let mut memory_ids = Vec::new();
    for id in all_memory_ids {
        if store.memory_document_is_active(connection, control, &id)? {
            memory_ids.push(id);
        }
    }
    if memory_ids != actual_ids {
        return Err(RetrievalError::CorruptIndex(
            "memory leaf catalog 集合不规范".into(),
        ));
    }
    let mut documents = Vec::with_capacity(expected.len());
    for (id, session, granularity, source, content, event, start, end, message_id, sequence) in
        expected
    {
        let granularity_str = match granularity {
            RetrievalDocumentGranularity::Message => "message",
            RetrievalDocumentGranularity::Fragment => "fragment",
            _ => unreachable!(),
        };
        let count:i64=connection.query_row("SELECT count(*) FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id JOIN retrieval_documents r ON r.document_id=d.document_id JOIN source_spans s ON s.event_id=m.event_id AND s.start_char=m.start_char AND s.end_char=m.end_char WHERE d.document_id=?1 AND d.session_id=?2 AND d.granularity=?3 AND d.source_sha256=?4 AND d.start_sequence=?5 AND d.end_sequence=?5 AND d.member_count=1 AND m.ordinal=0 AND m.event_id=?6 AND m.start_char=?7 AND m.end_char=?8 AND m.content_sha256=?4 AND r.event_id=?6 AND r.start_char=?7 AND r.end_char=?8 AND r.granularity=?3 AND r.content_sha256=?4 AND r.exact_content=?9 AND s.content_sha256=?4", params![id,session,granularity_str,source,sequence,event,usize_to_i64(start).map_err(|e|store.database_error(e))?,usize_to_i64(end).map_err(|e|store.database_error(e))?,content], |r|r.get(0)).map_err(|e|store.database_error(e))?;
        if count != 1 {
            return Err(RetrievalError::CorruptIndex(format!(
                "leaf document {id} provenance 不规范"
            )));
        }
        let reusable_vector = raw_embedding(connection, &id)?.and_then(|row| {
            if embedding_equals(Some(&row), spec, &fingerprint, &source, &row.vector_blob) {
                decode_f32_le(&row.vector_blob, spec.dimensions)
                    .ok()
                    .filter(|v| is_unit_vector(v))
            } else {
                None
            }
        });
        for value in [
            &id,
            &session,
            granularity_str,
            &source,
            &content,
            &event,
            &message_id,
        ] {
            hash_string(&mut hasher, value);
        }
        hash_usize(&mut hasher, start);
        hash_usize(&mut hasher, end);
        hash_field(&mut hasher, &sequence.to_be_bytes());
        documents.push(LeafEmbeddingDocument {
            document_id: id,
            session_id: session,
            granularity,
            source_sha256: source,
            content,
            source_event_id: event,
            start_char: start,
            end_char: end,
            message_document_id: message_id,
            reusable_vector,
        });
    }
    hash_embedding_rows(store, connection, &mut hasher, "'message','fragment'")?;
    Ok(LeafEmbeddingSnapshot {
        catalog_sha256: format!("{:x}", hasher.finalize()),
        control_generation_sha256,
        session_ids,
        documents,
    })
}

pub(crate) fn load_aggregate_embedding_snapshot(
    store: &RetrievalStore,
    connection: &Connection,
    spec: &VectorIndexSpec,
) -> RetrievalResult<AggregateEmbeddingSnapshot> {
    let control = store.replay_control_state_under_guard()?;
    load_aggregate_embedding_snapshot_with_control(store, connection, spec, &control)
}

pub(crate) fn load_aggregate_embedding_snapshot_with_control(
    store: &RetrievalStore,
    connection: &Connection,
    spec: &VectorIndexSpec,
    control: &ControlState,
) -> RetrievalResult<AggregateEmbeddingSnapshot> {
    let fingerprint = embedding_fingerprint(spec)?;
    let mut hasher = Sha256::new();
    hash_string(&mut hasher, "hippocampus.embedding.catalog.aggregate");
    hash_field(
        &mut hasher,
        &EMBEDDING_CATALOG_ALGORITHM_VERSION.to_be_bytes(),
    );
    hash_string(&mut hasher, &fingerprint);
    let control_generation_sha256 = control.generation_sha256();
    hash_string(&mut hasher, &control_generation_sha256);
    let session_ids = hash_session_catalog(store, connection, &mut hasher)?
        .into_iter()
        .filter(|id| control.allows_session(id))
        .collect::<Vec<_>>();
    let mut documents = Vec::new();
    for session_id in &session_ids {
        store.verify_indexed_session_source_projection_with_control(
            connection, session_id, control,
        )?;
        let audit = validate_episode_materialization(
            connection,
            &store.root,
            session_id,
            spec,
            &fingerprint,
            true,
        )?;
        if audit.readiness != AggregateReadiness::Ready {
            return Err(RetrievalError::CorruptIndex(format!(
                "会话 {session_id} 缺少当前 spec 的完整 aggregate materialization"
            )));
        }
        let persisted = load_persisted_aggregate_documents(connection, session_id)?;
        for document in persisted {
            if !store.memory_document_is_active(connection, control, &document.document_id)? {
                continue;
            }
            let granularity = match document.granularity.as_str() {
                "episode" => RetrievalDocumentGranularity::Episode,
                "session" => RetrievalDocumentGranularity::Session,
                _ => return Err(RetrievalError::CorruptIndex("aggregate 粒度无效".into())),
            };
            let mut direct_messages = Vec::with_capacity(document.members.len());
            for member in &document.members {
                if member.span.start_char != 0 || member.span.event_id != member.event_id {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "aggregate {} 包含非直接 message member",
                        document.document_id
                    )));
                }
                let expected_id = format!("{}:0:{}", member.event_id, member.span.end_char);
                if member.document_id != expected_id {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "aggregate {} 包含 fragment/间接 member",
                        document.document_id
                    )));
                }
                let row = raw_embedding(connection, &member.document_id)?.ok_or_else(|| {
                    RetrievalError::CorruptIndex(format!(
                        "direct message {} 缺少向量",
                        member.document_id
                    ))
                })?;
                if !embedding_equals(
                    Some(&row),
                    spec,
                    &fingerprint,
                    &member.content_sha256,
                    &row.vector_blob,
                ) {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "direct message {} 向量不兼容",
                        member.document_id
                    )));
                }
                let vector = decode_f32_le(&row.vector_blob, spec.dimensions).map_err(|e| {
                    RetrievalError::CorruptIndex(format!(
                        "direct message {} 向量损坏：{e}",
                        member.document_id
                    ))
                })?;
                if !is_unit_vector(&vector) {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "direct message {} 不是单位向量",
                        member.document_id
                    )));
                }
                direct_messages.push(DirectMessageEmbedding {
                    message_document_id: member.document_id.clone(),
                    source_event_id: member.event_id.clone(),
                    source_sha256: member.content_sha256.clone(),
                    start_char: member.span.start_char,
                    end_char: member.span.end_char,
                    vector,
                });
            }
            for value in [
                &document.document_id,
                &document.session_id,
                &document.granularity,
                &document.source_sha256,
            ] {
                hash_string(&mut hasher, value);
            }
            hash_usize(&mut hasher, direct_messages.len());
            for member in &direct_messages {
                for value in [
                    &member.message_document_id,
                    &member.source_event_id,
                    &member.source_sha256,
                ] {
                    hash_string(&mut hasher, value);
                }
                hash_usize(&mut hasher, member.start_char);
                hash_usize(&mut hasher, member.end_char);
                hash_field(
                    &mut hasher,
                    &encode_f32_le(&member.vector)
                        .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?,
                );
            }
            documents.push(AggregateEmbeddingDocument {
                document_id: document.document_id,
                session_id: document.session_id,
                granularity,
                source_sha256: document.source_sha256,
                direct_messages,
            });
        }
    }
    documents.sort_by(|a, b| a.document_id.cmp(&b.document_id));
    hash_aggregate_freshness(store, connection, &mut hasher)?;
    hash_embedding_rows(store, connection, &mut hasher, "'message'")?;
    hash_embedding_rows(store, connection, &mut hasher, "'episode','session'")?;
    Ok(AggregateEmbeddingSnapshot {
        catalog_sha256: format!("{:x}", hasher.finalize()),
        control_generation_sha256,
        documents,
    })
}

fn hash_aggregate_freshness(
    store: &RetrievalStore,
    connection: &Connection,
    hasher: &mut Sha256,
) -> RetrievalResult<()> {
    let mut statement=connection.prepare("SELECT session_id,source_session_sha256,ledger_snapshot_sha256,vector_index_fingerprint,plan_input_sha256,algorithm_version,gap_minutes,topic_similarity_threshold,episode_count,boundary_count,materialized_at FROM memory_episode_materializations ORDER BY session_id")
        .map_err(|error| store.database_error(error))?;
    let rows = statement
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, i64>(6)?,
                r.get::<_, f64>(7)?,
                r.get::<_, i64>(8)?,
                r.get::<_, i64>(9)?,
                r.get::<_, String>(10)?,
            ))
        })
        .map_err(|error| store.database_error(error))?;
    for row in rows {
        let (a, b, c, d, e, f, g, h, i, j, k) = row.map_err(|error| store.database_error(error))?;
        for v in [&a, &b, &c, &d, &e, &k] {
            hash_string(hasher, v);
        }
        for v in [f, g, i, j] {
            hash_field(hasher, &v.to_be_bytes());
        }
        hash_field(hasher, &h.to_bits().to_be_bytes());
    }
    let mut statement=connection.prepare("SELECT session_id,before_event_id,decision_json,input_sha256 FROM memory_episode_boundaries ORDER BY session_id,before_event_id")
        .map_err(|e|RetrievalError::CorruptIndex(format!("读取 boundary CAS state 失败：{e}")))?;
    let rows = statement
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| RetrievalError::CorruptIndex(format!("读取 boundary CAS state 失败：{e}")))?;
    for row in rows {
        let (a, b, c, d) = row.map_err(|e| {
            RetrievalError::CorruptIndex(format!("读取 boundary CAS state 失败：{e}"))
        })?;
        for v in [&a, &b, &c, &d] {
            hash_string(hasher, v);
        }
    }
    Ok(())
}

fn canonical_aggregate_blobs_from_snapshot(
    snapshot: &AggregateEmbeddingSnapshot,
    dimensions: usize,
) -> RetrievalResult<HashMap<String, Vec<u8>>> {
    let mut blobs = HashMap::with_capacity(snapshot.documents.len());
    for document in &snapshot.documents {
        let vectors = document
            .direct_messages
            .iter()
            .map(|m| (m.message_document_id.clone(), m.vector.clone()))
            .collect::<HashMap<_, _>>();
        let members = document
            .direct_messages
            .iter()
            .enumerate()
            .map(|(sequence, m)| EpisodeMember {
                document_id: m.message_document_id.clone(),
                event_id: m.source_event_id.clone(),
                sequence: sequence as u64,
                role: EventRole::User,
                span: SourceSpan {
                    event_id: m.source_event_id.clone(),
                    start_char: m.start_char,
                    end_char: m.end_char,
                },
                content_sha256: m.source_sha256.clone(),
            })
            .collect::<Vec<_>>();
        let vector =
            canonical_aggregate_vector(&members, &vectors, dimensions, &document.document_id)?;
        let blob =
            encode_f32_le(&vector).map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        blobs.insert(document.document_id.clone(), blob);
    }
    Ok(blobs)
}

fn aggregate_corruption(message: impl Into<String>) -> RetrievalError {
    RetrievalError::CorruptIndex(message.into())
}

fn validate_episode_materialization(
    connection: &Connection,
    root: &Path,
    expected_session: &str,
    spec: &VectorIndexSpec,
    fingerprint: &str,
    require_complete_message_embeddings: bool,
) -> RetrievalResult<AggregateSessionAudit> {
    let materialization = connection
        .query_row(
            "SELECT m.source_session_sha256, m.ledger_snapshot_sha256,
                    m.vector_index_fingerprint, m.plan_input_sha256,
                    m.algorithm_version, m.gap_minutes, m.topic_similarity_threshold,
                    m.episode_count, m.boundary_count, m.materialized_at,
                    s.source_sha256, s.source_file
             FROM memory_episode_materializations m
             JOIN indexed_sessions s ON s.session_id=m.session_id
             WHERE m.session_id=?1",
            [expected_session],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, f64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, String>(10)?,
                    row.get::<_, String>(11)?,
                ))
            },
        )
        .optional()
        .map_err(|error| {
            aggregate_corruption(format!("读取 aggregate materialization 失败：{error}"))
        })?;
    let Some((
        materialized_source,
        stored_ledger_snapshot,
        materialized_fingerprint,
        stored_plan_input,
        algorithm_version,
        gap_minutes,
        topic_similarity_threshold,
        episode_count,
        boundary_count,
        materialized_at,
        indexed_source,
        source_file,
    )) = materialization
    else {
        return Ok(AggregateSessionAudit::stale());
    };

    for (label, hash) in [
        ("materialization source", materialized_source.as_str()),
        ("ledger snapshot", stored_ledger_snapshot.as_str()),
        ("vector fingerprint", materialized_fingerprint.as_str()),
        ("plan input", stored_plan_input.as_str()),
        ("indexed source", indexed_source.as_str()),
    ] {
        if !is_sha256_hex(hash) {
            return Err(aggregate_corruption(format!(
                "会话 {expected_session} 的 {label} 哈希损坏"
            )));
        }
    }
    let expected_algorithm = i64::from(EPISODE_ALGORITHM_VERSION);
    if algorithm_version != expected_algorithm {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的 episode algorithm version 无效"
        )));
    }
    let gap_minutes = i64_to_u64(gap_minutes)
        .map_err(|error| aggregate_corruption(format!("episode gap minutes 损坏：{error}")))?;
    if !(1..=1_440).contains(&gap_minutes) {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的 episode gap minutes 无效"
        )));
    }
    if !topic_similarity_threshold.is_finite()
        || topic_similarity_threshold.to_bits() != EMBEDDING_COSINE_SIMILARITY_THRESHOLD.to_bits()
    {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的 episode topic threshold 无效"
        )));
    }
    let episode_count = i64_to_usize(episode_count)
        .map_err(|error| aggregate_corruption(format!("episode count 损坏：{error}")))?;
    let boundary_count = i64_to_usize(boundary_count)
        .map_err(|error| aggregate_corruption(format!("boundary count 损坏：{error}")))?;
    DateTime::parse_from_rfc3339(&materialized_at).map_err(|error| {
        aggregate_corruption(format!(
            "会话 {expected_session} 的 materialized_at 损坏：{error}"
        ))
    })?;
    let expected_source_file = format!("{expected_session}.json");
    if !is_safe_source_file(&source_file) || source_file != expected_source_file {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的源文件名没有绑定规范会话文件"
        )));
    }

    let mut persisted_documents = load_persisted_aggregate_documents(connection, expected_session)?;
    persisted_documents.sort_by(|left, right| left.document_id.cmp(&right.document_id));
    let actual_episode_count = persisted_documents
        .iter()
        .filter(|document| document.granularity == "episode")
        .count();
    let actual_session_count = persisted_documents
        .iter()
        .filter(|document| document.granularity == "session")
        .count();
    if actual_episode_count != episode_count
        || actual_session_count != usize::from(episode_count > 0)
    {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的 aggregate document count 不匹配"
        )));
    }
    let persisted_boundaries = load_persisted_episode_boundaries(connection, expected_session)?;
    if persisted_boundaries.len() != boundary_count {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的 boundary count 不匹配"
        )));
    }

    let raw_source_stale = validate_aggregate_raw_source(
        connection,
        root,
        expected_session,
        &source_file,
        &indexed_source,
    )? == AggregateReadiness::Stale;
    if materialized_source != indexed_source || materialized_fingerprint != fingerprint {
        return Ok(AggregateSessionAudit::stale());
    }

    let (messages, watermark, suggestions, current_ledger_snapshot) =
        load_episode_snapshot(connection, expected_session, spec, fingerprint)?;
    let complete_message_coverage = messages.iter().all(|message| message.embedding.is_some());
    let direct_message_embeddings = messages
        .iter()
        .filter_map(|message| {
            message
                .embedding
                .as_ref()
                .map(|embedding| (message.member.document_id.clone(), embedding.clone()))
        })
        .collect::<HashMap<_, _>>();
    let plan = plan_episodes(&EpisodePlanInput {
        session_id: expected_session.to_owned(),
        source_session_sha256: indexed_source,
        gap_minutes,
        consolidation_watermark: watermark,
        messages,
        suggestions,
    })
    .map_err(|error| aggregate_corruption(format!("重算 episode plan 失败：{error}")))?;
    if require_complete_message_embeddings && !complete_message_coverage {
        return Ok(AggregateSessionAudit::stale());
    }
    if current_ledger_snapshot != stored_ledger_snapshot
        || plan.plan_input_sha256 != stored_plan_input
        || plan.source_session_sha256 != materialized_source
        || plan.episode_documents.len() != episode_count
        || plan.boundary_decisions.len() != boundary_count
    {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的 materialization audit 与重算 plan 不匹配"
        )));
    }

    let mut expected_boundaries = plan.boundary_decisions.clone();
    expected_boundaries.sort_by(|left, right| left.before_event_id.cmp(&right.before_event_id));
    if persisted_boundaries != expected_boundaries {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的 boundary audit 与重算 plan 不匹配"
        )));
    }
    let mut expected_documents = aggregate_documents_for_plan(&plan);
    expected_documents.sort_by(|left, right| left.document_id.cmp(&right.document_id));
    if persisted_documents != expected_documents {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的 aggregate catalog 与重算 plan 不匹配"
        )));
    }
    if raw_source_stale {
        return Ok(AggregateSessionAudit::stale());
    }
    let canonical_vector_blobs = if complete_message_coverage {
        canonical_aggregate_vector_blobs(
            &expected_documents,
            &direct_message_embeddings,
            spec.dimensions,
        )?
    } else {
        HashMap::new()
    };
    Ok(AggregateSessionAudit::ready(canonical_vector_blobs))
}

fn canonical_aggregate_vector_blobs(
    documents: &[EpisodeDocument],
    direct_message_embeddings: &HashMap<String, Vec<f32>>,
    dimensions: usize,
) -> RetrievalResult<HashMap<String, Vec<u8>>> {
    let mut vectors = HashMap::with_capacity(documents.len());
    for document in documents {
        let vector = canonical_aggregate_vector(
            &document.members,
            direct_message_embeddings,
            dimensions,
            &document.document_id,
        )?;
        let bytes = encode_f32_le(&vector).map_err(|error| {
            aggregate_corruption(format!(
                "aggregate document {} 的规范向量无法编码：{error}",
                document.document_id
            ))
        })?;
        if vectors
            .insert(document.document_id.clone(), bytes)
            .is_some()
        {
            return Err(aggregate_corruption(format!(
                "aggregate catalog 包含重复文档 {}",
                document.document_id
            )));
        }
    }
    Ok(vectors)
}

fn canonical_aggregate_vector(
    members: &[EpisodeMember],
    direct_message_embeddings: &HashMap<String, Vec<f32>>,
    dimensions: usize,
    document_id: &str,
) -> RetrievalResult<Vec<f32>> {
    if members.is_empty() {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} 没有直接 message 成员"
        )));
    }
    let mut normalized_sum = vec![0.0_f64; dimensions];
    for member in members {
        let vector = direct_message_embeddings
            .get(&member.document_id)
            .ok_or_else(|| {
                aggregate_corruption(format!(
                    "aggregate document {document_id} 的直接 message {} 缺少兼容向量",
                    member.document_id
                ))
            })?;
        if vector.len() != dimensions {
            return Err(aggregate_corruption(format!(
                "aggregate document {document_id} 的直接 message {} 向量维度不匹配",
                member.document_id
            )));
        }
        let mut norm_squared = 0.0_f64;
        for value in vector {
            if !value.is_finite() {
                return Err(aggregate_corruption(format!(
                    "aggregate document {document_id} 的直接 message {} 向量包含非有限值",
                    member.document_id
                )));
            }
            let value = f64::from(*value);
            norm_squared += value * value;
        }
        if !norm_squared.is_finite() || norm_squared <= 0.0 {
            return Err(aggregate_corruption(format!(
                "aggregate document {document_id} 的直接 message {} 向量范数无效",
                member.document_id
            )));
        }
        let norm = norm_squared.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(aggregate_corruption(format!(
                "aggregate document {document_id} 的直接 message {} 向量范数无效",
                member.document_id
            )));
        }
        for (sum, value) in normalized_sum.iter_mut().zip(vector) {
            *sum += f64::from(*value) / norm;
            if !sum.is_finite() {
                return Err(aggregate_corruption(format!(
                    "aggregate document {document_id} 的规范向量累加溢出"
                )));
            }
        }
    }

    let member_count = members.len() as f64;
    let mean = normalized_sum
        .into_iter()
        .map(|value| value / member_count)
        .collect::<Vec<_>>();
    let mean_norm_squared = mean.iter().map(|value| value * value).sum::<f64>();
    if !mean_norm_squared.is_finite() || mean_norm_squared <= 0.0 {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} 的规范均值向量范数无效"
        )));
    }
    let mean_norm = mean_norm_squared.sqrt();
    if !mean_norm.is_finite() || mean_norm <= 0.0 {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} 的规范均值向量范数无效"
        )));
    }
    mean.into_iter()
        .map(|value| {
            let normalized = (value / mean_norm) as f32;
            if normalized.is_finite() {
                Ok(normalized)
            } else {
                Err(aggregate_corruption(format!(
                    "aggregate document {document_id} 的规范向量包含非有限值"
                )))
            }
        })
        .collect()
}

fn validate_canonical_aggregate_vector_blob(
    document_id: &str,
    actual: &[u8],
    canonical_vector_blobs: &HashMap<String, Vec<u8>>,
) -> RetrievalResult<()> {
    let expected = canonical_vector_blobs.get(document_id).ok_or_else(|| {
        aggregate_corruption(format!("aggregate document {document_id} 缺少规范向量"))
    })?;
    if actual != expected {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} 的向量不是直接 message 向量的规范聚合"
        )));
    }
    Ok(())
}

fn validate_existing_canonical_aggregate_embeddings(
    connection: &Connection,
    session_id: &str,
    spec: &VectorIndexSpec,
    fingerprint: &str,
    canonical_vector_blobs: &HashMap<String, Vec<u8>>,
) -> RetrievalResult<()> {
    let dimensions = usize_to_i64(spec.dimensions)
        .map_err(|error| aggregate_corruption(format!("aggregate 向量维度损坏：{error}")))?;
    let mut statement = connection
        .prepare(
            "SELECT d.document_id, e.vector_blob
             FROM memory_documents d
             JOIN memory_embeddings e ON e.document_id=d.document_id
             WHERE d.session_id=?1 AND d.granularity IN ('episode','session')
               AND e.model=?2 AND e.dimensions=?3 AND e.index_fingerprint=?4
               AND e.source_sha256=d.source_sha256
             ORDER BY d.document_id",
        )
        .map_err(|error| {
            aggregate_corruption(format!(
                "读取会话 {session_id} 的现有 aggregate 向量失败：{error}"
            ))
        })?;
    let rows = statement
        .query_map(
            params![session_id, spec.model, dimensions, fingerprint],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .map_err(|error| {
            aggregate_corruption(format!(
                "读取会话 {session_id} 的现有 aggregate 向量失败：{error}"
            ))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            aggregate_corruption(format!(
                "读取会话 {session_id} 的现有 aggregate 向量失败：{error}"
            ))
        })?;
    for (document_id, vector_blob) in rows {
        validate_canonical_aggregate_vector_blob(
            &document_id,
            &vector_blob,
            canonical_vector_blobs,
        )?;
    }
    Ok(())
}

fn validate_aggregate_document_source(
    connection: &Connection,
    document_id: &str,
    expected_session: &str,
    expected_source: &str,
) -> RetrievalResult<()> {
    let stored = connection
        .query_row(
            "SELECT session_id, granularity, source_sha256
             FROM memory_documents WHERE document_id=?1",
            [document_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|error| {
            aggregate_corruption(format!(
                "读取 aggregate document {document_id} 失败：{error}"
            ))
        })?
        .ok_or_else(|| aggregate_corruption(format!("缺少 aggregate document {document_id}")))?;
    if stored.0 != expected_session
        || !matches!(stored.1.as_str(), "episode" | "session")
        || stored.2 != expected_source
    {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} 与预期来源不匹配"
        )));
    }
    Ok(())
}

fn validate_aggregate_raw_source(
    connection: &Connection,
    root: &Path,
    expected_session: &str,
    source_file: &str,
    indexed_source_sha256: &str,
) -> RetrievalResult<AggregateReadiness> {
    let path = root.join(source_file);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(AggregateReadiness::Stale),
    };
    if bytes_sha256(&bytes) != indexed_source_sha256 {
        return Ok(AggregateReadiness::Stale);
    }
    let mut session: Session = serde_json::from_slice(&bytes).map_err(|error| {
        aggregate_corruption(format!(
            "会话 {expected_session} 的规范原文件 JSON 损坏：{error}"
        ))
    })?;
    session.normalize_legacy_provenance();
    session.validate().map_err(|error| {
        aggregate_corruption(format!(
            "会话 {expected_session} 的规范原文件语义损坏：{error}"
        ))
    })?;
    if session.id != expected_session
        || path.file_stem().and_then(|value| value.to_str()) != Some(session.id.as_str())
    {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的规范原文件内嵌会话 ID 不匹配"
        )));
    }
    session.refresh_cumulative_usage();
    let expected_events = derive_events(&session);
    let mut statement = connection
        .prepare(
            "SELECT event_id, session_id, turn_id, sequence, role, created_at, content,
                    content_sha256, reply_to_event_id, token_count, turn_status,
                    done_reason, error
             FROM events WHERE session_id=?1 ORDER BY sequence, event_id",
        )
        .map_err(|error| {
            aggregate_corruption(format!(
                "读取会话 {expected_session} 的索引事件投影失败：{error}"
            ))
        })?;
    let indexed_events = statement
        .query_map([expected_session], map_event)
        .map_err(|error| {
            aggregate_corruption(format!(
                "读取会话 {expected_session} 的索引事件投影失败：{error}"
            ))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            aggregate_corruption(format!(
                "读取会话 {expected_session} 的索引事件投影失败：{error}"
            ))
        })?;
    if indexed_events != expected_events {
        return Err(aggregate_corruption(format!(
            "会话 {expected_session} 的规范原文件事件投影与索引不匹配"
        )));
    }
    Ok(AggregateReadiness::Ready)
}

fn load_validated_consolidation_watermark(
    connection: &Connection,
    session_id: &str,
) -> RetrievalResult<Option<u64>> {
    let stored = connection
        .query_row(
            "SELECT through_sequence, through_event_id, through_event_sha256, updated_at
             FROM consolidation_watermarks WHERE session_id=?1",
            [session_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            },
        )
        .optional()
        .map_err(|error| aggregate_corruption(format!("读取巩固水位失败：{error}")))?;
    let Some((through_sequence, through_event_id, through_event_sha256, updated_at)) = stored
    else {
        return Ok(None);
    };
    let through_sequence = i64_to_u64(through_sequence)
        .map_err(|error| aggregate_corruption(format!("巩固水位序号损坏：{error}")))?;
    if through_sequence == 0 {
        return Err(aggregate_corruption(format!(
            "会话 {session_id} 的零巩固水位必须由缺失记录表示"
        )));
    }
    let through_event_id = through_event_id
        .ok_or_else(|| aggregate_corruption(format!("会话 {session_id} 的巩固水位缺少事件 ID")))?;
    let through_event_sha256 = through_event_sha256
        .ok_or_else(|| aggregate_corruption(format!("会话 {session_id} 的巩固水位缺少事件哈希")))?;
    let updated_at = updated_at
        .ok_or_else(|| aggregate_corruption(format!("会话 {session_id} 的巩固水位缺少更新时间")))?;
    if !is_sha256_hex(&through_event_sha256) {
        return Err(aggregate_corruption(format!(
            "会话 {session_id} 的巩固水位事件哈希损坏"
        )));
    }
    DateTime::parse_from_rfc3339(&updated_at).map_err(|error| {
        aggregate_corruption(format!("会话 {session_id} 的巩固水位更新时间损坏：{error}"))
    })?;

    let source = connection
        .query_row(
            "SELECT event_id, turn_id, role, content, content_sha256
             FROM events WHERE session_id=?1 AND sequence=?2",
            params![
                session_id,
                i64::try_from(through_sequence).map_err(|_| {
                    aggregate_corruption(format!(
                        "会话 {session_id} 的巩固水位序号超出 SQLite INTEGER"
                    ))
                })?
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()
        .map_err(|error| aggregate_corruption(format!("读取巩固水位事件失败：{error}")))?
        .ok_or_else(|| {
            aggregate_corruption(format!("会话 {session_id} 的巩固水位序号找不到原始事件"))
        })?;
    if source.0 != through_event_id
        || !matches!(source.2.as_str(), "user" | "assistant")
        || content_sha256(&source.3) != source.4
        || source.4 != through_event_sha256
    {
        return Err(aggregate_corruption(format!(
            "会话 {session_id} 的巩固水位事件来源不匹配"
        )));
    }
    let next_turn_id = connection
        .query_row(
            "SELECT turn_id FROM events WHERE session_id=?1 AND sequence>?2
             ORDER BY sequence LIMIT 1",
            params![
                session_id,
                i64::try_from(through_sequence).map_err(|_| {
                    aggregate_corruption(format!(
                        "会话 {session_id} 的巩固水位序号超出 SQLite INTEGER"
                    ))
                })?
            ],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()
        .map_err(|error| aggregate_corruption(format!("读取巩固水位后继事件失败：{error}")))?
        .flatten();
    if next_turn_id.is_some() && next_turn_id == source.1 {
        return Err(aggregate_corruption(format!(
            "会话 {session_id} 的巩固水位落在轮次内部"
        )));
    }

    let through_sequence_sql = i64::try_from(through_sequence).map_err(|_| {
        aggregate_corruption(format!(
            "会话 {session_id} 的巩固水位序号超出 SQLite INTEGER"
        ))
    })?;
    let applied = connection
        .prepare(
            "SELECT through_sequence, input_event_ids, input_event_hashes, completed_at
             FROM consolidation_batches WHERE session_id=?1 AND status='applied'
               AND projection_schema_version=4 AND through_sequence<=?2
             ORDER BY through_sequence DESC, attempt_id",
        )
        .and_then(|mut statement| {
            statement
                .query_map(params![session_id, through_sequence_sql], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .map_err(|error| aggregate_corruption(format!("读取 applied 巩固批次失败：{error}")))?;
    if applied.is_empty() {
        return Ok(Some(through_sequence));
    }
    let matching_latest = applied
        .iter()
        .filter(|row| row.0 == through_sequence_sql)
        .collect::<Vec<_>>();
    if applied.first().map(|row| row.0) != Some(through_sequence_sql) || matching_latest.len() != 1
    {
        return Err(aggregate_corruption(format!(
            "会话 {session_id} 的巩固水位未唯一绑定最新 applied 批次"
        )));
    }
    let latest = matching_latest[0];
    let event_ids: Vec<String> = serde_json::from_str(&latest.1).map_err(|error| {
        aggregate_corruption(format!(
            "会话 {session_id} 的 applied 事件 ID 损坏：{error}"
        ))
    })?;
    let event_hashes: Vec<String> = serde_json::from_str(&latest.2).map_err(|error| {
        aggregate_corruption(format!(
            "会话 {session_id} 的 applied 事件哈希损坏：{error}"
        ))
    })?;
    if event_ids.last() != Some(&through_event_id)
        || event_hashes.last() != Some(&through_event_sha256)
        || event_ids.len() != event_hashes.len()
        || latest.3 != updated_at
    {
        return Err(aggregate_corruption(format!(
            "会话 {session_id} 的巩固水位字段与最新 applied 批次不匹配"
        )));
    }
    Ok(Some(through_sequence))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn load_persisted_episode_boundaries(
    connection: &Connection,
    session_id: &str,
) -> RetrievalResult<Vec<EpisodeBoundaryDecision>> {
    let mut statement = connection
        .prepare(
            "SELECT before_event_id, decision_json, input_sha256
             FROM memory_episode_boundaries WHERE session_id=?1 ORDER BY before_event_id",
        )
        .map_err(|error| aggregate_corruption(format!("读取 boundary audit 失败：{error}")))?;
    let rows = statement
        .query_map([session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|error| aggregate_corruption(format!("读取 boundary audit 失败：{error}")))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| aggregate_corruption(format!("读取 boundary audit 失败：{error}")))?;
    let mut decisions = Vec::with_capacity(rows.len());
    for (before_event_id, decision_json, input_sha256) in rows {
        let decision: EpisodeBoundaryDecision =
            serde_json::from_str(&decision_json).map_err(|error| {
                aggregate_corruption(format!("boundary {before_event_id} JSON 损坏：{error}"))
            })?;
        let canonical_json = serde_json::to_string(&decision).map_err(|error| {
            aggregate_corruption(format!(
                "boundary {before_event_id} JSON 无法规范化：{error}"
            ))
        })?;
        let event_session = connection
            .query_row(
                "SELECT session_id FROM events WHERE event_id=?1",
                [&before_event_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|error| aggregate_corruption(format!("读取 boundary event 失败：{error}")))?;
        if decision.before_event_id != before_event_id
            || decision.input_sha256 != input_sha256
            || !is_sha256_hex(&input_sha256)
            || canonical_json != decision_json
            || event_session.as_deref() != Some(session_id)
        {
            return Err(aggregate_corruption(format!(
                "boundary {before_event_id} audit 损坏"
            )));
        }
        decisions.push(decision);
    }
    Ok(decisions)
}

fn load_persisted_aggregate_documents(
    connection: &Connection,
    session_id: &str,
) -> RetrievalResult<Vec<EpisodeDocument>> {
    let mut statement = connection
        .prepare(
            "SELECT document_id FROM memory_documents
             WHERE session_id=?1 AND granularity IN ('episode','session') ORDER BY document_id",
        )
        .map_err(|error| aggregate_corruption(format!("读取 aggregate catalog 失败：{error}")))?;
    let ids = statement
        .query_map([session_id], |row| row.get::<_, String>(0))
        .map_err(|error| aggregate_corruption(format!("读取 aggregate catalog 失败：{error}")))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| aggregate_corruption(format!("读取 aggregate catalog 失败：{error}")))?;
    ids.iter()
        .map(|document_id| load_persisted_aggregate_document(connection, document_id, session_id))
        .collect()
}

fn load_persisted_aggregate_document(
    connection: &Connection,
    document_id: &str,
    expected_session: &str,
) -> RetrievalResult<EpisodeDocument> {
    let (session_id, granularity, source_sha256, start_sequence, end_sequence, member_count) =
        connection
            .query_row(
                "SELECT session_id, granularity, source_sha256, start_sequence, end_sequence, member_count
                 FROM memory_documents WHERE document_id=?1",
                [document_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                },
            )
            .map_err(|error| aggregate_corruption(format!("读取 aggregate document 失败：{error}")))?;
    if session_id != expected_session
        || !matches!(granularity.as_str(), "episode" | "session")
        || !is_sha256_hex(&source_sha256)
    {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} 元数据损坏"
        )));
    }
    let start_sequence = i64_to_u64(start_sequence)
        .map_err(|error| aggregate_corruption(format!("aggregate start 损坏：{error}")))?;
    let end_sequence = i64_to_u64(end_sequence)
        .map_err(|error| aggregate_corruption(format!("aggregate end 损坏：{error}")))?;
    let member_count = i64_to_usize(member_count)
        .map_err(|error| aggregate_corruption(format!("aggregate count 损坏：{error}")))?;
    if member_count == 0 || end_sequence < start_sequence {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} range 或 count 损坏"
        )));
    }

    let mut statement = connection
        .prepare(
            "SELECT m.ordinal, m.event_id, m.start_char, m.end_char, m.content_sha256,
                    e.session_id, e.sequence, e.role, e.content, e.content_sha256
             FROM memory_document_members m JOIN events e ON e.event_id=m.event_id
             WHERE m.document_id=?1 ORDER BY m.ordinal",
        )
        .map_err(|error| aggregate_corruption(format!("读取 aggregate members 失败：{error}")))?;
    let rows = statement
        .query_map([document_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
            ))
        })
        .map_err(|error| aggregate_corruption(format!("读取 aggregate members 失败：{error}")))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| aggregate_corruption(format!("读取 aggregate members 失败：{error}")))?;
    if rows.len() != member_count {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} member_count 不匹配"
        )));
    }
    let mut members = Vec::with_capacity(rows.len());
    let mut previous_sequence = None;
    let mut seen_events = HashSet::new();
    for (expected_ordinal, row) in rows.into_iter().enumerate() {
        let (
            ordinal,
            event_id,
            start_char,
            end_char,
            member_hash,
            event_session,
            sequence,
            role,
            content,
            event_hash,
        ) = row;
        let sequence = i64_to_u64(sequence)
            .map_err(|error| aggregate_corruption(format!("member sequence 损坏：{error}")))?;
        let end_char = i64_to_usize(end_char)
            .map_err(|error| aggregate_corruption(format!("member end 损坏：{error}")))?;
        let actual_hash = content_sha256(&content);
        if ordinal
            != usize_to_i64(expected_ordinal)
                .map_err(|error| aggregate_corruption(error.to_string()))?
            || start_char != 0
            || event_session != expected_session
            || !matches!(role.as_str(), "user" | "assistant")
            || end_char != content.chars().count()
            || member_hash != actual_hash
            || event_hash != actual_hash
            || previous_sequence.is_some_and(|previous| sequence <= previous)
            || !seen_events.insert(event_id.clone())
        {
            return Err(aggregate_corruption(format!(
                "aggregate document {document_id} member provenance 损坏"
            )));
        }
        let span_count = connection
            .query_row(
                "SELECT count(*) FROM source_spans
                 WHERE event_id=?1 AND start_char=0 AND end_char=?2 AND content_sha256=?3",
                params![
                    event_id,
                    usize_to_i64(end_char)
                        .map_err(|error| aggregate_corruption(error.to_string()))?,
                    actual_hash
                ],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|error| aggregate_corruption(format!("读取 member span 失败：{error}")))?;
        if span_count != 1 {
            return Err(aggregate_corruption(format!(
                "aggregate document {document_id} member span 损坏"
            )));
        }
        let canonical_id = format!("{event_id}:0:{end_char}");
        let canonical = connection
            .query_row(
                "SELECT session_id,granularity,source_sha256,start_sequence,end_sequence,member_count
                 FROM memory_documents WHERE document_id=?1",
                [&canonical_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| {
                aggregate_corruption(format!("读取 canonical message document 失败：{error}"))
            })?;
        let sequence_i64 = i64::try_from(sequence)
            .map_err(|_| aggregate_corruption("canonical message sequence 溢出"))?;
        if canonical
            != Some((
                expected_session.to_owned(),
                "message".to_owned(),
                actual_hash.clone(),
                sequence_i64,
                sequence_i64,
                1,
            ))
        {
            return Err(aggregate_corruption(format!(
                "canonical message document {canonical_id} 损坏"
            )));
        }
        let canonical_member = connection
            .query_row(
                "SELECT count(*),min(ordinal),max(ordinal),min(event_id),max(event_id),
                        min(start_char),max(start_char),min(end_char),max(end_char),
                        min(content_sha256),max(content_sha256)
                 FROM memory_document_members WHERE document_id=?1",
                [&canonical_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Option<i64>>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<i64>>(5)?,
                        row.get::<_, Option<i64>>(6)?,
                        row.get::<_, Option<i64>>(7)?,
                        row.get::<_, Option<i64>>(8)?,
                        row.get::<_, Option<String>>(9)?,
                        row.get::<_, Option<String>>(10)?,
                    ))
                },
            )
            .map_err(|error| {
                aggregate_corruption(format!("读取 canonical message member 失败：{error}"))
            })?;
        let end_i64 =
            usize_to_i64(end_char).map_err(|error| aggregate_corruption(error.to_string()))?;
        if canonical_member
            != (
                1,
                Some(0),
                Some(0),
                Some(event_id.clone()),
                Some(event_id.clone()),
                Some(0),
                Some(0),
                Some(end_i64),
                Some(end_i64),
                Some(actual_hash.clone()),
                Some(actual_hash.clone()),
            )
        {
            return Err(aggregate_corruption(format!(
                "canonical message member {canonical_id} 损坏"
            )));
        }
        previous_sequence = Some(sequence);
        members.push(EpisodeMember {
            document_id: canonical_id,
            event_id: event_id.clone(),
            sequence,
            role: if role == "user" {
                EventRole::User
            } else {
                EventRole::Assistant
            },
            span: SourceSpan {
                event_id,
                start_char: 0,
                end_char,
            },
            content_sha256: actual_hash,
        });
    }
    if members.first().map(|member| member.sequence) != Some(start_sequence)
        || members.last().map(|member| member.sequence) != Some(end_sequence)
        || aggregate_members_hash(&granularity, expected_session, &members) != source_sha256
    {
        return Err(aggregate_corruption(format!(
            "aggregate document {document_id} range 或 source hash 不匹配"
        )));
    }
    Ok(EpisodeDocument {
        document_id: document_id.to_owned(),
        session_id,
        granularity,
        source_sha256,
        start_sequence,
        end_sequence,
        members,
    })
}

fn verify_embedding_writeback(
    connection: &Connection,
    prepared: &PreparedEmbedding,
    spec: &VectorIndexSpec,
    fingerprint: &str,
    embedded_at: &str,
) -> RetrievalResult<()> {
    let stored = connection
        .query_row(
            "SELECT document_id, model, dimensions, source_sha256, index_fingerprint, vector_blob, embedded_at
             FROM memory_embeddings WHERE document_id=?1",
            [&prepared.document_id],
            |row| Ok((
                row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?, row.get::<_, String>(4)?, row.get::<_, Vec<u8>>(5)?,
                row.get::<_, String>(6)?,
            )),
        )
        .optional()
        .map_err(|error| aggregate_corruption(format!("回读 embedding 失败：{error}")))?
        .ok_or_else(|| aggregate_corruption(format!("写入后缺少 embedding {}", prepared.document_id)))?;
    let expected_dimensions =
        usize_to_i64(spec.dimensions).map_err(|error| aggregate_corruption(error.to_string()))?;
    if stored.0 != prepared.document_id
        || stored.1 != spec.model
        || stored.2 != expected_dimensions
        || stored.3 != prepared.source_sha256
        || stored.4 != fingerprint
        || stored.5 != prepared.vector_blob
        || stored.6 != embedded_at
    {
        return Err(aggregate_corruption(format!(
            "embedding {} 写入后不匹配",
            prepared.document_id
        )));
    }
    let decoded = decode_f32_le(&stored.5, spec.dimensions).map_err(|error| {
        aggregate_corruption(format!(
            "embedding {} 写入后损坏：{error}",
            prepared.document_id
        ))
    })?;
    if encode_f32_le(&decoded).map_err(|error| aggregate_corruption(error.to_string()))?
        != prepared.vector_blob
    {
        return Err(aggregate_corruption(format!(
            "embedding {} 写入后位表示不匹配",
            prepared.document_id
        )));
    }
    Ok(())
}

fn parse_memory_granularity(value: &str) -> RetrievalResult<RetrievalDocumentGranularity> {
    match value {
        "message" => Ok(RetrievalDocumentGranularity::Message),
        "fragment" => Ok(RetrievalDocumentGranularity::Fragment),
        "episode" => Ok(RetrievalDocumentGranularity::Episode),
        "session" => Ok(RetrievalDocumentGranularity::Session),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知记忆文档粒度 {value}"
        ))),
    }
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
    untrusted_history_wrapped: bool,
    knowledge_trace: KnowledgeTrace,
}

pub(crate) fn derive_events(session: &Session) -> Vec<StoredEvent> {
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

fn canonical_exact_context_items(
    session: &Session,
    turn: &Turn,
) -> Result<Vec<ContextItemTrace>, String> {
    if turn.context_trace.untrusted_history_wrapped
        != !turn.context_trace.retrieval.selected_evidence.is_empty()
    {
        return Err(format!("回答 {} 的不可信历史标记与检索证据不一致", turn.id));
    }
    let current_index = session
        .turns
        .iter()
        .position(|candidate| candidate.id == turn.id)
        .ok_or_else(|| format!("回答 {} 不属于来源会话", turn.id))?;
    let mut items = Vec::new();
    if !session.system_prompt.is_empty() {
        items.push(ContextItemTrace {
            role: EventRole::System,
            span: SourceSpan {
                event_id: event_id(&session.id, None, EventRole::System),
                start_char: 0,
                end_char: session.system_prompt.chars().count(),
            },
            content_sha256: content_sha256(&session.system_prompt),
        });
    }
    if turn.context_trace.untrusted_history_wrapped {
        items.extend(
            turn.context_trace
                .retrieval
                .selected_evidence
                .iter()
                .map(|selected| ContextItemTrace {
                    role: EventRole::System,
                    span: selected.span.clone(),
                    content_sha256: selected.content_sha256.clone(),
                }),
        );
    }
    let mut included = HashSet::new();
    let mut previous_index = None;
    for included_turn_id in &turn.context_trace.included_turn_ids {
        if !included.insert(included_turn_id.as_str()) {
            return Err(format!("回答 {} 包含重复历史轮次", turn.id));
        }
        let index = session
            .turns
            .iter()
            .position(|candidate| candidate.id == *included_turn_id)
            .ok_or_else(|| format!("回答 {} 引用了非本会话历史轮次", turn.id))?;
        if index >= current_index || previous_index.is_some_and(|previous| index <= previous) {
            return Err(format!("回答 {} 的历史轮次顺序无效", turn.id));
        }
        previous_index = Some(index);
        let included_turn = &session.turns[index];
        items.push(full_turn_item(session, included_turn, EventRole::User));
        if has_assistant_event(session, included_turn) {
            items.push(full_turn_item(session, included_turn, EventRole::Assistant));
        }
    }
    items.push(full_turn_item(session, turn, EventRole::User));
    Ok(items)
}

fn full_turn_item(session: &Session, turn: &Turn, role: EventRole) -> ContextItemTrace {
    let content = match role {
        EventRole::User => &turn.user_content,
        EventRole::Assistant => &turn.assistant_content,
        EventRole::System => unreachable!("turn items are user or assistant"),
    };
    ContextItemTrace {
        role,
        span: SourceSpan {
            event_id: event_id(&session.id, Some(&turn.id), role),
            start_char: 0,
            end_char: content.chars().count(),
        },
        content_sha256: content_sha256(content),
    }
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
        let canonical_items = canonical_exact_context_items(session, turn).map_err(|message| {
            RetrievalError::InvalidSource {
                path: source_path.to_path_buf(),
                message,
            }
        })?;
        if turn.context_trace.context_items != canonical_items {
            return Err(RetrievalError::InvalidSource {
                path: source_path.to_path_buf(),
                message: format!("回答 {} 的精确上下文序列不规范", turn.id),
            });
        }
        return Ok(DerivedContext {
            items: canonical_items,
            context_sha256: context_hash,
            provenance_quality: ProvenanceQuality::Exact,
            request: Some(request),
            identity_instruction: turn.context_trace.identity_instruction.clone(),
            untrusted_history_wrapped: turn.context_trace.untrusted_history_wrapped,
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
        untrusted_history_wrapped: false,
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

pub(crate) fn parse_status(value: &str) -> rusqlite::Result<TurnStatus> {
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

fn read_existing_memory_state_version(connection: &Connection) -> rusqlite::Result<Option<i64>> {
    let exists = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master
                       WHERE type = 'table' AND name = 'memory_schema_meta')",
        [],
        |row| row.get::<_, bool>(0),
    )?;
    if !exists {
        return Ok(None);
    }
    connection
        .query_row(
            "SELECT value FROM memory_schema_meta WHERE key = 'state_schema_version'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()
}

fn read_existing_graph_schema_version(connection: &Connection) -> rusqlite::Result<Option<i64>> {
    let exists = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master
                       WHERE type = 'table' AND name = 'memory_schema_meta')",
        [],
        |row| row.get::<_, bool>(0),
    )?;
    if !exists {
        return Ok(None);
    }
    connection
        .query_row(
            "SELECT value FROM memory_schema_meta WHERE key = 'graph_schema_version'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()
}

fn prepare_graph_schema(
    connection: &Connection,
    existing_version: Option<i64>,
) -> rusqlite::Result<()> {
    if existing_version.is_none() {
        connection.execute_batch(
            "DELETE FROM memory_graph_edges;
             DELETE FROM memory_graph_nodes;
             DELETE FROM memory_graph_materializations;",
        )?;
    }
    connection.execute(
        "INSERT INTO memory_schema_meta(key,value) VALUES('graph_schema_version',1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [],
    )?;
    Ok(())
}

fn prepare_memory_state_schema(
    connection: &Connection,
    existing_version: Option<i64>,
) -> rusqlite::Result<()> {
    if !table_has_column(
        connection,
        "consolidation_batches",
        "projection_schema_version",
    )? {
        connection.execute_batch(
            "ALTER TABLE consolidation_batches ADD COLUMN projection_schema_version INTEGER
             CHECK(projection_schema_version IS NULL OR projection_schema_version = 4);",
        )?;
    }
    if existing_version != Some(MEMORY_STATE_SCHEMA_VERSION) {
        // Pre-v3 rows use different deterministic-ID, evidence, or transition-order contracts.
        // They cannot be upgraded row by row without trusting the old derived state, so discard
        // only the replayable projection and retain raw events plus the immutable attempt ledger.
        connection.execute_batch(
            "DELETE FROM memory_graph_edges;
             DELETE FROM memory_graph_nodes;
             DELETE FROM memory_graph_materializations;
             DELETE FROM memory_schema_meta WHERE key='graph_schema_version';
             DELETE FROM memory_embeddings
               WHERE document_id IN (SELECT document_id FROM memory_documents
                                     WHERE granularity IN ('episode','session'));
             DELETE FROM memory_episode_materializations;
             DELETE FROM memory_episode_boundaries;
             DELETE FROM memory_document_members
               WHERE document_id IN (SELECT document_id FROM memory_documents
                                     WHERE granularity IN ('episode','session'));
             DELETE FROM memory_documents WHERE granularity IN ('episode','session');
             DELETE FROM memory_entity_mentions;
             DELETE FROM memory_episode_boundaries;
             DELETE FROM memory_episode_materializations;
             DELETE FROM memory_claim_evidence;
             DELETE FROM memory_claim_transitions;
             DELETE FROM memory_boundary_suggestions;
             DELETE FROM memory_claims;
             DELETE FROM memory_entity_aliases;
             DELETE FROM memory_entities;
             DELETE FROM consolidation_watermarks;
             DROP TABLE memory_claim_evidence;
             DROP TABLE memory_claim_transitions;
             DROP TABLE memory_boundary_suggestions;
             DROP TABLE memory_claims;
             DROP TABLE memory_entity_aliases;
             DROP TABLE memory_entities;
             DROP TABLE consolidation_watermarks;",
        )?;
        connection.execute_batch(SCHEMA_SQL)?;
    }
    connection.execute(
        "INSERT INTO memory_schema_meta(key, value)
         VALUES ('state_schema_version', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [MEMORY_STATE_SCHEMA_VERSION],
    )?;
    Ok(())
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

-- The memory catalog is derived only.  It intentionally remains separate
-- from FTS because episode/session documents have provenance members but no
-- generated text to index.
CREATE TABLE IF NOT EXISTS memory_documents (
    document_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES indexed_sessions(session_id) ON DELETE CASCADE,
    granularity TEXT NOT NULL CHECK(granularity IN ('message', 'fragment', 'episode', 'session')),
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    start_sequence INTEGER NOT NULL CHECK(start_sequence >= 0),
    end_sequence INTEGER NOT NULL CHECK(end_sequence >= start_sequence),
    member_count INTEGER NOT NULL CHECK(member_count > 0)
);

CREATE TABLE IF NOT EXISTS memory_document_members (
    document_id TEXT NOT NULL REFERENCES memory_documents(document_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    event_id TEXT NOT NULL,
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char >= start_char),
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
    PRIMARY KEY(document_id, ordinal),
    FOREIGN KEY(event_id, start_char, end_char)
        REFERENCES source_spans(event_id, start_char, end_char)
);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    document_id TEXT PRIMARY KEY REFERENCES memory_documents(document_id) ON DELETE CASCADE,
    model TEXT NOT NULL CHECK(length(model) > 0),
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256) = 64),
    index_fingerprint TEXT NOT NULL CHECK(length(index_fingerprint) = 64),
    vector_blob BLOB NOT NULL CHECK(length(vector_blob) = dimensions * 4),
    embedded_at TEXT NOT NULL CHECK(length(embedded_at) > 0)
);

-- A provenance source is immutable but rebuild and corruption recovery may
-- delete it.  Remove the entire derived document before the source span goes
-- away, so an aggregate can never retain a partial member list or embedding.
CREATE TRIGGER IF NOT EXISTS memory_documents_before_source_span_delete
BEFORE DELETE ON source_spans
BEGIN
    DELETE FROM memory_episode_materializations
    WHERE session_id = (SELECT session_id FROM events WHERE event_id = OLD.event_id);
    DELETE FROM memory_documents
    WHERE document_id IN (
        SELECT document_id FROM memory_document_members
        WHERE event_id = OLD.event_id
          AND start_char = OLD.start_char
          AND end_char = OLD.end_char
    );
END;

CREATE INDEX IF NOT EXISTS memory_documents_session_granularity
    ON memory_documents(session_id, granularity, document_id);

CREATE TABLE IF NOT EXISTS memory_episode_boundaries (
    session_id TEXT NOT NULL REFERENCES indexed_sessions(session_id) ON DELETE CASCADE,
    before_event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    decision_json TEXT NOT NULL CHECK(length(decision_json) > 0),
    input_sha256 TEXT NOT NULL CHECK(length(input_sha256) = 64),
    PRIMARY KEY(session_id, before_event_id)
);

CREATE TABLE IF NOT EXISTS memory_episode_materializations (
    session_id TEXT PRIMARY KEY REFERENCES indexed_sessions(session_id) ON DELETE CASCADE,
    source_session_sha256 TEXT NOT NULL CHECK(length(source_session_sha256) = 64),
    ledger_snapshot_sha256 TEXT NOT NULL CHECK(length(ledger_snapshot_sha256) = 64),
    vector_index_fingerprint TEXT NOT NULL CHECK(length(vector_index_fingerprint) = 64),
    plan_input_sha256 TEXT NOT NULL CHECK(length(plan_input_sha256) = 64),
    algorithm_version INTEGER NOT NULL CHECK(algorithm_version = 1),
    gap_minutes INTEGER NOT NULL CHECK(gap_minutes >= 0),
    topic_similarity_threshold REAL NOT NULL CHECK(topic_similarity_threshold = 0.60),
    episode_count INTEGER NOT NULL CHECK(episode_count >= 0),
    boundary_count INTEGER NOT NULL CHECK(boundary_count >= 0),
    materialized_at TEXT NOT NULL
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
    projection_schema_version INTEGER CHECK(projection_schema_version IS NULL OR projection_schema_version = 4),
    CHECK(from_sequence <= through_sequence),
    CHECK((response_json IS NULL AND response_sha256 IS NULL)
       OR (response_json IS NOT NULL AND response_sha256 IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS consolidation_batches_session_started
    ON consolidation_batches(session_id, started_at, attempt_id);
CREATE INDEX IF NOT EXISTS consolidation_batches_batch_key
    ON consolidation_batches(batch_key);
CREATE TRIGGER IF NOT EXISTS consolidation_batches_immutable_update
BEFORE UPDATE ON consolidation_batches
BEGIN SELECT RAISE(ABORT, 'consolidation_batches is immutable'); END;
CREATE TRIGGER IF NOT EXISTS consolidation_batches_immutable_delete
BEFORE DELETE ON consolidation_batches
BEGIN SELECT RAISE(ABORT, 'consolidation_batches is immutable'); END;

CREATE TABLE IF NOT EXISTS memory_schema_meta (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entities (
    entity_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('person','organization','location','object','concept','unknown')),
    canonical_name TEXT NOT NULL CHECK(length(canonical_name) > 0),
    normalized_name TEXT NOT NULL CHECK(length(normalized_name) > 0),
    disambiguation TEXT NOT NULL CHECK(disambiguation IN ('resolved','pending')),
    created_session_id TEXT NOT NULL CHECK(length(created_session_id) > 0),
    created_batch_key TEXT NOT NULL CHECK(length(created_batch_key) > 0),
    created_event_id TEXT NOT NULL CHECK(length(created_event_id) > 0),
    created_start INTEGER NOT NULL CHECK(created_start >= 0),
    created_end INTEGER NOT NULL CHECK(created_end > created_start),
    created_hash TEXT NOT NULL CHECK(length(created_hash) = 64),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_entities_normalized
    ON memory_entities(normalized_name, kind, entity_id);

CREATE TABLE IF NOT EXISTS memory_entity_mentions (
    mention_id TEXT PRIMARY KEY CHECK(mention_id GLOB 'mention_*' AND length(mention_id) = 72
        AND substr(mention_id, 9) NOT GLOB '*[^0-9a-f]*'),
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    mention_kind TEXT NOT NULL CHECK(mention_kind IN ('entity_name','alias','claim_subject','claim_object')),
    source_record_id TEXT NOT NULL CHECK(length(source_record_id) > 0),
    entity_id TEXT NOT NULL CHECK(length(entity_id) > 0),
    entity_status TEXT NOT NULL CHECK(entity_status IN ('resolved','pending')),
    event_id TEXT NOT NULL CHECK(length(event_id) > 0),
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char > start_char),
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64
        AND content_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE(batch_key, mention_kind, source_record_id, entity_id, event_id, start_char, end_char),
    FOREIGN KEY(entity_id) REFERENCES memory_entities(entity_id)
);
CREATE INDEX IF NOT EXISTS memory_entity_mentions_session_event_status_entity
    ON memory_entity_mentions(session_id, event_id, entity_status, entity_id, mention_id);
CREATE INDEX IF NOT EXISTS memory_entity_mentions_entity_status_session_event
    ON memory_entity_mentions(entity_id, entity_status, session_id, event_id, mention_id);
CREATE INDEX IF NOT EXISTS memory_entity_mentions_batch
    ON memory_entity_mentions(batch_key, mention_id);

CREATE TABLE IF NOT EXISTS memory_entity_aliases (
    alias_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    alias_text TEXT NOT NULL CHECK(length(alias_text) > 0),
    normalized_alias TEXT NOT NULL CHECK(length(normalized_alias) > 0),
    alias_kind TEXT NOT NULL CHECK(alias_kind IN ('explicit_alias','stable_identifier')),
    stable_identifier_kind TEXT,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    event_id TEXT NOT NULL CHECK(length(event_id) > 0),
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char > start_char),
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
    proof_event_id TEXT NOT NULL CHECK(length(proof_event_id) > 0),
    proof_start_char INTEGER NOT NULL CHECK(proof_start_char >= 0),
    proof_end_char INTEGER NOT NULL CHECK(proof_end_char > proof_start_char),
    proof_sha256 TEXT NOT NULL CHECK(length(proof_sha256) = 64),
    identity_event_id TEXT NOT NULL CHECK(length(identity_event_id) > 0),
    identity_start_char INTEGER NOT NULL CHECK(identity_start_char >= 0),
    identity_end_char INTEGER NOT NULL CHECK(identity_end_char > identity_start_char),
    identity_sha256 TEXT NOT NULL CHECK(length(identity_sha256) = 64),
    created_at TEXT NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES memory_entities(entity_id),
    CHECK((alias_kind = 'explicit_alias' AND stable_identifier_kind IS NULL)
       OR (alias_kind = 'stable_identifier' AND stable_identifier_kind IS NOT NULL
           AND length(stable_identifier_kind) > 0))
);
CREATE INDEX IF NOT EXISTS memory_entity_aliases_entity
    ON memory_entity_aliases(entity_id, alias_id);
CREATE INDEX IF NOT EXISTS memory_entity_aliases_normalized
    ON memory_entity_aliases(alias_kind, stable_identifier_kind, normalized_alias, entity_id);

CREATE TABLE IF NOT EXISTS memory_claims (
    claim_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    subject_entity_id TEXT NOT NULL CHECK(length(subject_entity_id) > 0),
    predicate_key TEXT NOT NULL CHECK(length(predicate_key) > 0),
    normalized_relation TEXT NOT NULL CHECK(length(normalized_relation) > 0),
    object_kind TEXT NOT NULL CHECK(object_kind IN ('text','entity')),
    object_text TEXT,
    object_entity_id TEXT,
    normalized_object TEXT NOT NULL CHECK(length(normalized_object) > 0),
    polarity TEXT NOT NULL CHECK(polarity IN ('assert','deny')),
    cardinality TEXT NOT NULL CHECK(cardinality IN ('single','multi')),
    certainty TEXT NOT NULL CHECK(certainty IN ('certain','uncertain')),
    state TEXT NOT NULL CHECK(state IN ('active','superseded','conflicted','uncertain')),
    asserted_at TEXT NOT NULL,
    event_time TEXT,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    reference_time TEXT NOT NULL,
    created_batch_key TEXT NOT NULL CHECK(length(created_batch_key) > 0),
    updated_batch_key TEXT NOT NULL CHECK(length(updated_batch_key) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK((object_kind = 'text' AND object_text IS NOT NULL AND object_entity_id IS NULL)
       OR (object_kind = 'entity' AND object_text IS NULL AND object_entity_id IS NOT NULL)),
    FOREIGN KEY(subject_entity_id) REFERENCES memory_entities(entity_id),
    FOREIGN KEY(object_entity_id) REFERENCES memory_entities(entity_id)
);
CREATE INDEX IF NOT EXISTS memory_claims_subject_predicate
    ON memory_claims(subject_entity_id, predicate_key, state, claim_id);
CREATE INDEX IF NOT EXISTS memory_claims_updated
    ON memory_claims(updated_at DESC, claim_id);

CREATE TABLE IF NOT EXISTS memory_claim_evidence (
    evidence_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    event_id TEXT NOT NULL CHECK(length(event_id) > 0),
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    kind TEXT NOT NULL CHECK(kind IN ('assertion','user_confirmation','correction','temporal')),
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char > start_char),
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
    subject_start_char INTEGER NOT NULL CHECK(subject_start_char >= start_char),
    subject_end_char INTEGER NOT NULL CHECK(subject_end_char > subject_start_char AND subject_end_char <= end_char),
    subject_sha256 TEXT NOT NULL CHECK(length(subject_sha256) = 64),
    relation_start_char INTEGER NOT NULL CHECK(relation_start_char >= start_char),
    relation_end_char INTEGER NOT NULL CHECK(relation_end_char > relation_start_char AND relation_end_char <= end_char),
    relation_sha256 TEXT NOT NULL CHECK(length(relation_sha256) = 64),
    object_start_char INTEGER NOT NULL CHECK(object_start_char >= start_char),
    object_end_char INTEGER NOT NULL CHECK(object_end_char > object_start_char AND object_end_char <= end_char),
    object_sha256 TEXT NOT NULL CHECK(length(object_sha256) = 64),
    speech_act_event_id TEXT,
    speech_act_start_char INTEGER,
    speech_act_end_char INTEGER,
    speech_act_sha256 TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(claim_id) REFERENCES memory_claims(claim_id),
    CHECK((speech_act_event_id IS NULL AND speech_act_start_char IS NULL
           AND speech_act_end_char IS NULL AND speech_act_sha256 IS NULL)
       OR (speech_act_event_id IS NOT NULL AND speech_act_start_char IS NOT NULL
           AND speech_act_end_char IS NOT NULL AND speech_act_sha256 IS NOT NULL
           AND speech_act_start_char >= start_char
           AND speech_act_end_char > speech_act_start_char
           AND speech_act_end_char <= end_char
           AND speech_act_event_id = event_id
           AND length(speech_act_sha256) = 64))
);
CREATE INDEX IF NOT EXISTS memory_claim_evidence_claim
    ON memory_claim_evidence(claim_id, event_id, start_char, end_char, evidence_id);
CREATE INDEX IF NOT EXISTS memory_claim_evidence_event
    ON memory_claim_evidence(event_id, claim_id);

CREATE TABLE IF NOT EXISTS memory_claim_transitions (
    transition_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    from_state TEXT CHECK(from_state IS NULL OR from_state IN ('active','superseded','conflicted','uncertain')),
    to_state TEXT NOT NULL CHECK(to_state IN ('active','superseded','conflicted','uncertain')),
    reason TEXT NOT NULL CHECK(reason IN ('created','confirmed','certainty_upgraded','conflicted','corrected','replaced')),
    related_claim_id TEXT,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    created_at TEXT NOT NULL,
    FOREIGN KEY(claim_id) REFERENCES memory_claims(claim_id),
    FOREIGN KEY(related_claim_id) REFERENCES memory_claims(claim_id),
    UNIQUE(claim_id, ordinal)
);
CREATE INDEX IF NOT EXISTS memory_claim_transitions_claim
    ON memory_claim_transitions(claim_id, ordinal);

CREATE TABLE IF NOT EXISTS memory_boundary_suggestions (
    boundary_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    before_event_id TEXT NOT NULL CHECK(length(before_event_id) > 0),
    reason TEXT NOT NULL CHECK(reason IN ('explicit_topic_transition','model_topic_shift')),
    evidence_json TEXT NOT NULL CHECK(length(evidence_json) > 0),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_boundary_suggestions_session_event
    ON memory_boundary_suggestions(session_id, before_event_id, boundary_id);

CREATE TABLE IF NOT EXISTS memory_graph_nodes (
    node_id TEXT PRIMARY KEY,
    node_kind TEXT NOT NULL CHECK(node_kind IN ('document','entity','claim')),
    source_id TEXT NOT NULL,
    session_id TEXT,
    granularity TEXT CHECK(granularity IS NULL OR granularity IN ('message','fragment','episode','session')),
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256)=64 AND source_sha256 NOT GLOB '*[^0-9a-f]*'),
    UNIQUE(node_kind,source_id),
    CHECK((node_kind='document' AND session_id IS NOT NULL AND granularity IS NOT NULL)
       OR (node_kind='entity' AND session_id IS NULL AND granularity IS NULL)
       OR (node_kind='claim' AND session_id IS NOT NULL AND granularity IS NULL))
);
CREATE INDEX IF NOT EXISTS memory_graph_nodes_kind_source
    ON memory_graph_nodes(node_kind,source_id);

CREATE TABLE IF NOT EXISTS memory_graph_edges (
    edge_id TEXT PRIMARY KEY,
    edge_type TEXT NOT NULL CHECK(edge_type IN ('reply','adjacent','episode_member','entity_mention','shared_entity','keyword_cooccurrence','embedding_mutual_top_k','common_recall','support','conflict','replacement')),
    source_node_id TEXT NOT NULL REFERENCES memory_graph_nodes(node_id) ON DELETE CASCADE,
    target_node_id TEXT NOT NULL REFERENCES memory_graph_nodes(node_id) ON DELETE CASCADE,
    weight REAL NOT NULL CHECK(weight>0),
    directed INTEGER NOT NULL CHECK(directed IN (0,1)),
    provenance_json TEXT NOT NULL CHECK(length(provenance_json)>0),
    provenance_sha256 TEXT NOT NULL CHECK(length(provenance_sha256)=64 AND provenance_sha256 NOT GLOB '*[^0-9a-f]*'),
    CHECK(source_node_id<>target_node_id),
    CHECK((edge_type='replacement' AND directed=1) OR (edge_type<>'replacement' AND directed=0)),
    CHECK(directed=1 OR source_node_id<target_node_id),
    UNIQUE(edge_type,source_node_id,target_node_id)
);
CREATE INDEX IF NOT EXISTS memory_graph_edges_source_type
    ON memory_graph_edges(source_node_id,edge_type,target_node_id);
CREATE INDEX IF NOT EXISTS memory_graph_edges_target_type
    ON memory_graph_edges(target_node_id,edge_type,source_node_id);

CREATE TABLE IF NOT EXISTS memory_graph_materializations (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    algorithm_version INTEGER NOT NULL CHECK(algorithm_version=1),
    vector_index_fingerprint TEXT NOT NULL CHECK(length(vector_index_fingerprint)=64),
    config_sha256 TEXT NOT NULL CHECK(length(config_sha256)=64),
    source_sha256 TEXT NOT NULL CHECK(length(source_sha256)=64),
    catalog_sha256 TEXT NOT NULL CHECK(length(catalog_sha256)=64),
    node_count INTEGER NOT NULL CHECK(node_count>=0),
    edge_count INTEGER NOT NULL CHECK(edge_count>=0),
    materialized_at TEXT NOT NULL CHECK(length(materialized_at)>0)
);

CREATE TRIGGER IF NOT EXISTS graph_invalidate_indexed_sessions_insert AFTER INSERT ON indexed_sessions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_indexed_sessions_update AFTER UPDATE ON indexed_sessions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_indexed_sessions_delete AFTER DELETE ON indexed_sessions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_events_insert AFTER INSERT ON events BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_events_update AFTER UPDATE ON events BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_events_delete AFTER DELETE ON events BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_source_spans_insert AFTER INSERT ON source_spans BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_source_spans_update AFTER UPDATE ON source_spans BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_source_spans_delete AFTER DELETE ON source_spans BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_retrieval_runs_insert AFTER INSERT ON retrieval_runs BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_retrieval_runs_update AFTER UPDATE ON retrieval_runs BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_retrieval_runs_delete AFTER DELETE ON retrieval_runs BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_retrieval_documents_insert AFTER INSERT ON retrieval_documents BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_retrieval_documents_update AFTER UPDATE ON retrieval_documents BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_retrieval_documents_delete AFTER DELETE ON retrieval_documents BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_documents_insert AFTER INSERT ON memory_documents BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_documents_update AFTER UPDATE ON memory_documents BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_documents_delete AFTER DELETE ON memory_documents BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_document_members_insert AFTER INSERT ON memory_document_members BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_document_members_update AFTER UPDATE ON memory_document_members BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_document_members_delete AFTER DELETE ON memory_document_members BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_embeddings_insert AFTER INSERT ON memory_embeddings BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_embeddings_update AFTER UPDATE ON memory_embeddings BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_embeddings_delete AFTER DELETE ON memory_embeddings BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_episode_boundaries_insert AFTER INSERT ON memory_episode_boundaries BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_episode_boundaries_update AFTER UPDATE ON memory_episode_boundaries BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_episode_boundaries_delete AFTER DELETE ON memory_episode_boundaries BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_episode_materializations_insert AFTER INSERT ON memory_episode_materializations BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_episode_materializations_update AFTER UPDATE ON memory_episode_materializations BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_episode_materializations_delete AFTER DELETE ON memory_episode_materializations BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_consolidation_watermarks_insert AFTER INSERT ON consolidation_watermarks BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_consolidation_watermarks_update AFTER UPDATE ON consolidation_watermarks BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_consolidation_watermarks_delete AFTER DELETE ON consolidation_watermarks BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_consolidation_batches_insert AFTER INSERT ON consolidation_batches BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_consolidation_batches_update AFTER UPDATE ON consolidation_batches BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_consolidation_batches_delete AFTER DELETE ON consolidation_batches BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entities_insert AFTER INSERT ON memory_entities BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entities_update AFTER UPDATE ON memory_entities BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entities_delete AFTER DELETE ON memory_entities BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entity_aliases_insert AFTER INSERT ON memory_entity_aliases BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entity_aliases_update AFTER UPDATE ON memory_entity_aliases BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entity_aliases_delete AFTER DELETE ON memory_entity_aliases BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entity_mentions_insert AFTER INSERT ON memory_entity_mentions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entity_mentions_update AFTER UPDATE ON memory_entity_mentions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_entity_mentions_delete AFTER DELETE ON memory_entity_mentions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claims_insert AFTER INSERT ON memory_claims BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claims_update AFTER UPDATE ON memory_claims BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claims_delete AFTER DELETE ON memory_claims BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claim_evidence_insert AFTER INSERT ON memory_claim_evidence BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claim_evidence_update AFTER UPDATE ON memory_claim_evidence BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claim_evidence_delete AFTER DELETE ON memory_claim_evidence BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claim_transitions_insert AFTER INSERT ON memory_claim_transitions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claim_transitions_update AFTER UPDATE ON memory_claim_transitions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_claim_transitions_delete AFTER DELETE ON memory_claim_transitions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_boundary_suggestions_insert AFTER INSERT ON memory_boundary_suggestions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_boundary_suggestions_update AFTER UPDATE ON memory_boundary_suggestions BEGIN DELETE FROM memory_graph_materializations; END;
CREATE TRIGGER IF NOT EXISTS graph_invalidate_memory_boundary_suggestions_delete AFTER DELETE ON memory_boundary_suggestions BEGIN DELETE FROM memory_graph_materializations; END;
"#;

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

async fn join_blocking<T>(
    handle: tokio::task::JoinHandle<RetrievalResult<T>>,
) -> RetrievalResult<T> {
    join_blocking_result(handle.await)
}

fn join_blocking_result<T>(
    result: std::result::Result<RetrievalResult<T>, tokio::task::JoinError>,
) -> RetrievalResult<T> {
    result.map_err(|error| RetrievalError::CorruptIndex(format!("阻塞检索任务失败：{error}")))?
}

fn validate_query_embedding(
    response: std::result::Result<
        std::result::Result<crate::ollama::EmbeddingResponse, crate::ollama::OllamaError>,
        tokio::time::error::Elapsed,
    >,
    spec: &VectorIndexSpec,
) -> RetrievalResult<Vec<f32>> {
    let response = response
        .map_err(|_| RetrievalError::CorruptIndex("查询 embedding 超时".into()))?
        .map_err(|error| RetrievalError::CorruptIndex(format!("查询 embedding 失败：{error}")))?;
    if response.model != spec.model {
        return Err(RetrievalError::CorruptIndex(format!(
            "embedding 响应模型不匹配：期望 {}，实际 {}",
            spec.model, response.model
        )));
    }
    let [vector] = response.embeddings.as_slice() else {
        return Err(RetrievalError::CorruptIndex(
            "embedding 响应必须恰好包含一个向量".into(),
        ));
    };
    if vector.len() != spec.dimensions {
        return Err(RetrievalError::CorruptIndex(format!(
            "查询 embedding 维度不匹配：期望 {}，实际 {}",
            spec.dimensions,
            vector.len()
        )));
    }
    l2_normalize(vector).map_err(|error| RetrievalError::CorruptIndex(error.to_string()))
}

fn embedding_catalog_identity(rows: &[StoredEmbedding]) -> String {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.document_id.cmp(&right.document_id));
    let mut hasher = Sha256::new();
    hasher.update(b"hippocampus-hybrid-recall-catalog-v1\0");
    for row in ordered {
        for bytes in [
            row.document_id.as_bytes(),
            row.session_id.as_bytes(),
            granularity_name(row.granularity).as_bytes(),
            row.source_sha256.as_bytes(),
            row.model.as_bytes(),
            row.index_fingerprint.as_bytes(),
            row.embedded_at.as_bytes(),
        ] {
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
        hasher.update((row.dimensions as u64).to_le_bytes());
        hasher.update((row.vector.len() as u64).to_le_bytes());
        for value in &row.vector {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn channel_trace(
    channel: RetrievalChannel,
    status: &str,
    candidate_count: usize,
    elapsed_ms: u64,
    error: Option<String>,
) -> ChannelTrace {
    ChannelTrace {
        channel,
        status: status.into(),
        candidate_count,
        elapsed_ms,
        error,
    }
}

fn requested_channel_traces(
    channels: RecallChannels,
    bm25_status: &str,
    bm25_count: usize,
    bm25_ms: u64,
) -> Vec<ChannelTrace> {
    [
        (RetrievalChannel::Bm25, channels.bm25),
        (RetrievalChannel::Vector, channels.vector),
        (RetrievalChannel::Entity, channels.entity),
        (RetrievalChannel::State, channels.state),
        (RetrievalChannel::Episode, channels.episode),
        (RetrievalChannel::Graph, channels.graph),
    ]
    .into_iter()
    .map(|(channel, enabled)| {
        if channel == RetrievalChannel::Bm25 && enabled {
            channel_trace(channel, bm25_status, bm25_count, bm25_ms, None)
        } else {
            channel_trace(channel, if enabled { "ok" } else { "disabled" }, 0, 0, None)
        }
    })
    .collect()
}

fn empty_recall_base(
    raw_query: &str,
    current_user_event_id: &str,
    config: RetrievalConfig,
) -> RecallResult {
    let query_kind = classify_query(raw_query);
    RecallResult {
        trace: RetrievalTrace {
            status: "ok".into(),
            current_query_event_id: current_user_event_id.into(),
            query_terms: query_terms(raw_query),
            config,
            query_kind,
            budget_allocation: BudgetAllocationTrace::for_query_kind(query_kind),
            ..Default::default()
        },
        evidence: Vec::new(),
    }
}

fn empty_state_sidecar() -> StateSidecar {
    StateSidecar {
        entity_matches: Vec::new(),
        candidates: Vec::new(),
        warnings: Vec::new(),
        entity_ms: 0,
        state_ms: 0,
    }
}

fn empty_graph_sidecar() -> GraphSidecar {
    GraphSidecar {
        paths: Vec::new(),
        candidates: Vec::new(),
        candidate_count: 0,
        aggregate_source_count: 0,
        elapsed_ms: 0,
        warning: None,
    }
}

fn memory_budget_for_query(config: &MemoryConfig, kind: QueryKind) -> MemoryBudgetConfig {
    match kind {
        QueryKind::ExactFact => config.budgets.exact_fact,
        QueryKind::GeneralSemantic => config.budgets.general_semantic,
        QueryKind::EventRecap => config.budgets.event_recap,
        QueryKind::TemporalState => config.budgets.temporal_state,
        QueryKind::MultiHop => config.budgets.multi_hop,
    }
}

pub(crate) fn memory_budget_trace(config: &MemoryConfig, kind: QueryKind) -> BudgetAllocationTrace {
    let budget = memory_budget_for_query(config, kind);
    BudgetAllocationTrace {
        query_kind: kind,
        recent_history_percent: budget.recent_history,
        exact_or_state_percent: budget.exact_or_state,
        episode_percent: budget.episode,
        graph_percent: budget.graph,
        ..Default::default()
    }
}

/// Builds the bounded, fully materialized candidate-pool limits used by the
/// engine's exact-token allocator. Public recall callers keep their legacy
/// final-selection limits.
pub(crate) fn candidate_pool_config(config: &RetrievalConfig) -> RetrievalConfig {
    RetrievalConfig {
        candidate_limit: config.candidate_limit,
        max_selected: DEFERRED_HARD_LIMIT,
        evidence_char_budget: DEFERRED_HARD_LIMIT,
        expansion_char_budget: DEFERRED_HARD_LIMIT,
    }
}

fn parse_granularity(value: &str) -> RetrievalResult<RetrievalDocumentGranularity> {
    match value {
        "message" => Ok(RetrievalDocumentGranularity::Message),
        "fragment" => Ok(RetrievalDocumentGranularity::Fragment),
        "episode" => Ok(RetrievalDocumentGranularity::Episode),
        "session" => Ok(RetrievalDocumentGranularity::Session),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "无效文档粒度 {value}"
        ))),
    }
}

fn apply_vector_fallback(
    result: &mut RecallResult,
    bm25_ms: u64,
    vector_ms: u64,
    error: String,
    channels: RecallChannels,
) {
    result.trace.status = "bm25_fallback".into();
    result.trace.channels = vec![
        channel_trace(
            RetrievalChannel::Bm25,
            "ok",
            result.trace.candidates.len(),
            bm25_ms,
            None,
        ),
        channel_trace(
            RetrievalChannel::Vector,
            "error",
            0,
            vector_ms,
            Some(error.clone()),
        ),
        channel_trace(RetrievalChannel::Entity, "skipped", 0, 0, None),
        channel_trace(RetrievalChannel::State, "skipped", 0, 0, None),
        channel_trace(RetrievalChannel::Episode, "skipped", 0, 0, None),
        channel_trace(RetrievalChannel::Graph, "skipped", 0, 0, None),
    ];
    result.trace.warnings.push(error);
    disable_unrequested_channels(&mut result.trace.channels, channels);
}

fn apply_sidecar_fallback(
    result: &mut RecallResult,
    bm25_ms: u64,
    vector_ms: u64,
    error: String,
    channels: RecallChannels,
) {
    result.trace.status = "bm25_fallback".into();
    result.trace.fusion_candidates.clear();
    result.trace.entity_matches.clear();
    result.trace.state_selections.clear();
    result.trace.channels = vec![
        channel_trace(
            RetrievalChannel::Bm25,
            "ok",
            result.trace.candidates.len(),
            bm25_ms,
            None,
        ),
        channel_trace(
            RetrievalChannel::Vector,
            "discarded",
            0,
            vector_ms,
            Some("advanced evidence discarded".into()),
        ),
        channel_trace(RetrievalChannel::Entity, "error", 0, 0, Some(error.clone())),
        channel_trace(RetrievalChannel::State, "error", 0, 0, Some(error.clone())),
        channel_trace(RetrievalChannel::Episode, "skipped", 0, 0, None),
        channel_trace(RetrievalChannel::Graph, "skipped", 0, 0, None),
    ];
    result.trace.warnings.push(error);
    disable_unrequested_channels(&mut result.trace.channels, channels);
}

fn apply_graph_fallback(
    result: &mut RecallResult,
    bm25_ms: u64,
    vector_ms: u64,
    graph_ms: u64,
    error: String,
    channels: RecallChannels,
) {
    result.trace.status = "bm25_fallback".into();
    result.trace.fusion_candidates.clear();
    result.trace.entity_matches.clear();
    result.trace.state_selections.clear();
    result.trace.graph_paths.clear();
    result.trace.channels = vec![
        channel_trace(
            RetrievalChannel::Bm25,
            "ok",
            result.trace.candidates.len(),
            bm25_ms,
            None,
        ),
        channel_trace(
            RetrievalChannel::Vector,
            "discarded",
            0,
            vector_ms,
            Some("advanced evidence discarded".into()),
        ),
        channel_trace(RetrievalChannel::Entity, "discarded", 0, 0, None),
        channel_trace(RetrievalChannel::State, "discarded", 0, 0, None),
        channel_trace(RetrievalChannel::Episode, "discarded", 0, 0, None),
        channel_trace(
            RetrievalChannel::Graph,
            "error",
            0,
            graph_ms,
            Some(error.clone()),
        ),
    ];
    result.trace.warnings.push(error);
    disable_unrequested_channels(&mut result.trace.channels, channels);
}

fn disable_unrequested_channels(traces: &mut [ChannelTrace], channels: RecallChannels) {
    for trace in traces {
        let enabled = match trace.channel {
            RetrievalChannel::Bm25 => channels.bm25,
            RetrievalChannel::Vector => channels.vector,
            RetrievalChannel::Entity => channels.entity,
            RetrievalChannel::State => channels.state,
            RetrievalChannel::Episode => channels.episode,
            RetrievalChannel::Graph => channels.graph,
        };
        if !enabled {
            *trace = channel_trace(trace.channel, "disabled", 0, 0, None);
        }
    }
}

fn is_generic_pronoun(value: &str) -> bool {
    matches!(
        value,
        "我" | "本人"
            | "你"
            | "您"
            | "他"
            | "她"
            | "它"
            | "他们"
            | "她们"
            | "它们"
            | "i"
            | "me"
            | "my"
            | "you"
            | "he"
            | "him"
            | "she"
            | "her"
            | "it"
            | "they"
            | "them"
    )
}

fn query_contains_match(query: &str, term: &str) -> bool {
    if term.trim().is_empty() {
        return false;
    }
    if term.is_ascii() {
        query.match_indices(term).any(|(start, _)| {
            let before = query[..start].chars().next_back();
            let end = start + term.len();
            let after = query[end..].chars().next();
            !before.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
                && !after
                    .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
        })
    } else {
        query.contains(term)
    }
}

fn identity_match_ranges(
    query: &str,
    surface: &str,
    query_tokens: &BTreeMap<String, Vec<(usize, usize)>>,
) -> Vec<(usize, usize)> {
    if surface.trim().is_empty() {
        return Vec::new();
    }
    if surface.is_ascii() {
        return query
            .match_indices(surface)
            .filter_map(|(start, _)| {
                let before = query[..start].chars().next_back();
                let end = start + surface.len();
                let after = query[end..].chars().next();
                (!before
                    .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
                    && !after.is_some_and(|character| {
                        character.is_ascii_alphanumeric() || character == '_'
                    }))
                .then_some((start, end))
            })
            .collect();
    }
    let has_cjk = surface.chars().any(is_cjk);
    if has_cjk && (surface.chars().count() == 1 || is_generic_pronoun(surface)) {
        return query_tokens.get(surface).cloned().unwrap_or_default();
    }
    query
        .match_indices(surface)
        .map(|(start, _)| (start, start + surface.len()))
        .collect()
}

fn identity_token_ranges(query: &str) -> BTreeMap<String, Vec<(usize, usize)>> {
    let mut ranges = BTreeMap::<String, Vec<(usize, usize)>>::new();
    let mut cursor = 0usize;
    for token in jieba().cut(query, false) {
        if token.word.is_empty() {
            continue;
        }
        let Some(relative) = query[cursor..].find(token.word) else {
            continue;
        };
        let start = cursor + relative;
        let end = start + token.word.len();
        let normalized = normalize_match(token.word);
        if !normalized.is_empty() {
            ranges.entry(normalized).or_default().push((start, end));
        }
        cursor = end;
    }
    ranges
}

fn explicit_query_date(query: &str) -> (Option<DateTime<Utc>>, bool) {
    let bytes = query.as_bytes();
    let mut invalid = false;
    let mut first_valid = None;
    for start in 0..bytes.len().saturating_sub(9) {
        let value = &bytes[start..start + 10];
        if !(value[0..4].iter().all(u8::is_ascii_digit)
            && value[4] == b'-'
            && value[5..7].iter().all(u8::is_ascii_digit)
            && value[7] == b'-'
            && value[8..10].iter().all(u8::is_ascii_digit))
        {
            continue;
        }
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let after_ok = start + 10 == bytes.len() || !bytes[start + 10].is_ascii_alphanumeric();
        if !before_ok || !after_ok {
            continue;
        }
        let text = std::str::from_utf8(value).unwrap_or_default();
        if let Ok(date) = NaiveDate::parse_from_str(text, "%Y-%m-%d") {
            let next = date.succ_opt().unwrap_or(date);
            let instant = Utc.from_utc_datetime(&next.and_hms_opt(0, 0, 0).unwrap())
                - chrono::Duration::nanoseconds(1);
            if first_valid.is_none() {
                first_valid = Some(instant);
            }
        } else {
            invalid = true;
        }
    }
    (first_valid, invalid)
}

fn has_historical_cue(query: &str) -> bool {
    [
        "曾经",
        "以前",
        "过去",
        "原来",
        "当时",
        "后来",
        "什么时候",
        "何时",
    ]
    .iter()
    .any(|cue| query.contains(cue))
        || [
            "formerly",
            "previously",
            "before",
            "after",
            "used to",
            "when",
        ]
        .iter()
        .any(|cue| query_contains_match(query, cue))
}

fn parse_retrieval_time(value: &str, label: &str) -> RetrievalResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|time| time.with_timezone(&Utc))
        .map_err(|_| RetrievalError::CorruptIndex(format!("{label} 时间损坏")))
}

fn claim_overlap(terms: &BTreeSet<String>, predicate: &str, relation: &str, object: &str) -> usize {
    let fields = [
        normalize_match(predicate),
        normalize_match(relation),
        normalize_match(object),
    ];
    terms
        .iter()
        .filter(|term| {
            fields
                .iter()
                .any(|field| query_contains_match(field, term) || query_contains_match(term, field))
        })
        .count()
}

fn state_priority(state: &str) -> u8 {
    match state {
        "active" => 0,
        "conflicted" => 1,
        "uncertain" => 2,
        _ => 3,
    }
}

fn complete_conflict_group(
    conflicts: &BTreeMap<String, BTreeSet<String>>,
    claim_id: &str,
) -> Vec<String> {
    let mut found = BTreeSet::from([claim_id.to_owned()]);
    let mut pending = vec![claim_id.to_owned()];
    while let Some(current) = pending.pop() {
        if let Some(neighbors) = conflicts.get(&current) {
            for neighbor in neighbors {
                if found.insert(neighbor.clone()) {
                    pending.push(neighbor.clone());
                }
            }
        }
    }
    found.into_iter().collect()
}

fn granularity_name(granularity: RetrievalDocumentGranularity) -> &'static str {
    match granularity {
        RetrievalDocumentGranularity::Message => "message",
        RetrievalDocumentGranularity::Fragment => "fragment",
        RetrievalDocumentGranularity::Episode => "episode",
        RetrievalDocumentGranularity::Session => "session",
    }
}

fn entity_graph_seeds(matches: &[EntityMatchTrace]) -> Vec<GraphRecallSeed> {
    let mut best = BTreeMap::<String, (u8, String)>::new();
    for matched in matches {
        let Some(entity_id) = matched.selected_entity_id.as_ref() else {
            continue;
        };
        let priority = match matched.match_basis.as_str() {
            "stable_identifier" => 0,
            "explicit_alias" => 1,
            "ent_self" => 2,
            _ => 3,
        };
        let candidate = (priority, matched.normalized_text.clone());
        if best.get(entity_id).is_none_or(|old| candidate < *old) {
            best.insert(entity_id.clone(), candidate);
        }
    }
    let mut ordered = best
        .into_iter()
        .map(|(entity_id, (priority, normalized))| (priority, normalized, entity_id))
        .collect::<Vec<_>>();
    ordered.sort();
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, (_, _, entity_id))| GraphRecallSeed {
            channel: RetrievalChannel::Entity,
            source_id: entity_id,
            document_id: None,
            rank: index + 1,
            score: 1.0 / (index + 1) as f64,
        })
        .collect()
}

fn trace_graph_hash_parts(domain: &str, parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    let domain = domain.as_bytes();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    format!("{:x}", hasher.finalize())
}

fn trace_episode_document_id(session_id: &str, first_event_id: &str) -> String {
    format!(
        "episode_{}",
        trace_graph_hash_parts(
            "hippocampus-episode-document-v1",
            &[session_id.as_bytes(), first_event_id.as_bytes()],
        )
    )
}

fn trace_graph_id_is_valid(value: &str, prefix: &str) -> bool {
    value.strip_prefix(prefix).is_some_and(is_sha256_hex)
}

fn trace_graph_node_id(kind: &str, source_id: &str) -> String {
    format!(
        "gnode_{}",
        trace_graph_hash_parts(
            "hippocampus.graph.node-id.v1",
            &[kind.as_bytes(), source_id.as_bytes()],
        )
    )
}

fn trace_graph_edge_id(edge_type: &str, source: &str, target: &str) -> String {
    format!(
        "gedge_{}",
        trace_graph_hash_parts(
            "hippocampus.graph.edge-id.v1",
            &[edge_type.as_bytes(), source.as_bytes(), target.as_bytes(),],
        )
    )
}

fn trace_graph_edge_id_matches(edge_type: &str, left: &str, right: &str, edge_id: &str) -> bool {
    const EDGE_TYPES: &[&str] = &[
        "reply",
        "adjacent",
        "episode_member",
        "entity_mention",
        "shared_entity",
        "keyword_cooccurrence",
        "embedding_mutual_top_k",
        "common_recall",
        "support",
        "conflict",
        "replacement",
    ];
    if !EDGE_TYPES.contains(&edge_type) || left == right {
        return false;
    }
    if edge_type == "replacement" {
        edge_id == trace_graph_edge_id(edge_type, left, right)
            || edge_id == trace_graph_edge_id(edge_type, right, left)
    } else {
        let (source, target) = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        edge_id == trace_graph_edge_id(edge_type, source, target)
    }
}

fn raw_candidate_key(span: &SourceSpan) -> (String, usize, usize) {
    (span.event_id.clone(), span.start_char, span.end_char)
}

fn rrf(k: usize, rank: usize) -> f64 {
    1.0 / (k + rank) as f64
}

fn exact_cosine_f64(left: &[f32], right: &[f32]) -> RetrievalResult<f64> {
    if left.len() != right.len() {
        return Err(RetrievalError::CorruptIndex(
            "候选向量维度与查询向量不匹配".into(),
        ));
    }
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    let denominator = (left_norm * right_norm).sqrt();
    if !dot.is_finite() || !denominator.is_finite() || denominator == 0.0 {
        return Err(RetrievalError::CorruptIndex(
            "候选向量 cosine 计算失败".into(),
        ));
    }
    Ok((dot / denominator).clamp(-1.0, 1.0))
}

fn mmr_score(
    candidate: &FusedRawCandidate,
    selected: &[usize],
    candidates: &[FusedRawCandidate],
    max_rrf: f64,
) -> f64 {
    let redundancy = candidate.vector.as_ref().map_or(0.0, |vector| {
        selected
            .iter()
            .filter_map(|&index| candidates[index].vector.as_ref())
            .filter_map(|selected_vector| exact_cosine_f64(vector, selected_vector).ok())
            .fold(0.0_f64, f64::max)
    });
    0.75 * (candidate.rrf_score / max_rrf) - 0.25 * redundancy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidation::{
        BoundarySuggestionReason, ClaimCardinality, ClaimCertainty, ClaimDisposition,
        ClaimPolarity, ConsolidatedClaimObject, ConsolidatedClaimOutput, ConsolidatedEntityOutput,
        ConsolidationAttemptRecord, ConsolidationAttemptStatus, ConsolidationBoundaryOutput,
        ConsolidationClaimEvidence, ConsolidationClaimObjectKind, ConsolidationEvent,
        ConsolidationEvidenceKind, ConsolidationQuote, EntityAliasOutput, EntityDisambiguation,
        EntityResolution, EntityResolutionBasis, MemoryAliasKind, MemoryEntityKind,
        StructuredConsolidationOutput, canonical_consolidation_request,
    };
    use crate::context::ContextAssembler;
    use crate::model::{ContextTrace, ModelRequestTrace, TokenUsage, Turn, TurnStatus, utc_now};
    use crate::store::SessionStore;

    fn embedding_catalog_fixture(
        message: &str,
    ) -> (tempfile::TempDir, SessionStore, Session, VectorIndexSpec) {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, message, "回答", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let spec = VectorIndexSpec::from_config(&aggregate_memory_config()).unwrap();
        (root, store, session, spec)
    }

    fn embedding_catalog_two_session_fixture(
        messages: [&str; 2],
    ) -> (
        tempfile::TempDir,
        SessionStore,
        Vec<Session>,
        VectorIndexSpec,
    ) {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut sessions = Vec::new();
        for message in messages {
            let mut session = store
                .create("model", "http://localhost", None, Default::default(), false)
                .unwrap();
            append_complete_turn(&mut session, message, "answer", "");
            let path = store.save(&mut session).unwrap();
            store.retrieval().sync_session(&session, &path).unwrap();
            sessions.push(session);
        }
        let spec = VectorIndexSpec::from_config(&aggregate_memory_config()).unwrap();
        (root, store, sessions, spec)
    }

    fn embedding_catalog_assert_leaf_snapshot_rejects(
        corrupt: impl FnOnce(&Connection, &Session, &LeafEmbeddingSnapshot),
    ) {
        let (_root, store, session, spec) = embedding_catalog_fixture("corruption fixture");
        let snapshot = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        corrupt(&connection, &session, &snapshot);
        assert!(matches!(
            store.retrieval().leaf_embedding_snapshot(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
    }

    fn embedding_catalog_unit_writes(
        snapshot: &LeafEmbeddingSnapshot,
        dimensions: usize,
    ) -> Vec<EmbeddingWrite> {
        snapshot
            .documents
            .iter()
            .enumerate()
            .map(|(index, document)| {
                let mut vector = vec![0.0; dimensions];
                vector[index % dimensions] = 1.0;
                EmbeddingWrite {
                    document_id: document.document_id.clone(),
                    expected_source_sha256: document.source_sha256.clone(),
                    vector,
                }
            })
            .collect()
    }

    #[test]
    fn embedding_catalog_leaf_snapshot_is_owned_complete_unicode_canonical() {
        let message = "界".repeat(441);
        let (_root, store, session, spec) = embedding_catalog_fixture(&message);
        let snapshot = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        assert_eq!(snapshot.session_ids, vec![session.id]);
        let event = snapshot
            .documents
            .iter()
            .find(|d| d.content == message)
            .unwrap()
            .source_event_id
            .clone();
        let docs = snapshot
            .documents
            .iter()
            .filter(|d| d.source_event_id == event)
            .collect::<Vec<_>>();
        assert_eq!(docs.len(), 4);
        assert!(
            docs.iter()
                .any(|d| d.granularity == RetrievalDocumentGranularity::Message
                    && d.start_char == 0
                    && d.end_char == 441
                    && d.message_document_id == d.document_id)
        );
        assert_eq!(
            docs.iter()
                .filter(|d| d.granularity == RetrievalDocumentGranularity::Fragment)
                .map(|d| (d.start_char, d.end_char))
                .collect::<Vec<_>>(),
            vec![(0, 240), (200, 440), (400, 441)]
        );
        drop(store);
        assert_eq!(
            snapshot
                .documents
                .iter()
                .find(|d| d.source_event_id == event && d.start_char == 200)
                .unwrap()
                .content,
            "界".repeat(240)
        );
        assert!(
            snapshot
                .documents
                .windows(2)
                .all(|w| w[0].document_id < w[1].document_id)
        );
    }

    #[test]
    fn embedding_catalog_leaf_snapshot_rejects_catalog_and_member_corruption() {
        embedding_catalog_assert_leaf_snapshot_rejects(|connection, _, snapshot| {
            connection
                .execute(
                    "DELETE FROM retrieval_documents WHERE document_id=?1",
                    [&snapshot.documents[0].document_id],
                )
                .unwrap();
        });
        embedding_catalog_assert_leaf_snapshot_rejects(|connection, _, snapshot| {
            connection
                .execute(
                    "INSERT INTO retrieval_documents
                     (document_id,event_id,start_char,end_char,granularity,content_sha256,
                      exact_content,lexical_content,ngram_content)
                     SELECT 'corrupt-extra-retrieval',event_id,start_char,end_char,granularity,
                            content_sha256,exact_content,lexical_content,ngram_content
                     FROM retrieval_documents WHERE document_id=?1",
                    [&snapshot.documents[0].document_id],
                )
                .unwrap();
        });
        embedding_catalog_assert_leaf_snapshot_rejects(|connection, _, snapshot| {
            connection
                .execute(
                    "DELETE FROM memory_documents WHERE document_id=?1",
                    [&snapshot.documents[0].document_id],
                )
                .unwrap();
        });
        embedding_catalog_assert_leaf_snapshot_rejects(|connection, _, snapshot| {
            connection
                .execute(
                    "INSERT INTO memory_documents
                     (document_id,session_id,granularity,source_sha256,start_sequence,
                      end_sequence,member_count)
                     SELECT 'corrupt-extra-memory',session_id,granularity,source_sha256,
                            start_sequence,end_sequence,member_count
                     FROM memory_documents WHERE document_id=?1",
                    [&snapshot.documents[0].document_id],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO memory_document_members
                     (document_id,ordinal,event_id,start_char,end_char,content_sha256)
                     SELECT 'corrupt-extra-memory',ordinal,event_id,start_char,end_char,content_sha256
                     FROM memory_document_members WHERE document_id=?1",
                    [&snapshot.documents[0].document_id],
                )
                .unwrap();
        });
        embedding_catalog_assert_leaf_snapshot_rejects(|connection, _, snapshot| {
            connection
                .execute(
                    "INSERT INTO memory_document_members
                     (document_id,ordinal,event_id,start_char,end_char,content_sha256)
                     SELECT document_id,1,event_id,start_char,end_char,content_sha256
                     FROM memory_document_members WHERE document_id=?1 AND ordinal=0",
                    [&snapshot.documents[0].document_id],
                )
                .unwrap();
        });

        let (_root, store, sessions, spec) =
            embedding_catalog_two_session_fixture(["first session", "second session"]);
        let snapshot = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let target = snapshot
            .documents
            .iter()
            .find(|document| document.session_id == sessions[0].id)
            .unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "UPDATE memory_documents SET session_id=?1 WHERE document_id=?2",
                params![sessions[1].id, target.document_id],
            )
            .unwrap();
        assert!(matches!(
            store.retrieval().leaf_embedding_snapshot(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
    }

    #[test]
    fn embedding_catalog_leaf_complete_publish_noop_and_stale_cas() {
        let (_root, store, session, spec) = embedding_catalog_fixture("hello catalog");
        let snapshot = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let writes = embedding_catalog_unit_writes(&snapshot, spec.dimensions);
        assert!(
            store
                .retrieval()
                .publish_leaf_embedding_catalog(&spec, &snapshot, &writes[..writes.len() - 1])
                .is_err()
        );
        let report = store
            .retrieval()
            .publish_leaf_embedding_catalog(&spec, &snapshot, &writes)
            .unwrap();
        assert!(report.changed);
        assert_eq!(report.documents, writes.len());
        let fresh = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let before = episode_sql_rows(
            &connection,
            "SELECT e.document_id,e.model,e.dimensions,e.source_sha256,e.index_fingerprint,e.vector_blob,e.embedded_at FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id WHERE d.session_id=?1 ORDER BY e.document_id",
            &session.id,
        );
        let noop = store
            .retrieval()
            .publish_leaf_embedding_catalog(&spec, &fresh, &writes)
            .unwrap();
        assert!(!noop.changed);
        assert_eq!(noop.reused, writes.len());
        let after = episode_sql_rows(
            &connection,
            "SELECT e.document_id,e.model,e.dimensions,e.source_sha256,e.index_fingerprint,e.vector_blob,e.embedded_at FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id WHERE d.session_id=?1 ORDER BY e.document_id",
            &session.id,
        );
        assert_eq!(before, after);
        let mut tampered_vector = vec![0.0; spec.dimensions];
        tampered_vector[spec.dimensions - 1] = 1.0;
        let tampered_blob = encode_f32_le(&tampered_vector).unwrap();
        connection
            .execute(
                "UPDATE memory_embeddings SET vector_blob=?1,embedded_at='tampered'
                 WHERE document_id=?2",
                params![tampered_blob, writes[0].document_id],
            )
            .unwrap();
        assert!(matches!(
            store
                .retrieval()
                .publish_leaf_embedding_catalog(&spec, &fresh, &writes),
            Err(RetrievalError::EmbeddingCatalogStale { .. })
        ));
        let retained = connection
            .query_row(
                "SELECT vector_blob,embedded_at FROM memory_embeddings WHERE document_id=?1",
                [&writes[0].document_id],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, String>(1)?)),
            )
            .unwrap();
        assert_eq!(retained, (tampered_blob, "tampered".into()));
    }

    #[test]
    fn embedding_catalog_aggregate_snapshot_and_atomic_complete_publish() {
        let (_root, store, session, spec) = embedding_catalog_fixture("aggregate source");
        let leaf = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let leaf_writes = embedding_catalog_unit_writes(&leaf, spec.dimensions);
        store
            .retrieval()
            .publish_leaf_embedding_catalog(&spec, &leaf, &leaf_writes)
            .unwrap();
        store
            .retrieval()
            .materialize_episode_documents(&session.id, &aggregate_memory_config())
            .unwrap();
        let snapshot = store
            .retrieval()
            .aggregate_embedding_snapshot(&spec)
            .unwrap();
        assert!(!snapshot.documents.is_empty());
        assert!(
            snapshot
                .documents
                .iter()
                .all(|d| !d.direct_messages.is_empty())
        );
        let blobs = canonical_aggregate_blobs_from_snapshot(&snapshot, spec.dimensions).unwrap();
        let writes = snapshot
            .documents
            .iter()
            .map(|d| EmbeddingWrite {
                document_id: d.document_id.clone(),
                expected_source_sha256: d.source_sha256.clone(),
                vector: decode_f32_le(&blobs[&d.document_id], spec.dimensions).unwrap(),
            })
            .collect::<Vec<_>>();
        assert!(
            store
                .retrieval()
                .publish_aggregate_embedding_catalog(&spec, &snapshot, &writes[..writes.len() - 1])
                .is_err()
        );
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let fail_id = writes.last().unwrap().document_id.replace('\'', "''");
        connection.execute_batch(&format!("CREATE TRIGGER embedding_catalog_late_aggregate_failure BEFORE INSERT ON memory_embeddings WHEN NEW.document_id='{fail_id}' BEGIN SELECT RAISE(ABORT,'late aggregate failure'); END;")).unwrap();
        assert!(
            store
                .retrieval()
                .publish_aggregate_embedding_catalog(&spec, &snapshot, &writes)
                .is_err()
        );
        assert_eq!(connection.query_row("SELECT count(*) FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id WHERE d.granularity IN ('episode','session')",[],|row|row.get::<_,i64>(0)).unwrap(),0);
        connection
            .execute_batch("DROP TRIGGER embedding_catalog_late_aggregate_failure")
            .unwrap();
        let report = store
            .retrieval()
            .publish_aggregate_embedding_catalog(&spec, &snapshot, &writes)
            .unwrap();
        assert!(report.changed);
        let fresh = store
            .retrieval()
            .aggregate_embedding_snapshot(&spec)
            .unwrap();
        let noop = store
            .retrieval()
            .publish_aggregate_embedding_catalog(&spec, &fresh, &writes)
            .unwrap();
        assert!(!noop.changed);
    }

    #[test]
    fn embedding_catalog_leaf_invalid_inputs_and_late_failure_are_zero_write() {
        let (_root, store, session, spec) = embedding_catalog_fixture(&"汉".repeat(300));
        let source_path = store.root().join(format!("{}.json", session.id));
        let source_before = fs::read(&source_path).unwrap();
        let snapshot = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let writes = embedding_catalog_unit_writes(&snapshot, spec.dimensions);
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        for invalid in [
            vec![writes[0].clone(), writes[0].clone()],
            {
                let mut v = writes.clone();
                v.push(EmbeddingWrite {
                    document_id: "extra".into(),
                    expected_source_sha256: "a".repeat(64),
                    vector: vec![0.0; spec.dimensions],
                });
                v
            },
            {
                let mut v = writes.clone();
                v[0].expected_source_sha256 = "b".repeat(64);
                v
            },
            {
                let mut v = writes.clone();
                v[0].vector = vec![0.0; spec.dimensions - 1];
                v
            },
            {
                let mut v = writes.clone();
                v[0].vector[0] = f32::NAN;
                v
            },
            {
                let mut v = writes.clone();
                v[0].vector.fill(0.0);
                v
            },
            {
                let mut v = writes.clone();
                v[0].vector.fill(0.0);
                v[0].vector[0] = 0.5;
                v
            },
        ] {
            assert!(matches!(
                store
                    .retrieval()
                    .publish_leaf_embedding_catalog(&spec, &snapshot, &invalid),
                Err(RetrievalError::CorruptIndex(_))
            ));
            assert_eq!(
                connection
                    .query_row("SELECT count(*) FROM memory_embeddings", [], |r| r
                        .get::<_, i64>(0))
                    .unwrap(),
                0
            );
        }
        let fail_id = &writes.last().unwrap().document_id.replace('\'', "''");
        connection.execute_batch(&format!("CREATE TRIGGER embedding_catalog_late_leaf_failure BEFORE INSERT ON memory_embeddings WHEN NEW.document_id='{fail_id}' BEGIN SELECT RAISE(ABORT,'late failure'); END;")).unwrap();
        assert!(
            store
                .retrieval()
                .publish_leaf_embedding_catalog(&spec, &snapshot, &writes)
                .is_err()
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM memory_embeddings", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(fs::read(source_path).unwrap(), source_before);
    }

    #[test]
    fn embedding_catalog_stale_leaf_row_is_refreshed_exactly_and_invalidates_aggregate() {
        let (_root, store, session, spec) = embedding_catalog_fixture("stale row refresh");
        let source_path = store.root().join(format!("{}.json", session.id));
        let source_before = fs::read(&source_path).unwrap();
        let leaf = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let writes = embedding_catalog_unit_writes(&leaf, spec.dimensions);
        store
            .retrieval()
            .publish_leaf_embedding_catalog(&spec, &leaf, &writes)
            .unwrap();
        store
            .retrieval()
            .materialize_episode_documents(&session.id, &aggregate_memory_config())
            .unwrap();
        let aggregate = store
            .retrieval()
            .aggregate_embedding_snapshot(&spec)
            .unwrap();
        let aggregate_blobs =
            canonical_aggregate_blobs_from_snapshot(&aggregate, spec.dimensions).unwrap();
        let aggregate_writes = aggregate
            .documents
            .iter()
            .map(|document| EmbeddingWrite {
                document_id: document.document_id.clone(),
                expected_source_sha256: document.source_sha256.clone(),
                vector: decode_f32_le(&aggregate_blobs[&document.document_id], spec.dimensions)
                    .unwrap(),
            })
            .collect::<Vec<_>>();
        store
            .retrieval()
            .publish_aggregate_embedding_catalog(&spec, &aggregate, &aggregate_writes)
            .unwrap();

        let target = 0;
        let stale_blob = encode_f32_le(&vec![0.25; spec.dimensions]).unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "UPDATE memory_embeddings
                 SET model='stale-model',source_sha256=?1,index_fingerprint=?2,
                     vector_blob=?3,embedded_at='stale-time'
                 WHERE document_id=?4",
                params![
                    "b".repeat(64),
                    "c".repeat(64),
                    stale_blob,
                    writes[target].document_id
                ],
            )
            .unwrap();
        let fresh = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        assert!(fresh.documents[target].reusable_vector.is_none());
        let report = store
            .retrieval()
            .publish_leaf_embedding_catalog(&spec, &fresh, &writes)
            .unwrap();
        assert!(report.changed);
        assert_eq!(report.reused, writes.len() - 1);

        let fingerprint = spec.fingerprint().unwrap();
        let expected_blob = encode_f32_le(&writes[target].vector).unwrap();
        let restored = connection
            .query_row(
                "SELECT model,dimensions,source_sha256,index_fingerprint,vector_blob,embedded_at
                 FROM memory_embeddings WHERE document_id=?1",
                [&writes[target].document_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Vec<u8>>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(restored.0, spec.model);
        assert_eq!(restored.1, i64::try_from(spec.dimensions).unwrap());
        assert_eq!(restored.2, writes[target].expected_source_sha256);
        assert_eq!(restored.3, fingerprint);
        assert_eq!(restored.4, expected_blob);
        assert_ne!(restored.5, "stale-time");
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_episode_materializations WHERE session_id=?1",
                    [&session.id],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_embeddings e JOIN memory_documents d
                     ON d.document_id=e.document_id
                     WHERE d.session_id=?1 AND d.granularity IN ('episode','session')",
                    [&session.id],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
        assert_eq!(fs::read(source_path).unwrap(), source_before);
    }

    #[test]
    fn embedding_catalog_fragment_change_invalidates_only_own_session_aggregates() {
        let long_first = "甲".repeat(441);
        let long_second = "乙".repeat(441);
        let (_root, store, sessions, spec) =
            embedding_catalog_two_session_fixture([&long_first, &long_second]);
        let config = aggregate_memory_config();
        let leaf = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let writes = embedding_catalog_unit_writes(&leaf, spec.dimensions);
        store
            .retrieval()
            .publish_leaf_embedding_catalog(&spec, &leaf, &writes)
            .unwrap();
        for session in &sessions {
            store
                .retrieval()
                .materialize_episode_documents(&session.id, &config)
                .unwrap();
        }
        let aggregate = store
            .retrieval()
            .aggregate_embedding_snapshot(&spec)
            .unwrap();
        let blobs = canonical_aggregate_blobs_from_snapshot(&aggregate, spec.dimensions).unwrap();
        let aggregate_writes = aggregate
            .documents
            .iter()
            .map(|d| EmbeddingWrite {
                document_id: d.document_id.clone(),
                expected_source_sha256: d.source_sha256.clone(),
                vector: decode_f32_le(&blobs[&d.document_id], spec.dimensions).unwrap(),
            })
            .collect::<Vec<_>>();
        store
            .retrieval()
            .publish_aggregate_embedding_catalog(&spec, &aggregate, &aggregate_writes)
            .unwrap();
        let aggregate_counts = sessions
            .iter()
            .map(|session| {
                aggregate
                    .documents
                    .iter()
                    .filter(|document| document.session_id == session.id)
                    .count()
            })
            .collect::<Vec<_>>();
        let fresh = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let mut changed = writes.clone();
        let target = fresh
            .documents
            .iter()
            .position(|document| {
                document.session_id == sessions[0].id
                    && document.granularity == RetrievalDocumentGranularity::Fragment
            })
            .unwrap();
        assert_eq!(
            fresh.documents[target].granularity,
            RetrievalDocumentGranularity::Fragment
        );
        changed[target].vector.fill(0.0);
        changed[target].vector[(target + 1) % spec.dimensions] = 1.0;
        assert_ne!(writes[target].vector, changed[target].vector);
        assert!(
            writes
                .iter()
                .zip(&changed)
                .enumerate()
                .all(|(index, (before, after))| index == target || before.vector == after.vector)
        );
        store
            .retrieval()
            .publish_leaf_embedding_catalog(&spec, &fresh, &changed)
            .unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_episode_materializations WHERE session_id=?1",
                    [&sessions[0].id],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_embeddings e JOIN memory_documents d
                     ON d.document_id=e.document_id
                     WHERE d.session_id=?1 AND d.granularity IN ('episode','session')",
                    [&sessions[0].id],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_episode_materializations WHERE session_id=?1",
                    [&sessions[1].id],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_embeddings e JOIN memory_documents d
                     ON d.document_id=e.document_id
                     WHERE d.session_id=?1 AND d.granularity IN ('episode','session')",
                    [&sessions[1].id],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            i64::try_from(aggregate_counts[1]).unwrap()
        );
    }

    #[test]
    fn embedding_catalog_aggregate_requires_ready_unit_direct_messages_and_cas() {
        let (_root, store, session, spec) = embedding_catalog_fixture("unit required");
        assert!(matches!(
            store.retrieval().aggregate_embedding_snapshot(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
        let leaf = store.retrieval().leaf_embedding_snapshot(&spec).unwrap();
        let writes = embedding_catalog_unit_writes(&leaf, spec.dimensions);
        store
            .retrieval()
            .publish_leaf_embedding_catalog(&spec, &leaf, &writes)
            .unwrap();
        store
            .retrieval()
            .materialize_episode_documents(&session.id, &aggregate_memory_config())
            .unwrap();
        let message_write = writes
            .iter()
            .find(|write| {
                leaf.documents
                    .iter()
                    .find(|document| document.document_id == write.document_id)
                    .is_some_and(|document| {
                        document.granularity == RetrievalDocumentGranularity::Message
                    })
            })
            .unwrap();
        let non_unit_blob = encode_f32_le(&vec![0.5; spec.dimensions]).unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "UPDATE memory_embeddings SET vector_blob=?1 WHERE document_id=?2",
                params![non_unit_blob, message_write.document_id],
            )
            .unwrap();
        assert!(matches!(
            store.retrieval().aggregate_embedding_snapshot(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
        connection
            .execute(
                "UPDATE memory_embeddings SET vector_blob=?1 WHERE document_id=?2",
                params![
                    encode_f32_le(&message_write.vector).unwrap(),
                    message_write.document_id
                ],
            )
            .unwrap();
        let aggregate = store
            .retrieval()
            .aggregate_embedding_snapshot(&spec)
            .unwrap();
        connection
            .execute(
                "UPDATE memory_episode_materializations
                 SET materialized_at='2030-01-01T00:00:00Z' WHERE session_id=?1",
                [&session.id],
            )
            .unwrap();
        let blobs = canonical_aggregate_blobs_from_snapshot(&aggregate, spec.dimensions).unwrap();
        let aggregate_writes = aggregate
            .documents
            .iter()
            .map(|d| EmbeddingWrite {
                document_id: d.document_id.clone(),
                expected_source_sha256: d.source_sha256.clone(),
                vector: decode_f32_le(&blobs[&d.document_id], spec.dimensions).unwrap(),
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            store.retrieval().publish_aggregate_embedding_catalog(
                &spec,
                &aggregate,
                &aggregate_writes
            ),
            Err(RetrievalError::EmbeddingCatalogStale { .. })
        ));
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_embeddings e JOIN memory_documents d
                     ON d.document_id=e.document_id
                     WHERE d.granularity IN ('episode','session')",
                    [],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn mention_ddl_rejects_non_lowercase_ids_and_hashes() {
        let root = tempfile::tempdir().unwrap();
        let store = RetrievalStore::new(root.path()).unwrap();
        let connection = store.open_connection().unwrap();
        connection.execute(
            "INSERT INTO memory_entities(entity_id,kind,canonical_name,normalized_name,disambiguation,
             created_session_id,created_batch_key,created_event_id,created_start,created_end,created_hash,created_at,updated_at)
             VALUES('ent_test','person','Test','test','resolved','s','b','e',0,1,?1,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')",
            ["a".repeat(64)],
        ).unwrap();
        let suffix = |byte: u8, index: usize| {
            let mut value = "a".repeat(64).into_bytes();
            value[index] = byte;
            String::from_utf8(value).unwrap()
        };
        let valid_hash = "b".repeat(64);
        let valid_id = |index: usize| format!("mention_{}", suffix(b'c', index));
        let mut expected = std::collections::BTreeSet::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut insert = |label: &str, mention_id: String, hash: String, valid: bool| {
            expected.insert(label.to_owned());
            let result = connection.execute(
                "INSERT INTO memory_entity_mentions(mention_id,session_id,batch_key,mention_kind,source_record_id,entity_id,entity_status,event_id,sequence,role,start_char,end_char,content_sha256,created_at)
                 VALUES(?1,?2,'b','entity_name',?3,'ent_test','resolved','e',1,'user',0,1,?4,'2026-01-01T00:00:00Z')",
                params![
                    mention_id,
                    format!("session-{label}"),
                    format!("source-{label}"),
                    hash
                ],
            );
            if valid {
                assert_eq!(result.unwrap(), 1, "{label}");
            } else {
                assert!(
                    matches!(
                        result,
                        Err(rusqlite::Error::SqliteFailure(error, _))
                            if error.code == rusqlite::ErrorCode::ConstraintViolation
                    ),
                    "{label}: {result:?}"
                );
            }
            seen.insert(label.to_owned());
        };

        insert("id_valid_control", valid_id(0), valid_hash.clone(), true);
        for (position, index) in [("start", 0), ("middle", 31), ("end", 63)] {
            insert(
                &format!("id_uppercase_{position}"),
                format!("mention_{}", suffix(b'A', index)),
                valid_hash.clone(),
                false,
            );
            insert(
                &format!("id_nonhex_g_{position}"),
                format!("mention_{}", suffix(b'g', index)),
                valid_hash.clone(),
                false,
            );
            insert(
                &format!("id_nonhex_underscore_{position}"),
                format!("mention_{}", suffix(b'_', index)),
                valid_hash.clone(),
                false,
            );
        }
        for (label, id) in [
            ("id_wrong_prefix", format!("wrong___{}", "a".repeat(64))),
            ("id_missing_prefix", "a".repeat(64)),
            ("id_suffix_short_63", format!("mention_{}", "a".repeat(63))),
            ("id_suffix_long_65", format!("mention_{}", "a".repeat(65))),
            ("id_empty", String::new()),
            ("id_blank", " ".into()),
        ] {
            insert(label, id, valid_hash.clone(), false);
        }
        insert("hash_valid_control", valid_id(1), valid_hash.clone(), true);
        for (ordinal, (position, index)) in [("start", 0), ("middle", 31), ("end", 63)]
            .into_iter()
            .enumerate()
        {
            insert(
                &format!("hash_uppercase_{position}"),
                valid_id(2 + ordinal * 3),
                suffix(b'B', index),
                false,
            );
            insert(
                &format!("hash_nonhex_g_{position}"),
                valid_id(3 + ordinal * 3),
                suffix(b'g', index),
                false,
            );
            insert(
                &format!("hash_nonhex_underscore_{position}"),
                valid_id(4 + ordinal * 3),
                suffix(b'_', index),
                false,
            );
        }
        for (ordinal, (label, hash)) in [
            ("hash_short_63", "b".repeat(63)),
            ("hash_long_65", "b".repeat(65)),
            ("hash_empty", String::new()),
        ]
        .into_iter()
        .enumerate()
        {
            insert(label, valid_id(11 + ordinal), hash, false);
        }
        assert_eq!(seen, expected);
    }

    #[test]
    fn memory_state_v4_fresh_index_records_current_schema() {
        let root = tempfile::tempdir().unwrap();
        let store = RetrievalStore::new(root.path()).unwrap();
        let connection = store.open_connection().unwrap();
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            7
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT value FROM memory_schema_meta WHERE key='state_schema_version'",
                    [],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            4
        );
    }

    const OLD_MEMORY_V1_SCHEMA: &str = r#"
CREATE TABLE consolidation_watermarks (
    session_id TEXT PRIMARY KEY,
    through_sequence INTEGER NOT NULL CHECK(through_sequence >= 0),
    through_event_id TEXT,
    through_event_sha256 TEXT,
    updated_at TEXT,
    CHECK((through_event_id IS NULL AND through_event_sha256 IS NULL)
       OR (through_event_id IS NOT NULL AND through_event_sha256 IS NOT NULL))
);
CREATE TABLE memory_entities (
    entity_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('person','organization','location','object','concept','unknown')),
    canonical_name TEXT NOT NULL CHECK(length(canonical_name) > 0),
    normalized_name TEXT NOT NULL CHECK(length(normalized_name) > 0),
    disambiguation TEXT NOT NULL CHECK(disambiguation IN ('resolved','pending')),
    created_session_id TEXT NOT NULL CHECK(length(created_session_id) > 0),
    created_batch_key TEXT NOT NULL CHECK(length(created_batch_key) > 0),
    created_event_id TEXT NOT NULL CHECK(length(created_event_id) > 0),
    created_start INTEGER NOT NULL CHECK(created_start >= 0),
    created_end INTEGER NOT NULL CHECK(created_end > created_start),
    created_hash TEXT NOT NULL CHECK(length(created_hash) = 64),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX memory_entities_normalized ON memory_entities(normalized_name, kind, entity_id);
CREATE TABLE memory_entity_aliases (
    alias_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    alias_text TEXT NOT NULL CHECK(length(alias_text) > 0),
    normalized_alias TEXT NOT NULL CHECK(length(normalized_alias) > 0),
    alias_kind TEXT NOT NULL CHECK(alias_kind IN ('explicit_alias','stable_identifier')),
    stable_identifier_kind TEXT,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    event_id TEXT NOT NULL CHECK(length(event_id) > 0),
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char > start_char),
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
    created_at TEXT NOT NULL,
    CHECK((alias_kind = 'explicit_alias' AND stable_identifier_kind IS NULL)
       OR (alias_kind = 'stable_identifier' AND stable_identifier_kind IS NOT NULL
           AND length(stable_identifier_kind) > 0))
);
CREATE INDEX memory_entity_aliases_entity ON memory_entity_aliases(entity_id, alias_id);
CREATE INDEX memory_entity_aliases_normalized
    ON memory_entity_aliases(alias_kind, stable_identifier_kind, normalized_alias, entity_id);
CREATE TABLE memory_claims (
    claim_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    subject_entity_id TEXT NOT NULL CHECK(length(subject_entity_id) > 0),
    predicate_key TEXT NOT NULL CHECK(length(predicate_key) > 0),
    object_kind TEXT NOT NULL CHECK(object_kind IN ('text','entity')),
    object_text TEXT,
    object_entity_id TEXT,
    normalized_object TEXT NOT NULL CHECK(length(normalized_object) > 0),
    polarity TEXT NOT NULL CHECK(polarity IN ('assert','deny')),
    cardinality TEXT NOT NULL CHECK(cardinality IN ('single','multi')),
    certainty TEXT NOT NULL CHECK(certainty IN ('certain','uncertain')),
    state TEXT NOT NULL CHECK(state IN ('active','superseded','conflicted','uncertain')),
    asserted_at TEXT NOT NULL,
    event_time TEXT,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    reference_time TEXT NOT NULL,
    created_batch_key TEXT NOT NULL CHECK(length(created_batch_key) > 0),
    updated_batch_key TEXT NOT NULL CHECK(length(updated_batch_key) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK((object_kind = 'text' AND object_text IS NOT NULL AND object_entity_id IS NULL)
       OR (object_kind = 'entity' AND object_text IS NULL AND object_entity_id IS NOT NULL))
);
CREATE INDEX memory_claims_subject_predicate
    ON memory_claims(subject_entity_id, predicate_key, state, claim_id);
CREATE INDEX memory_claims_updated ON memory_claims(updated_at DESC, claim_id);
CREATE TABLE memory_claim_evidence (
    evidence_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    event_id TEXT NOT NULL CHECK(length(event_id) > 0),
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    kind TEXT NOT NULL CHECK(kind IN ('assertion','user_confirmation','correction','temporal')),
    start_char INTEGER NOT NULL CHECK(start_char >= 0),
    end_char INTEGER NOT NULL CHECK(end_char > start_char),
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
    created_at TEXT NOT NULL
);
CREATE INDEX memory_claim_evidence_claim
    ON memory_claim_evidence(claim_id, event_id, start_char, end_char, evidence_id);
CREATE INDEX memory_claim_evidence_event ON memory_claim_evidence(event_id, claim_id);
CREATE TABLE memory_claim_transitions (
    transition_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    from_state TEXT CHECK(from_state IS NULL OR from_state IN ('active','superseded','conflicted','uncertain')),
    to_state TEXT NOT NULL CHECK(to_state IN ('active','superseded','conflicted','uncertain')),
    reason TEXT NOT NULL CHECK(reason IN ('created','confirmed','certainty_upgraded','conflicted','corrected','replaced')),
    related_claim_id TEXT,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    created_at TEXT NOT NULL
);
CREATE INDEX memory_claim_transitions_claim
    ON memory_claim_transitions(claim_id, created_at, transition_id);
CREATE TABLE memory_boundary_suggestions (
    boundary_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
    before_event_id TEXT NOT NULL CHECK(length(before_event_id) > 0),
    reason TEXT NOT NULL CHECK(reason IN ('explicit_topic_transition','model_topic_shift')),
    evidence_json TEXT NOT NULL CHECK(length(evidence_json) > 0),
    created_at TEXT NOT NULL
);
CREATE INDEX memory_boundary_suggestions_session_event
    ON memory_boundary_suggestions(session_id, before_event_id, boundary_id);
"#;

    fn replace_with_old_memory_v1_schema(connection: &Connection) {
        connection
            .execute_batch(
                "PRAGMA foreign_keys=OFF;
                 DROP TABLE memory_schema_meta;
                 DROP TABLE memory_claim_evidence;
                 DROP TABLE memory_claim_transitions;
                 DROP TABLE memory_boundary_suggestions;
                 DROP TABLE memory_claims;
                 DROP TABLE memory_entity_aliases;
                 DROP TABLE memory_entities;
                 DROP TABLE consolidation_watermarks;",
            )
            .unwrap();
        connection.execute_batch(OLD_MEMORY_V1_SCHEMA).unwrap();
    }

    fn replace_transition_table_with_memory_v2_schema(connection: &Connection) {
        connection
            .execute_batch(
                "PRAGMA foreign_keys=OFF;
                 DROP INDEX memory_claim_transitions_claim;
                 DROP TABLE memory_claim_transitions;
                 CREATE TABLE memory_claim_transitions (
                    transition_id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    from_state TEXT CHECK(from_state IS NULL OR from_state IN
                        ('active','superseded','conflicted','uncertain')),
                    to_state TEXT NOT NULL CHECK(to_state IN
                        ('active','superseded','conflicted','uncertain')),
                    reason TEXT NOT NULL CHECK(reason IN
                        ('created','confirmed','certainty_upgraded','conflicted','corrected','replaced')),
                    related_claim_id TEXT,
                    session_id TEXT NOT NULL CHECK(length(session_id) > 0),
                    batch_key TEXT NOT NULL CHECK(length(batch_key) > 0),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(claim_id) REFERENCES memory_claims(claim_id),
                    FOREIGN KEY(related_claim_id) REFERENCES memory_claims(claim_id)
                 );
                 CREATE INDEX memory_claim_transitions_claim
                    ON memory_claim_transitions(claim_id, created_at, transition_id);
                 UPDATE memory_schema_meta SET value=2 WHERE key='state_schema_version';",
            )
            .unwrap();
    }

    #[test]
    fn episode_materialize_preserves_direct_messages_and_is_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "短", "较长的原始回复", "ignored");
        append_complete_turn(&mut session, "第二条", "第二个回复", "ignored");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = MemoryConfig {
            enabled: true,
            ..Default::default()
        };
        let first = store
            .retrieval()
            .materialize_episode_documents(&session.id, &config)
            .unwrap();
        let second = store
            .retrieval()
            .materialize_episode_documents(&session.id, &config)
            .unwrap();
        assert_eq!(first.episode_documents, second.episode_documents);
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let aggregate_members: i64 = connection.query_row("SELECT count(*) FROM memory_document_members m JOIN memory_documents d ON d.document_id=m.document_id WHERE d.granularity IN ('episode','session') AND (m.start_char <> 0 OR m.event_id || ':0:' || m.end_char NOT IN (SELECT document_id FROM memory_documents WHERE granularity='message'))", [], |row| row.get(0)).unwrap();
        assert_eq!(aggregate_members, 0);
        assert!(first.session_document_id.is_some());
    }

    #[test]
    fn embedding_aggregate_freshness_excludes_stale_aggregate_vectors() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let doc = report.episode_documents[0].clone();
        let write = aggregate_writes(&store, &report, &spec, 1.0)
            .into_iter()
            .find(|write| write.document_id == doc.document_id)
            .unwrap();
        store
            .retrieval()
            .upsert_embeddings(&spec, &[write])
            .unwrap();
        assert!(
            store
                .retrieval()
                .compatible_embeddings(&spec)
                .unwrap()
                .iter()
                .any(|value| value.document_id == doc.document_id)
        );
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "DELETE FROM memory_episode_materializations WHERE session_id=?1",
                [&session.id],
            )
            .unwrap();
        assert!(
            !store
                .retrieval()
                .compatible_embeddings(&spec)
                .unwrap()
                .iter()
                .any(|value| value.document_id == doc.document_id)
        );
    }

    #[test]
    fn v5_to_v7_preserves_leaf_vectors_and_replaces_trigger() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "你好🙂", "回复", "");
        let path = store.save(&mut session).unwrap();
        let source_bytes = fs::read(&path).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let replay_before = store.retrieval().replay_session(&session.id).unwrap();
        let spec = VectorIndexSpec {
            model: "fixture-embedding-model".into(),
            dimensions: 32,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
        };
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let (document_id, source_sha256): (String, String) = connection
            .query_row(
                "SELECT document_id, source_sha256 FROM memory_documents
                 WHERE session_id=?1 AND granularity='message' ORDER BY document_id LIMIT 1",
                [&session.id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        drop(connection);
        store
            .retrieval()
            .upsert_embeddings(
                &spec,
                &[EmbeddingWrite {
                    document_id: document_id.clone(),
                    expected_source_sha256: source_sha256,
                    vector: (0..spec.dimensions)
                        .map(|index| index as f32 - 7.5)
                        .collect(),
                }],
            )
            .unwrap();
        let embedding_before = store
            .retrieval()
            .compatible_embeddings(&spec)
            .unwrap()
            .into_iter()
            .find(|embedding| embedding.document_id == document_id)
            .unwrap();
        assert_eq!(
            embedding_before.granularity,
            RetrievalDocumentGranularity::Message
        );

        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute_batch(
                "DROP TRIGGER memory_documents_before_source_span_delete;
                 DROP TABLE memory_episode_boundaries;
                 DROP TABLE memory_episode_materializations;
                 CREATE TRIGGER memory_documents_before_source_span_delete
                 BEFORE DELETE ON source_spans
                 BEGIN
                     DELETE FROM memory_documents
                     WHERE document_id IN (
                         SELECT document_id FROM memory_document_members
                         WHERE event_id = OLD.event_id
                           AND start_char = OLD.start_char
                           AND end_char = OLD.end_char
                     );
                 END;",
            )
            .unwrap();
        connection
            .pragma_update(None, "user_version", 5_i64)
            .unwrap();
        for table in [
            "memory_episode_boundaries",
            "memory_episode_materializations",
        ] {
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?1",
                        [table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                0
            );
        }
        let historical_trigger: String = connection
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='memory_documents_before_source_span_delete'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!historical_trigger.contains("memory_episode_materializations"));
        drop(connection);
        let reopened = RetrievalStore::new(root.path()).unwrap();
        assert_eq!(reopened.replay_session(&session.id).unwrap(), replay_before);
        let connection = Connection::open(reopened.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            7
        );
        for table in [
            "memory_episode_boundaries",
            "memory_episode_materializations",
        ] {
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?1",
                        [table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                1
            );
        }
        let trigger: String = connection.query_row("SELECT sql FROM sqlite_master WHERE type='trigger' AND name='memory_documents_before_source_span_delete'", [], |row| row.get(0)).unwrap();
        assert!(trigger.contains("memory_episode_materializations"));
        drop(connection);
        let embedding_after = reopened
            .compatible_embeddings(&spec)
            .unwrap()
            .into_iter()
            .find(|embedding| embedding.document_id == document_id)
            .unwrap();
        assert_eq!(embedding_after, embedding_before);
        assert_eq!(fs::read(path).unwrap(), source_bytes);
    }

    #[test]
    fn memory_state_v4_migration_clears_aggregates_retains_leaf_raw_legacy_ledger_and_replays() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, &"字".repeat(241), "complete answer", "");
        let source_path = store.save(&mut session).unwrap();
        let source_before = fs::read(&source_path).unwrap();
        let batch = store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        let config = MemoryConfig {
            enabled: true,
            embedding_dimensions: 32,
            ..MemoryConfig::default()
        };
        let spec = VectorIndexSpec::from_config(&config).unwrap();
        let leaf_writes = {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            connection
                .prepare(
                    "SELECT document_id,source_sha256 FROM memory_documents
                     WHERE session_id=?1 AND granularity IN ('message','fragment')
                     ORDER BY document_id",
                )
                .unwrap()
                .query_map([&session.id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .unwrap()
                .enumerate()
                .map(|(index, row)| {
                    let (document_id, expected_source_sha256) = row.unwrap();
                    EmbeddingWrite {
                        document_id,
                        expected_source_sha256,
                        vector: vec![index as f32 + 1.0; spec.dimensions],
                    }
                })
                .collect::<Vec<_>>()
        };
        assert!(leaf_writes.iter().any(|write| {
            write.document_id.contains(":0:240") || write.document_id.contains(":200:241")
        }));
        store
            .retrieval()
            .upsert_embeddings(&spec, &leaf_writes)
            .unwrap();
        let plan = store
            .retrieval()
            .materialize_episode_documents(&session.id, &config)
            .unwrap();
        let aggregate_writes = aggregate_writes(&store, &plan, &spec, 10.0);
        assert!(aggregate_writes.len() >= 2);
        store
            .retrieval()
            .upsert_embeddings(&spec, &aggregate_writes)
            .unwrap();

        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let boundary_event_id: String = connection
            .query_row(
                "SELECT event_id FROM events WHERE session_id=?1 AND role='user'",
                [&session.id],
                |row| row.get(0),
            )
            .unwrap();
        connection
            .execute(
                "INSERT OR REPLACE INTO memory_episode_boundaries(session_id,before_event_id,decision_json,input_sha256)
                 VALUES(?1,?2,'{\"fixture\":true}',?3)",
                params![session.id, boundary_event_id, "a".repeat(64)],
            )
            .unwrap();
        let aggregate_before = episode_materialization_sql_state(&connection, &session.id);
        assert!(!aggregate_before.catalog.is_empty());
        assert!(!aggregate_before.members.is_empty());
        assert!(!aggregate_before.boundaries.is_empty());
        assert!(!aggregate_before.materialization.is_empty());
        assert!(!aggregate_before.embeddings.is_empty());
        let leaf_before = (
            episode_sql_rows(
                &connection,
                "SELECT d.document_id,d.granularity,d.source_sha256,d.start_sequence,d.end_sequence,
                        d.member_count,m.ordinal,m.event_id,m.start_char,m.end_char,m.content_sha256
                 FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id
                 WHERE d.session_id=?1 AND d.granularity IN ('message','fragment')
                 ORDER BY d.document_id,m.ordinal",
                &session.id,
            ),
            episode_sql_rows(
                &connection,
                "SELECT r.document_id,r.event_id,r.start_char,r.end_char,r.granularity,r.content_sha256,
                        r.exact_content,r.lexical_content,r.ngram_content
                 FROM retrieval_documents r JOIN events e ON e.event_id=r.event_id
                 WHERE e.session_id=?1 ORDER BY r.document_id",
                &session.id,
            ),
            episode_sql_rows(
                &connection,
                "SELECT s.event_id,s.start_char,s.end_char,s.content_sha256
                 FROM source_spans s JOIN events e ON e.event_id=s.event_id
                 WHERE e.session_id=?1 ORDER BY s.event_id,s.start_char,s.end_char",
                &session.id,
            ),
            episode_sql_rows(
                &connection,
                "SELECT e.document_id,e.model,e.dimensions,e.source_sha256,e.index_fingerprint,
                        e.vector_blob,e.embedded_at
                 FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id
                 WHERE d.session_id=?1 AND d.granularity IN ('message','fragment')
                 ORDER BY e.document_id",
                &session.id,
            ),
        );
        assert!(!leaf_before.0.is_empty());
        assert!(!leaf_before.1.is_empty());
        assert!(!leaf_before.2.is_empty());
        assert_eq!(leaf_before.3.len(), leaf_writes.len());
        let raw_events_before = episode_sql_rows(
            &connection,
            "SELECT event_id,session_id,turn_id,sequence,role,created_at,content,content_sha256,
                    reply_to_event_id,token_count,turn_status,done_reason,error
             FROM events WHERE session_id=?1 ORDER BY sequence,event_id",
            &session.id,
        );
        let event_ids = serde_json::to_string(
            &batch
                .events
                .iter()
                .map(|event| event.event_id.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let event_hashes = serde_json::to_string(
            &batch
                .events
                .iter()
                .map(|event| event.content_sha256.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let legacy_request_json = "{\"legacy\":true}";
        let legacy_response_json = "{\"legacy_response\":true}";
        connection
            .execute_batch(
                "DROP TRIGGER consolidation_batches_immutable_update;
                 DROP TRIGGER consolidation_batches_immutable_delete;
                 ALTER TABLE consolidation_batches DROP COLUMN projection_schema_version;",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO consolidation_batches
                 (attempt_id,batch_key,session_id,from_sequence,through_sequence,trigger,model,
                  request_json,request_sha256,input_event_ids,input_event_hashes,response_json,
                  response_sha256,status,input_tokens,output_tokens,latency_ms,started_at,
                  completed_at,validation_json,error_json)
                 VALUES('legacy-v3-applied',?1,?2,?3,?4,'legacy','legacy-model',?5,?6,?7,?8,
                        ?9,?10,'applied',7,3,1,?11,?12,'{\"legacy\":true}',NULL)",
                params![
                    batch.batch_key,
                    batch.session_id,
                    batch.from_sequence as i64,
                    batch.through_sequence as i64,
                    legacy_request_json,
                    content_sha256(legacy_request_json),
                    event_ids,
                    event_hashes,
                    legacy_response_json,
                    content_sha256(legacy_response_json),
                    batch.events.last().unwrap().created_at,
                    (DateTime::parse_from_rfc3339(&batch.events.last().unwrap().created_at)
                        .unwrap()
                        + chrono::Duration::seconds(1))
                    .to_rfc3339(),
                ],
            )
            .unwrap();
        let legacy_before = episode_sql_rows(
            &connection,
            "SELECT attempt_id,batch_key,session_id,from_sequence,through_sequence,trigger,model,
                    request_json,request_sha256,input_event_ids,input_event_hashes,response_json,
                    response_sha256,status,input_tokens,output_tokens,latency_ms,started_at,
                    completed_at,validation_json,error_json
             FROM consolidation_batches WHERE session_id=?1 ORDER BY attempt_id",
            &session.id,
        );
        assert_eq!(legacy_before.len(), 1);
        connection
            .execute(
                "UPDATE memory_schema_meta SET value=3 WHERE key='state_schema_version'",
                [],
            )
            .unwrap();
        connection
            .pragma_update(None, "user_version", 6_i64)
            .unwrap();
        drop(connection);

        let reopened = RetrievalStore::new(root.path()).unwrap();
        reopened.open_connection().unwrap();
        let connection = Connection::open(reopened.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            7
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT value FROM memory_schema_meta WHERE key='state_schema_version'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            4
        );
        let aggregate_after = episode_materialization_sql_state(&connection, &session.id);
        assert!(aggregate_after.catalog.is_empty());
        assert!(aggregate_after.members.is_empty());
        assert!(aggregate_after.boundaries.is_empty());
        assert!(aggregate_after.materialization.is_empty());
        assert!(aggregate_after.embeddings.is_empty());
        let leaf_after = (
            episode_sql_rows(
                &connection,
                "SELECT d.document_id,d.granularity,d.source_sha256,d.start_sequence,d.end_sequence,
                        d.member_count,m.ordinal,m.event_id,m.start_char,m.end_char,m.content_sha256
                 FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id
                 WHERE d.session_id=?1 AND d.granularity IN ('message','fragment')
                 ORDER BY d.document_id,m.ordinal",
                &session.id,
            ),
            episode_sql_rows(
                &connection,
                "SELECT r.document_id,r.event_id,r.start_char,r.end_char,r.granularity,r.content_sha256,
                        r.exact_content,r.lexical_content,r.ngram_content
                 FROM retrieval_documents r JOIN events e ON e.event_id=r.event_id
                 WHERE e.session_id=?1 ORDER BY r.document_id",
                &session.id,
            ),
            episode_sql_rows(
                &connection,
                "SELECT s.event_id,s.start_char,s.end_char,s.content_sha256
                 FROM source_spans s JOIN events e ON e.event_id=s.event_id
                 WHERE e.session_id=?1 ORDER BY s.event_id,s.start_char,s.end_char",
                &session.id,
            ),
            episode_sql_rows(
                &connection,
                "SELECT e.document_id,e.model,e.dimensions,e.source_sha256,e.index_fingerprint,
                        e.vector_blob,e.embedded_at
                 FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id
                 WHERE d.session_id=?1 AND d.granularity IN ('message','fragment')
                 ORDER BY e.document_id",
                &session.id,
            ),
        );
        assert_eq!(leaf_after, leaf_before);
        assert_eq!(
            episode_sql_rows(
                &connection,
                "SELECT event_id,session_id,turn_id,sequence,role,created_at,content,content_sha256,
                        reply_to_event_id,token_count,turn_status,done_reason,error
                 FROM events WHERE session_id=?1 ORDER BY sequence,event_id",
                &session.id,
            ),
            raw_events_before
        );
        let legacy_after = episode_sql_rows(
            &connection,
            "SELECT attempt_id,batch_key,session_id,from_sequence,through_sequence,trigger,model,
                    request_json,request_sha256,input_event_ids,input_event_hashes,response_json,
                    response_sha256,status,input_tokens,output_tokens,latency_ms,started_at,
                    completed_at,validation_json,error_json
             FROM consolidation_batches WHERE session_id=?1 AND attempt_id='legacy-v3-applied'",
            &session.id,
        );
        assert_eq!(legacy_after, legacy_before);
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM consolidation_batches
                     WHERE attempt_id='legacy-v3-applied' AND projection_schema_version IS NULL",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        drop(connection);
        assert_eq!(fs::read(&source_path).unwrap(), source_before);
        assert_eq!(
            reopened.next_consolidation_batch(&session.id).unwrap(),
            Some(batch.clone())
        );

        let candidates = reopened.consolidation_candidates(16, 16).unwrap();
        let output = StructuredConsolidationOutput {
            entities: vec![],
            claims: vec![],
            boundaries: vec![],
        };
        let request_json = serde_json::to_string(
            &canonical_consolidation_request(
                "current-model".into(),
                &batch,
                &candidates,
                4096,
                1024,
            )
            .unwrap(),
        )
        .unwrap();
        let response_json = serde_json::to_string(&output).unwrap();
        let started_at = batch.events.last().unwrap().created_at.clone();
        let attempt = ConsolidationAttemptRecord {
            attempt_id: "current-v4-replay".into(),
            batch_key: batch.batch_key.clone(),
            session_id: batch.session_id.clone(),
            from_sequence: batch.from_sequence,
            through_sequence: batch.through_sequence,
            trigger: "test".into(),
            model: "current-model".into(),
            request_sha256: content_sha256(&request_json),
            request_json,
            input_event_ids: batch
                .events
                .iter()
                .map(|event| event.event_id.clone())
                .collect(),
            input_event_hashes: batch
                .events
                .iter()
                .map(|event| event.content_sha256.clone())
                .collect(),
            response_sha256: Some(content_sha256(&response_json)),
            response_json: Some(response_json),
            status: ConsolidationAttemptStatus::Applied,
            input_tokens: Some(1),
            output_tokens: Some(1),
            latency_ms: 1,
            started_at: started_at.clone(),
            completed_at: (DateTime::parse_from_rfc3339(&started_at).unwrap()
                + chrono::Duration::seconds(1))
            .to_rfc3339(),
            validation_json: Some("{\"valid\":true}".into()),
            error_json: None,
        };
        let report = reopened
            .apply_consolidation_attempt(&batch, &candidates, &attempt)
            .unwrap();
        assert_eq!(report.mentions_created, 0);
        let connection = Connection::open(reopened.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM consolidation_batches WHERE batch_key=?1 AND status='applied' AND projection_schema_version=4",
                    [&batch.batch_key],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM consolidation_batches WHERE attempt_id='legacy-v3-applied' AND projection_schema_version IS NULL",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn episode_unicode_and_corruption_roll_back_existing_materialization() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "a🙂", "好", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = MemoryConfig {
            enabled: true,
            ..Default::default()
        };
        store
            .retrieval()
            .materialize_episode_documents(&session.id, &config)
            .unwrap();
        let event = store
            .retrieval()
            .replay_session(&session.id)
            .unwrap()
            .into_iter()
            .find(|event| event.role == EventRole::User)
            .unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "UPDATE source_spans SET content_sha256=?1 WHERE event_id=?2 AND start_char=0",
                params!["0".repeat(64), event.id],
            )
            .unwrap();
        assert!(matches!(
            store
                .retrieval()
                .materialize_episode_documents(&session.id, &config),
            Err(RetrievalError::CorruptIndex(_))
        ));
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_episode_materializations WHERE session_id=?1",
                    [&session.id],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn episode_aggregate_leaf_and_aggregate_batch_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let plan = store
            .retrieval()
            .materialize_episode_documents(&session.id, &config)
            .unwrap();
        let spec = VectorIndexSpec::from_config(&config).unwrap();
        let leaf = store
            .retrieval()
            .replay_session(&session.id)
            .unwrap()
            .into_iter()
            .find(|event| event.role == EventRole::User)
            .unwrap();
        let leaf_id = format!("{}:0:{}", leaf.id, leaf.content.chars().count());
        let leaf_hash = content_sha256(&leaf.content);
        let aggregate = &plan.episode_documents[0];
        assert!(
            store
                .retrieval()
                .upsert_embeddings(
                    &spec,
                    &[
                        EmbeddingWrite {
                            document_id: leaf_id,
                            expected_source_sha256: leaf_hash,
                            vector: vec![1.0; spec.dimensions]
                        },
                        EmbeddingWrite {
                            document_id: aggregate.document_id.clone(),
                            expected_source_sha256: aggregate.source_sha256.clone(),
                            vector: vec![1.0; spec.dimensions]
                        }
                    ]
                )
                .is_err()
        );
    }

    #[test]
    fn episode_aggregate_complete_message_coverage_is_published() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u1", "a1", "");
        append_complete_turn(&mut session, "u2", "a2", "");
        let path = store.save(&mut session).unwrap();
        let source_before = fs::read(&path).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let aggregates = aggregate_writes(&store, &report, &spec, 10.0);
        store
            .retrieval()
            .upsert_embeddings(&spec, &aggregates)
            .unwrap();
        let compatible = store.retrieval().compatible_embeddings(&spec).unwrap();
        for write in &aggregates {
            assert!(
                compatible
                    .iter()
                    .any(|value| value.document_id == write.document_id)
            );
        }
        assert!(
            compatible
                .iter()
                .any(|value| matches!(value.granularity, RetrievalDocumentGranularity::Episode))
        );
        assert!(
            compatible
                .iter()
                .any(|value| matches!(value.granularity, RetrievalDocumentGranularity::Session))
        );
        assert_eq!(
            store
                .retrieval()
                .embedding_coverage(&spec)
                .unwrap()
                .compatible,
            compatible.len()
        );
        assert_eq!(fs::read(path).unwrap(), source_before);
    }

    #[test]
    fn episode_aggregate_readiness_audit_scales_with_unique_sessions() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut first = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut first, "first user", "first assistant", "");
        let first_path = store.save(&mut first).unwrap();
        store.retrieval().sync_session(&first, &first_path).unwrap();
        let mut second = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut second, "second user", "second assistant", "");
        let second_path = store.save(&mut second).unwrap();
        store
            .retrieval()
            .sync_session(&second, &second_path)
            .unwrap();

        let config = aggregate_memory_config();
        let (spec, first_report) = materialize_with_canonical_embeddings(&store, &first, &config);
        let (_, second_report) = materialize_with_canonical_embeddings(&store, &second, &config);
        let mut writes = aggregate_writes(&store, &first_report, &spec, 101.0);
        writes.extend(aggregate_writes(&store, &second_report, &spec, 201.0));
        assert_eq!(writes.len(), 4, "two documents per aggregate session");

        let expected_sessions = [first.id.clone(), second.id.clone()]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        let observations = Arc::new(Mutex::new(Vec::new()));
        let hook_observations = Arc::clone(&observations);
        store
            .retrieval()
            .set_aggregate_audit_test_hook(Some(Arc::new(move |point| {
                hook_observations.lock().unwrap().push(point);
            })));

        store.retrieval().upsert_embeddings(&spec, &writes).unwrap();
        let write_observations = observations.lock().unwrap().clone();
        assert_eq!(
            write_observations
                .iter()
                .filter(|point| matches!(point, AggregateAuditHookPoint::DerivedIntegrity))
                .count(),
            1,
            "the full projection audit runs once for the write transaction"
        );
        for phase in [
            AggregateAuditPhase::PreWrite,
            AggregateAuditPhase::FinalWrite,
        ] {
            let sessions = write_observations
                .iter()
                .filter_map(|point| match point {
                    AggregateAuditHookPoint::Materialization {
                        session_id,
                        phase: observed,
                    } if *observed == phase => Some(session_id.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(sessions.len(), expected_sessions.len());
            assert_eq!(
                sessions
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>(),
                expected_sessions,
                "one cached readiness computation per session in {phase:?}"
            );
        }

        observations.lock().unwrap().clear();
        let compatible = store.retrieval().compatible_embeddings(&spec).unwrap();
        assert!(writes.iter().all(|write| {
            compatible
                .iter()
                .any(|embedding| embedding.document_id == write.document_id)
        }));
        let read_observations = observations.lock().unwrap().clone();
        assert_eq!(
            read_observations
                .iter()
                .filter(|point| matches!(point, AggregateAuditHookPoint::DerivedIntegrity))
                .count(),
            1,
            "the full projection audit runs once for the read operation"
        );
        let read_sessions = read_observations
            .iter()
            .filter_map(|point| match point {
                AggregateAuditHookPoint::Materialization {
                    session_id,
                    phase: AggregateAuditPhase::Read,
                } => Some(session_id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(read_sessions.len(), expected_sessions.len());
        assert_eq!(
            read_sessions
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            expected_sessions,
            "readiness recomputation follows unique sessions, not four aggregate documents"
        );
        store.retrieval().set_aggregate_audit_test_hook(None);
    }

    #[test]
    fn episode_aggregate_incomplete_or_incompatible_coverage_rolls_back() {
        for case in ["omitted", "fragment_only", "incompatible"] {
            let root = tempfile::tempdir().unwrap();
            let store = SessionStore::new(root.path()).unwrap();
            let mut session = store
                .create("model", "http://localhost", None, Default::default(), false)
                .unwrap();
            append_complete_turn(&mut session, &"字".repeat(241), "assistant", "");
            append_complete_turn(&mut session, "another user", "another assistant", "");
            let path = store.save(&mut session).unwrap();
            let source_before = fs::read(&path).unwrap();
            store.retrieval().sync_session(&session, &path).unwrap();
            let config = aggregate_memory_config();
            let spec = VectorIndexSpec::from_config(&config).unwrap();
            store
                .retrieval()
                .materialize_episode_documents(&session.id, &config)
                .unwrap();
            let mut leaves = canonical_message_writes(&store, &session, &spec, 1.0);
            match case {
                "omitted" => {
                    leaves.pop();
                    store.retrieval().upsert_embeddings(&spec, &leaves).unwrap();
                }
                "fragment_only" => {
                    let (fragment_id, fragment_hash, fragment_event_id): (String, String, String) = Connection::open(store.retrieval().index_path())
                        .unwrap().query_row(
                            "SELECT d.document_id, d.source_sha256, m.event_id
                             FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id
                             WHERE d.session_id=?1 AND d.granularity='fragment' ORDER BY d.document_id LIMIT 1",
                            [&session.id],
                            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                        )
                        .unwrap();
                    let prefix = format!("{fragment_event_id}:0:");
                    leaves.remove(
                        leaves
                            .iter()
                            .position(|write| write.document_id.starts_with(&prefix))
                            .unwrap(),
                    );
                    leaves.push(EmbeddingWrite {
                        document_id: fragment_id,
                        expected_source_sha256: fragment_hash,
                        vector: vec![3.0; spec.dimensions],
                    });
                    store.retrieval().upsert_embeddings(&spec, &leaves).unwrap();
                }
                "incompatible" => {
                    let mut alternative = spec.clone();
                    alternative.model = "other-compatible-model".into();
                    store
                        .retrieval()
                        .upsert_embeddings(&alternative, &leaves)
                        .unwrap();
                }
                _ => unreachable!(),
            }
            let report = store
                .retrieval()
                .materialize_episode_documents(&session.id, &config)
                .unwrap();
            let aggregate = aggregate_writes(&store, &report, &spec, 9.0).remove(0);
            assert!(matches!(
                store.retrieval().upsert_embeddings(&spec, &[aggregate]),
                Err(RetrievalError::CorruptIndex(_))
            ));
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            assert_eq!(connection.query_row("SELECT count(*) FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id WHERE d.session_id=?1 AND d.granularity IN ('episode','session')", [&session.id], |row| row.get::<_, i64>(0)).unwrap(), 0);
            assert_eq!(fs::read(path).unwrap(), source_before, "case {case}");
        }
    }

    #[test]
    fn episode_aggregate_leaf_update_invalidates_all_aggregates() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let aggregates = aggregate_writes(&store, &report, &spec, 5.0);
        store
            .retrieval()
            .upsert_embeddings(&spec, &aggregates)
            .unwrap();
        let replacement = canonical_message_writes(&store, &session, &spec, 42.0).remove(0);
        store
            .retrieval()
            .upsert_embeddings(&spec, &[replacement])
            .unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_episode_materializations WHERE session_id=?1",
                    [&session.id],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
        assert_eq!(connection.query_row("SELECT count(*) FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id WHERE d.session_id=?1 AND d.granularity IN ('episode','session')", [&session.id], |row| row.get::<_, i64>(0)).unwrap(), 0);
        assert!(
            !store
                .retrieval()
                .compatible_embeddings(&spec)
                .unwrap()
                .iter()
                .any(|value| matches!(
                    value.granularity,
                    RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session
                ))
        );
    }

    #[test]
    fn episode_aggregate_rematerialization_drops_old_vectors() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        store
            .retrieval()
            .upsert_embeddings(&spec, &aggregate_writes(&store, &report, &spec, 5.0))
            .unwrap();
        store
            .retrieval()
            .materialize_episode_documents(&session.id, &config)
            .unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        assert_eq!(connection.query_row("SELECT count(*) FROM memory_embeddings e JOIN memory_documents d ON d.document_id=e.document_id WHERE d.session_id=?1 AND d.granularity IN ('episode','session')", [&session.id], |row| row.get::<_, i64>(0)).unwrap(), 0);
        assert!(
            !store
                .retrieval()
                .compatible_embeddings(&spec)
                .unwrap()
                .iter()
                .any(|value| matches!(
                    value.granularity,
                    RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session
                ))
        );
    }

    #[test]
    fn episode_aggregate_post_write_corruption_rolls_back() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let original = aggregate_writes(&store, &report, &spec, 5.0).remove(0);
        store
            .retrieval()
            .upsert_embeddings(&spec, std::slice::from_ref(&original))
            .unwrap();
        let changed_bytes = encode_f32_le(&vec![7.0; spec.dimensions]).unwrap();
        let blob_literal = changed_bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection.execute_batch(&format!(
            "CREATE TRIGGER aggregate_writeback_corrupt AFTER UPDATE OF vector_blob ON memory_embeddings
             WHEN NEW.document_id = '{}' BEGIN
               UPDATE memory_embeddings SET vector_blob=X'{}' WHERE document_id=NEW.document_id;
             END;",
            original.document_id.replace('\'', "''"), blob_literal
        )).unwrap();
        let replacement = original.clone();
        assert!(matches!(
            store.retrieval().upsert_embeddings(&spec, &[replacement]),
            Err(RetrievalError::CorruptIndex(_))
        ));
        connection
            .execute_batch("DROP TRIGGER aggregate_writeback_corrupt;")
            .unwrap();
        let exposed = store.retrieval().compatible_embeddings(&spec).unwrap();
        assert!(
            exposed
                .iter()
                .any(|value| value.document_id == original.document_id
                    && value.vector == original.vector)
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_episode_materializations WHERE session_id=?1",
                    [&session.id],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn episode_aggregate_arbitrary_valid_length_vector_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let mut write = aggregate_writes(&store, &report, &spec, 5.0).remove(0);
        write.vector = vec![1.0; spec.dimensions];

        assert!(matches!(
            store.retrieval().upsert_embeddings(&spec, &[write]),
            Err(RetrievalError::CorruptIndex(_))
        ));
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_embeddings e
                     JOIN memory_documents d ON d.document_id=e.document_id
                     WHERE d.session_id=?1 AND d.granularity IN ('episode','session')",
                    [&session.id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn episode_aggregate_vector_is_normalized_mean_of_direct_messages() {
        let dimensions = 32;
        let members = [
            EpisodeMember {
                document_id: "message-one".into(),
                event_id: "event-one".into(),
                sequence: 1,
                role: EventRole::User,
                span: SourceSpan {
                    event_id: "event-one".into(),
                    start_char: 0,
                    end_char: 1,
                },
                content_sha256: "1".repeat(64),
            },
            EpisodeMember {
                document_id: "message-two".into(),
                event_id: "event-two".into(),
                sequence: 2,
                role: EventRole::Assistant,
                span: SourceSpan {
                    event_id: "event-two".into(),
                    start_char: 0,
                    end_char: 1,
                },
                content_sha256: "2".repeat(64),
            },
        ];
        let mut first = vec![0.0; dimensions];
        first[0] = 3.0;
        let mut second = vec![0.0; dimensions];
        second[1] = 4.0;
        let mut embeddings = HashMap::from([
            ("message-one".to_owned(), first),
            ("message-two".to_owned(), second),
        ]);

        let vector =
            canonical_aggregate_vector(&members, &embeddings, dimensions, "aggregate").unwrap();
        let expected = std::f64::consts::FRAC_1_SQRT_2 as f32;
        assert_eq!(vector[0].to_bits(), expected.to_bits());
        assert_eq!(vector[1].to_bits(), expected.to_bits());
        assert!(vector[2..].iter().all(|value| *value == 0.0));

        embeddings.insert("message-one".into(), vec![0.0; dimensions]);
        assert!(
            canonical_aggregate_vector(&members, &embeddings, dimensions, "aggregate").is_err()
        );
        embeddings.insert("message-one".into(), vec![1.0; dimensions - 1]);
        assert!(
            canonical_aggregate_vector(&members, &embeddings, dimensions, "aggregate").is_err()
        );
        embeddings.insert("message-one".into(), vec![f32::NAN; dimensions]);
        assert!(
            canonical_aggregate_vector(&members, &embeddings, dimensions, "aggregate").is_err()
        );
    }

    #[test]
    fn episode_aggregate_later_write_corruption_rolls_back_entire_batch() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        let source_before = fs::read(&path).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let initial_writes = aggregate_writes(&store, &report, &spec, 21.0);
        assert_eq!(initial_writes.len(), 2);
        for write in &initial_writes {
            store
                .retrieval()
                .upsert_embeddings(&spec, std::slice::from_ref(write))
                .unwrap();
        }
        let before = initial_writes
            .iter()
            .map(|write| {
                store
                    .retrieval()
                    .compatible_embeddings(&spec)
                    .unwrap()
                    .into_iter()
                    .find(|embedding| embedding.document_id == write.document_id)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let corrupted_bytes = encode_f32_le(&vec![91.0; spec.dimensions]).unwrap();
        let blob_literal = corrupted_bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute_batch(&format!(
                "CREATE TRIGGER aggregate_later_write_corrupts_earlier
                 AFTER UPDATE OF vector_blob ON memory_embeddings
                 WHEN NEW.document_id = '{}'
                 BEGIN
                     UPDATE memory_embeddings SET vector_blob=X'{}'
                     WHERE document_id='{}';
                 END;",
                initial_writes[1].document_id.replace('\'', "''"),
                blob_literal,
                initial_writes[0].document_id.replace('\'', "''"),
            ))
            .unwrap();
        let replacement_writes = aggregate_writes(&store, &report, &spec, 41.0);
        assert!(matches!(
            store
                .retrieval()
                .upsert_embeddings(&spec, &replacement_writes),
            Err(RetrievalError::CorruptIndex(_))
        ));
        connection
            .execute_batch("DROP TRIGGER aggregate_later_write_corrupts_earlier;")
            .unwrap();
        let after = initial_writes
            .iter()
            .map(|write| stored_embedding_unchecked(&connection, &write.document_id))
            .collect::<Vec<_>>();
        assert_eq!(after, before);
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_episode_materializations WHERE session_id=?1",
                    [&session.id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(fs::read(path).unwrap(), source_before);
    }

    #[test]
    fn episode_aggregate_unsubmitted_existing_corruption_rolls_back() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let writes = aggregate_writes(&store, &report, &spec, 23.0);
        assert_eq!(writes.len(), 2);
        store.retrieval().upsert_embeddings(&spec, &writes).unwrap();

        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let before = writes
            .iter()
            .map(|write| stored_embedding_unchecked(&connection, &write.document_id))
            .collect::<Vec<_>>();
        let corrupted_bytes = encode_f32_le(&vec![93.0; spec.dimensions]).unwrap();
        let blob_literal = corrupted_bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        connection
            .execute_batch(&format!(
                "CREATE TRIGGER aggregate_submitted_write_corrupts_unsubmitted
                 AFTER UPDATE OF vector_blob ON memory_embeddings
                 WHEN NEW.document_id = '{}'
                 BEGIN
                     UPDATE memory_embeddings SET vector_blob=X'{}'
                     WHERE document_id='{}';
                 END;",
                writes[1].document_id.replace('\'', "''"),
                blob_literal,
                writes[0].document_id.replace('\'', "''"),
            ))
            .unwrap();

        assert!(matches!(
            store
                .retrieval()
                .upsert_embeddings(&spec, std::slice::from_ref(&writes[1])),
            Err(RetrievalError::CorruptIndex(_))
        ));
        connection
            .execute_batch("DROP TRIGGER aggregate_submitted_write_corrupts_unsubmitted;")
            .unwrap();
        let after = writes
            .iter()
            .map(|write| stored_embedding_unchecked(&connection, &write.document_id))
            .collect::<Vec<_>>();
        assert_eq!(after, before);
    }

    #[test]
    fn episode_aggregate_materialization_later_write_rolls_back_entire_transaction() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u1", "a1", "");
        append_complete_turn(&mut session, "u2", "a2", "");
        let path = store.save(&mut session).unwrap();
        let source_before = fs::read(&path).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let aggregate_writes = aggregate_writes(&store, &report, &spec, 71.0);
        store
            .retrieval()
            .upsert_embeddings(&spec, &aggregate_writes)
            .unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let before = episode_materialization_sql_state(&connection, &session.id);
        let earlier_episode = &report.episode_documents[0].document_id;
        connection
            .execute_batch(&format!(
                "CREATE TRIGGER materialization_later_write_corrupts_earlier
                 AFTER UPDATE ON memory_episode_materializations
                 WHEN NEW.session_id = '{}'
                 BEGIN
                     UPDATE memory_documents SET source_sha256='{}'
                     WHERE document_id='{}';
                 END;",
                session.id.replace('\'', "''"),
                "0".repeat(64),
                earlier_episode.replace('\'', "''"),
            ))
            .unwrap();

        assert!(matches!(
            store
                .retrieval()
                .materialize_episode_documents(&session.id, &config),
            Err(RetrievalError::CorruptIndex(_))
        ));
        connection
            .execute_batch("DROP TRIGGER materialization_later_write_corrupts_earlier;")
            .unwrap();
        let after = episode_materialization_sql_state(&connection, &session.id);
        assert_eq!(after, before);
        assert_eq!(fs::read(path).unwrap(), source_before);
        let compatible = store.retrieval().compatible_embeddings(&spec).unwrap();
        assert!(aggregate_writes.iter().all(|write| {
            compatible
                .iter()
                .any(|embedding| embedding.document_id == write.document_id)
        }));
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum AggregateAuditTamper {
        LedgerSnapshot,
        PlanInput,
        AlgorithmVersion,
        GapMinutes,
        TopicThreshold,
        EpisodeCount,
        BoundaryCount,
        MaterializedAt,
        BoundaryDecision,
        BoundaryInputHash,
        AggregateRange,
        AggregateMember,
        AggregateSource,
        CoherentCatalog,
        RawSource,
    }

    #[test]
    fn episode_aggregate_materialization_audit_classifies_tampering() {
        let cases = [
            AggregateAuditTamper::LedgerSnapshot,
            AggregateAuditTamper::PlanInput,
            AggregateAuditTamper::AlgorithmVersion,
            AggregateAuditTamper::GapMinutes,
            AggregateAuditTamper::TopicThreshold,
            AggregateAuditTamper::EpisodeCount,
            AggregateAuditTamper::BoundaryCount,
            AggregateAuditTamper::MaterializedAt,
            AggregateAuditTamper::BoundaryDecision,
            AggregateAuditTamper::BoundaryInputHash,
            AggregateAuditTamper::AggregateRange,
            AggregateAuditTamper::AggregateMember,
            AggregateAuditTamper::AggregateSource,
            AggregateAuditTamper::CoherentCatalog,
            AggregateAuditTamper::RawSource,
        ];
        for tamper in cases {
            let root = tempfile::tempdir().unwrap();
            let store = SessionStore::new(root.path()).unwrap();
            let mut session = store
                .create("model", "http://localhost", None, Default::default(), false)
                .unwrap();
            append_complete_turn(&mut session, "first user", "first assistant", "");
            append_complete_turn(&mut session, "second user", "second assistant", "");
            let path = store.save(&mut session).unwrap();
            let source_before = fs::read(&path).unwrap();
            store.retrieval().sync_session(&session, &path).unwrap();
            let config = aggregate_memory_config();
            let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
            let aggregate_writes = aggregate_writes(&store, &report, &spec, 31.0);
            store
                .retrieval()
                .upsert_embeddings(&spec, &aggregate_writes)
                .unwrap();
            let target_write = aggregate_writes.last().unwrap().clone();
            let target_before = store
                .retrieval()
                .compatible_embeddings(&spec)
                .unwrap()
                .into_iter()
                .find(|embedding| embedding.document_id == target_write.document_id)
                .unwrap();

            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            apply_aggregate_audit_tamper(&connection, &session.id, tamper);
            if tamper == AggregateAuditTamper::RawSource {
                let mut changed_source = source_before.clone();
                changed_source.push(b'\n');
                fs::write(&path, changed_source).unwrap();
            }

            let replacement = target_write.clone();
            assert!(
                matches!(
                    store.retrieval().upsert_embeddings(&spec, &[replacement]),
                    Err(RetrievalError::CorruptIndex(_))
                ),
                "publication classification for {tamper:?}"
            );
            assert_eq!(
                stored_embedding_unchecked(&connection, &target_write.document_id),
                target_before,
                "publication rollback for {tamper:?}"
            );

            if tamper == AggregateAuditTamper::RawSource {
                let compatible = store.retrieval().compatible_embeddings(&spec).unwrap();
                assert!(compatible.iter().all(|embedding| !matches!(
                    embedding.granularity,
                    RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session
                )));
                let coverage = store.retrieval().embedding_coverage(&spec).unwrap();
                assert_eq!(coverage.compatible, compatible.len());
                assert!(coverage.stale >= aggregate_writes.len());
                fs::write(&path, &source_before).unwrap();
                let restored = store.retrieval().compatible_embeddings(&spec).unwrap();
                assert!(
                    restored
                        .iter()
                        .any(|embedding| embedding.document_id == target_write.document_id)
                );
            } else {
                assert!(
                    matches!(
                        store.retrieval().compatible_embeddings(&spec),
                        Err(RetrievalError::CorruptIndex(_))
                    ),
                    "exposure classification for {tamper:?}"
                );
                assert!(
                    matches!(
                        store.retrieval().embedding_coverage(&spec),
                        Err(RetrievalError::CorruptIndex(_))
                    ),
                    "coverage classification for {tamper:?}"
                );
                assert_eq!(fs::read(&path).unwrap(), source_before, "{tamper:?}");
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum ProjectionAuditTamper {
        WatermarkEventId,
        WatermarkEventHash,
        WatermarkUpdatedAt,
        BoundaryEvidence,
        BoundaryProvenance,
        EntityProjection,
        AliasProjection,
        ClaimProjection,
        ClaimEvidenceProjection,
        ClaimTransitionProjection,
    }

    #[test]
    fn episode_aggregate_projection_audit_rejects_corruption() {
        let cases = [
            ProjectionAuditTamper::WatermarkEventId,
            ProjectionAuditTamper::WatermarkEventHash,
            ProjectionAuditTamper::WatermarkUpdatedAt,
            ProjectionAuditTamper::BoundaryEvidence,
            ProjectionAuditTamper::BoundaryProvenance,
            ProjectionAuditTamper::EntityProjection,
            ProjectionAuditTamper::AliasProjection,
            ProjectionAuditTamper::ClaimProjection,
            ProjectionAuditTamper::ClaimEvidenceProjection,
            ProjectionAuditTamper::ClaimTransitionProjection,
        ];
        for tamper in cases {
            let root = tempfile::tempdir().unwrap();
            let store = SessionStore::new(root.path()).unwrap();
            let mut session = store
                .create("model", "http://localhost", None, Default::default(), false)
                .unwrap();
            append_complete_turn(&mut session, "Alice aka Al likes tea.", "Acknowledged.", "");
            append_complete_turn(&mut session, "change topic now.", "Changed.", "");
            let path = store.save(&mut session).unwrap();
            let source_before = fs::read(&path).unwrap();
            store.retrieval().sync_session(&session, &path).unwrap();
            seed_episode_projection(&store, &session.id);

            let config = aggregate_memory_config();
            let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
            let aggregate_writes = aggregate_writes(&store, &report, &spec, 51.0);
            store
                .retrieval()
                .upsert_embeddings(&spec, &aggregate_writes)
                .unwrap();
            let target_write = aggregate_writes.last().unwrap().clone();
            let target_before = stored_embedding_unchecked(
                &Connection::open(store.retrieval().index_path()).unwrap(),
                &target_write.document_id,
            );

            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            apply_projection_audit_tamper(&connection, &session.id, tamper);
            let replacement = target_write.clone();
            assert!(
                matches!(
                    store.retrieval().upsert_embeddings(&spec, &[replacement]),
                    Err(RetrievalError::CorruptIndex(_))
                ),
                "publication classification for {tamper:?}"
            );
            assert_eq!(
                stored_embedding_unchecked(&connection, &target_write.document_id),
                target_before,
                "publication rollback for {tamper:?}"
            );
            assert!(
                matches!(
                    store.retrieval().compatible_embeddings(&spec),
                    Err(RetrievalError::CorruptIndex(_))
                ),
                "exposure classification for {tamper:?}"
            );
            assert!(
                matches!(
                    store.retrieval().embedding_coverage(&spec),
                    Err(RetrievalError::CorruptIndex(_))
                ),
                "coverage classification for {tamper:?}"
            );
            assert_eq!(fs::read(&path).unwrap(), source_before, "{tamper:?}");
        }
    }

    fn apply_projection_audit_tamper(
        connection: &Connection,
        session_id: &str,
        tamper: ProjectionAuditTamper,
    ) {
        let fake_hash = "0".repeat(64);
        match tamper {
            ProjectionAuditTamper::WatermarkEventId => {
                connection
                    .execute(
                        "UPDATE consolidation_watermarks
                         SET through_event_id=(SELECT event_id FROM events
                             WHERE session_id=?1 AND role='user' ORDER BY sequence LIMIT 1)
                         WHERE session_id=?1",
                        [session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::WatermarkEventHash => {
                connection
                    .execute(
                        "UPDATE consolidation_watermarks SET through_event_sha256=?1
                         WHERE session_id=?2",
                        params![fake_hash, session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::WatermarkUpdatedAt => {
                connection
                    .execute(
                        "UPDATE consolidation_watermarks
                         SET updated_at='2000-01-01T00:00:00Z' WHERE session_id=?1",
                        [session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::BoundaryEvidence => {
                connection
                    .execute(
                        "UPDATE memory_boundary_suggestions SET evidence_json='[]'
                         WHERE session_id=?1",
                        [session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::BoundaryProvenance => {
                connection
                    .execute(
                        "UPDATE memory_boundary_suggestions SET batch_key='missing-batch'
                         WHERE session_id=?1",
                        [session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::EntityProjection => {
                connection
                    .execute(
                        "UPDATE memory_entities
                         SET canonical_name='Mallory', normalized_name='mallory'
                         WHERE created_session_id=?1 AND canonical_name='Alice'",
                        [session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::AliasProjection => {
                connection
                    .execute(
                        "UPDATE memory_entity_aliases
                         SET alias_text='Eve', normalized_alias='eve' WHERE session_id=?1",
                        [session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::ClaimProjection => {
                connection
                    .execute(
                        "UPDATE memory_claims
                         SET predicate_key='tampered.preference',
                             normalized_relation='tampered preference'
                         WHERE session_id=?1",
                        [session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::ClaimEvidenceProjection => {
                connection
                    .execute(
                        "UPDATE memory_claim_evidence SET relation_sha256=?1
                         WHERE session_id=?2",
                        params![fake_hash, session_id],
                    )
                    .unwrap();
            }
            ProjectionAuditTamper::ClaimTransitionProjection => {
                connection
                    .execute(
                        "UPDATE memory_claim_transitions SET to_state='uncertain'
                         WHERE session_id=?1",
                        [session_id],
                    )
                    .unwrap();
            }
        }
    }

    #[test]
    fn episode_aggregate_source_binding_rejects_alternate_same_byte_file() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        let source_before = fs::read(&path).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let aggregate_writes = aggregate_writes(&store, &report, &spec, 61.0);
        store
            .retrieval()
            .upsert_embeddings(&spec, &aggregate_writes)
            .unwrap();
        let target_write = aggregate_writes.last().unwrap().clone();
        let target_before = stored_embedding_unchecked(
            &Connection::open(store.retrieval().index_path()).unwrap(),
            &target_write.document_id,
        );

        let alternate = root.path().join("alternate.json");
        fs::write(&alternate, &source_before).unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "UPDATE indexed_sessions SET source_file='alternate.json' WHERE session_id=?1",
                [&session.id],
            )
            .unwrap();

        let replacement = target_write.clone();
        assert!(matches!(
            store.retrieval().upsert_embeddings(&spec, &[replacement]),
            Err(RetrievalError::CorruptIndex(_))
        ));
        assert_eq!(
            stored_embedding_unchecked(&connection, &target_write.document_id),
            target_before
        );
        assert!(matches!(
            store.retrieval().compatible_embeddings(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
        assert!(matches!(
            store.retrieval().embedding_coverage(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
        assert_eq!(fs::read(&path).unwrap(), source_before);
        assert_eq!(fs::read(alternate).unwrap(), source_before);
    }

    #[test]
    fn episode_aggregate_source_semantics_rejects_hash_coherent_wrong_session() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "u", "a", "");
        let path = store.save(&mut session).unwrap();
        let source_before = fs::read(&path).unwrap();
        store.retrieval().sync_session(&session, &path).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let aggregate_writes = aggregate_writes(&store, &report, &spec, 63.0);
        store
            .retrieval()
            .upsert_embeddings(&spec, &aggregate_writes)
            .unwrap();
        let target_write = aggregate_writes.last().unwrap().clone();
        let target_before = stored_embedding_unchecked(
            &Connection::open(store.retrieval().index_path()).unwrap(),
            &target_write.document_id,
        );

        let mut wrong_session = session.clone();
        wrong_session.id = "wrong-embedded-session".into();
        let wrong_bytes = serde_json::to_vec_pretty(&wrong_session).unwrap();
        let wrong_hash = bytes_sha256(&wrong_bytes);
        fs::write(&path, wrong_bytes).unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "UPDATE indexed_sessions SET source_sha256=?1 WHERE session_id=?2",
                params![wrong_hash, session.id],
            )
            .unwrap();
        connection
            .execute(
                "UPDATE memory_episode_materializations SET source_session_sha256=?1
                 WHERE session_id=?2",
                params![wrong_hash, session.id],
            )
            .unwrap();

        let replacement = target_write.clone();
        assert!(matches!(
            store.retrieval().upsert_embeddings(&spec, &[replacement]),
            Err(RetrievalError::CorruptIndex(_))
        ));
        assert_eq!(
            stored_embedding_unchecked(&connection, &target_write.document_id),
            target_before
        );
        assert!(matches!(
            store.retrieval().compatible_embeddings(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
        assert!(matches!(
            store.retrieval().embedding_coverage(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
        fs::write(&path, source_before).unwrap();
    }

    #[test]
    fn episode_aggregate_source_projection_rejects_hash_coherent_content_replacement() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        session.turns.push(Turn::pending("original content".into()));
        let path = store.save(&mut session).unwrap();
        let config = aggregate_memory_config();
        let (spec, report) = materialize_with_canonical_embeddings(&store, &session, &config);
        let aggregate_writes = aggregate_writes(&store, &report, &spec, 67.0);
        store
            .retrieval()
            .upsert_embeddings(&spec, &aggregate_writes)
            .unwrap();
        let target_write = aggregate_writes.last().unwrap().clone();

        let mut replacement = session.clone();
        replacement.turns[0].user_content = "replacement content".into();
        replacement.validate().unwrap();
        replacement.refresh_cumulative_usage();
        let mut replacement_bytes = serde_json::to_vec_pretty(&replacement).unwrap();
        replacement_bytes.push(b'\n');
        let replacement_hash = bytes_sha256(&replacement_bytes);
        fs::write(&path, &replacement_bytes).unwrap();

        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "UPDATE indexed_sessions SET source_sha256=?1 WHERE session_id=?2",
                params![replacement_hash, session.id],
            )
            .unwrap();
        connection
            .execute(
                "UPDATE memory_episode_materializations SET source_session_sha256=?1
                 WHERE session_id=?2",
                params![replacement_hash, session.id],
            )
            .unwrap();
        let before = episode_materialization_sql_state(&connection, &session.id);
        let target_before = stored_embedding_unchecked(&connection, &target_write.document_id);

        assert!(matches!(
            store
                .retrieval()
                .upsert_embeddings(&spec, std::slice::from_ref(&target_write)),
            Err(RetrievalError::CorruptIndex(_))
        ));
        assert!(matches!(
            store.retrieval().compatible_embeddings(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
        assert!(matches!(
            store
                .retrieval()
                .materialize_episode_documents(&session.id, &config),
            Err(RetrievalError::CorruptIndex(_))
        ));

        assert_eq!(
            episode_materialization_sql_state(&connection, &session.id),
            before
        );
        assert_eq!(
            stored_embedding_unchecked(&connection, &target_write.document_id),
            target_before
        );
        assert_eq!(fs::read(path).unwrap(), replacement_bytes);
    }

    fn apply_aggregate_audit_tamper(
        connection: &Connection,
        session_id: &str,
        tamper: AggregateAuditTamper,
    ) {
        let fake_hash = "0".repeat(64);
        match tamper {
            AggregateAuditTamper::LedgerSnapshot => {
                connection.execute("UPDATE memory_episode_materializations SET ledger_snapshot_sha256=?1 WHERE session_id=?2", params![fake_hash, session_id]).unwrap();
            }
            AggregateAuditTamper::PlanInput => {
                connection.execute("UPDATE memory_episode_materializations SET plan_input_sha256=?1 WHERE session_id=?2", params![fake_hash, session_id]).unwrap();
            }
            AggregateAuditTamper::AlgorithmVersion => {
                connection
                    .execute_batch("PRAGMA ignore_check_constraints=ON;")
                    .unwrap();
                connection.execute("UPDATE memory_episode_materializations SET algorithm_version=2 WHERE session_id=?1", [session_id]).unwrap();
            }
            AggregateAuditTamper::GapMinutes => {
                connection.execute("UPDATE memory_episode_materializations SET gap_minutes=gap_minutes+1 WHERE session_id=?1", [session_id]).unwrap();
            }
            AggregateAuditTamper::TopicThreshold => {
                connection
                    .execute_batch("PRAGMA ignore_check_constraints=ON;")
                    .unwrap();
                connection.execute("UPDATE memory_episode_materializations SET topic_similarity_threshold=0.59 WHERE session_id=?1", [session_id]).unwrap();
            }
            AggregateAuditTamper::EpisodeCount => {
                connection.execute("UPDATE memory_episode_materializations SET episode_count=episode_count+1 WHERE session_id=?1", [session_id]).unwrap();
            }
            AggregateAuditTamper::BoundaryCount => {
                connection.execute("UPDATE memory_episode_materializations SET boundary_count=boundary_count+1 WHERE session_id=?1", [session_id]).unwrap();
            }
            AggregateAuditTamper::MaterializedAt => {
                connection.execute("UPDATE memory_episode_materializations SET materialized_at='not-a-time' WHERE session_id=?1", [session_id]).unwrap();
            }
            AggregateAuditTamper::BoundaryDecision => {
                let (before_event_id, json): (String, String) = connection.query_row(
                    "SELECT before_event_id, decision_json FROM memory_episode_boundaries WHERE session_id=?1 ORDER BY before_event_id LIMIT 1",
                    [session_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                ).unwrap();
                let mut decision: EpisodeBoundaryDecision = serde_json::from_str(&json).unwrap();
                decision.is_boundary = !decision.is_boundary;
                connection.execute(
                    "UPDATE memory_episode_boundaries SET decision_json=?1 WHERE session_id=?2 AND before_event_id=?3",
                    params![serde_json::to_string(&decision).unwrap(), session_id, before_event_id],
                ).unwrap();
            }
            AggregateAuditTamper::BoundaryInputHash => {
                let (before_event_id, json): (String, String) = connection.query_row(
                    "SELECT before_event_id, decision_json FROM memory_episode_boundaries WHERE session_id=?1 ORDER BY before_event_id LIMIT 1",
                    [session_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                ).unwrap();
                let mut decision: EpisodeBoundaryDecision = serde_json::from_str(&json).unwrap();
                decision.input_sha256 = fake_hash.clone();
                connection.execute(
                    "UPDATE memory_episode_boundaries SET decision_json=?1,input_sha256=?2 WHERE session_id=?3 AND before_event_id=?4",
                    params![serde_json::to_string(&decision).unwrap(), fake_hash, session_id, before_event_id],
                ).unwrap();
            }
            AggregateAuditTamper::AggregateRange => {
                connection.execute(
                    "UPDATE memory_documents SET start_sequence=start_sequence+1
                     WHERE document_id=(SELECT document_id FROM memory_documents WHERE session_id=?1 AND granularity='episode' ORDER BY document_id LIMIT 1)",
                    [session_id],
                ).unwrap();
            }
            AggregateAuditTamper::AggregateMember => {
                connection.execute(
                    "UPDATE memory_document_members SET content_sha256=?1
                     WHERE document_id=(SELECT document_id FROM memory_documents WHERE session_id=?2 AND granularity='episode' ORDER BY document_id LIMIT 1)
                       AND ordinal=0",
                    params![fake_hash, session_id],
                ).unwrap();
            }
            AggregateAuditTamper::AggregateSource => {
                connection.execute(
                    "UPDATE memory_documents SET source_sha256=?1
                     WHERE document_id=(SELECT document_id FROM memory_documents WHERE session_id=?2 AND granularity='episode' ORDER BY document_id LIMIT 1)",
                    params![fake_hash, session_id],
                ).unwrap();
            }
            AggregateAuditTamper::CoherentCatalog => {
                let original: String = connection.query_row(
                    "SELECT document_id FROM memory_documents WHERE session_id=?1 AND granularity='episode' ORDER BY document_id LIMIT 1",
                    [session_id],
                    |row| row.get(0),
                ).unwrap();
                connection.execute(
                    "INSERT INTO memory_documents(document_id,session_id,granularity,source_sha256,start_sequence,end_sequence,member_count)
                     SELECT 'episode_coherent_tamper',session_id,granularity,source_sha256,start_sequence,end_sequence,member_count
                     FROM memory_documents WHERE document_id=?1",
                    [&original],
                ).unwrap();
                connection.execute(
                    "INSERT INTO memory_document_members(document_id,ordinal,event_id,start_char,end_char,content_sha256)
                     SELECT 'episode_coherent_tamper',ordinal,event_id,start_char,end_char,content_sha256
                     FROM memory_document_members WHERE document_id=?1 ORDER BY ordinal",
                    [&original],
                ).unwrap();
                connection.execute(
                    "UPDATE memory_episode_materializations
                     SET episode_count=episode_count+1, ledger_snapshot_sha256=?1, plan_input_sha256=?2
                     WHERE session_id=?3",
                    params![fake_hash, "1".repeat(64), session_id],
                ).unwrap();
            }
            AggregateAuditTamper::RawSource => {}
        }
    }

    fn stored_embedding_unchecked(connection: &Connection, document_id: &str) -> StoredEmbedding {
        connection
            .query_row(
                "SELECT d.session_id,d.granularity,d.source_sha256,e.model,e.dimensions,
                    e.index_fingerprint,e.vector_blob,e.embedded_at
             FROM memory_documents d JOIN memory_embeddings e ON e.document_id=d.document_id
             WHERE d.document_id=?1",
                [document_id],
                |row| {
                    let dimensions = i64_to_usize(row.get(4)?)?;
                    let bytes = row.get::<_, Vec<u8>>(6)?;
                    Ok(StoredEmbedding {
                        document_id: document_id.to_owned(),
                        session_id: row.get(0)?,
                        granularity: parse_memory_granularity(&row.get::<_, String>(1)?)
                            .map_err(|_| rusqlite::Error::InvalidQuery)?,
                        source_sha256: row.get(2)?,
                        model: row.get(3)?,
                        dimensions,
                        index_fingerprint: row.get(5)?,
                        vector: decode_f32_le(&bytes, dimensions)
                            .map_err(|_| rusqlite::Error::InvalidQuery)?,
                        embedded_at: row.get(7)?,
                    })
                },
            )
            .unwrap()
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum EpisodeSqlCell {
        Null,
        Integer(i64),
        Real(u64),
        Text(String),
        Blob(Vec<u8>),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct EpisodeMaterializationSqlState {
        catalog: Vec<Vec<EpisodeSqlCell>>,
        members: Vec<Vec<EpisodeSqlCell>>,
        boundaries: Vec<Vec<EpisodeSqlCell>>,
        materialization: Vec<Vec<EpisodeSqlCell>>,
        embeddings: Vec<Vec<EpisodeSqlCell>>,
    }

    fn episode_sql_rows(
        connection: &Connection,
        query: &str,
        session_id: &str,
    ) -> Vec<Vec<EpisodeSqlCell>> {
        let mut statement = connection.prepare(query).unwrap();
        let column_count = statement.column_count();
        statement
            .query_map([session_id], |row| {
                (0..column_count)
                    .map(|index| {
                        Ok(match row.get_ref(index)? {
                            rusqlite::types::ValueRef::Null => EpisodeSqlCell::Null,
                            rusqlite::types::ValueRef::Integer(value) => {
                                EpisodeSqlCell::Integer(value)
                            }
                            rusqlite::types::ValueRef::Real(value) => {
                                EpisodeSqlCell::Real(value.to_bits())
                            }
                            rusqlite::types::ValueRef::Text(value) => {
                                EpisodeSqlCell::Text(String::from_utf8_lossy(value).into_owned())
                            }
                            rusqlite::types::ValueRef::Blob(value) => {
                                EpisodeSqlCell::Blob(value.to_vec())
                            }
                        })
                    })
                    .collect::<rusqlite::Result<Vec<_>>>()
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn episode_materialization_sql_state(
        connection: &Connection,
        session_id: &str,
    ) -> EpisodeMaterializationSqlState {
        EpisodeMaterializationSqlState {
            catalog: episode_sql_rows(
                connection,
                "SELECT document_id,session_id,granularity,source_sha256,start_sequence,
                        end_sequence,member_count
                 FROM memory_documents WHERE session_id=?1
                   AND granularity IN ('episode','session') ORDER BY document_id",
                session_id,
            ),
            members: episode_sql_rows(
                connection,
                "SELECT m.document_id,m.ordinal,m.event_id,m.start_char,m.end_char,
                        m.content_sha256
                 FROM memory_document_members m
                 JOIN memory_documents d ON d.document_id=m.document_id
                 WHERE d.session_id=?1 AND d.granularity IN ('episode','session')
                 ORDER BY m.document_id,m.ordinal",
                session_id,
            ),
            boundaries: episode_sql_rows(
                connection,
                "SELECT session_id,before_event_id,decision_json,input_sha256
                 FROM memory_episode_boundaries WHERE session_id=?1 ORDER BY before_event_id",
                session_id,
            ),
            materialization: episode_sql_rows(
                connection,
                "SELECT session_id,source_session_sha256,ledger_snapshot_sha256,
                        vector_index_fingerprint,plan_input_sha256,algorithm_version,gap_minutes,
                        topic_similarity_threshold,episode_count,boundary_count,materialized_at
                 FROM memory_episode_materializations WHERE session_id=?1",
                session_id,
            ),
            embeddings: episode_sql_rows(
                connection,
                "SELECT e.document_id,e.model,e.dimensions,e.source_sha256,
                        e.index_fingerprint,e.vector_blob,e.embedded_at
                 FROM memory_embeddings e
                 JOIN memory_documents d ON d.document_id=e.document_id
                 WHERE d.session_id=?1 AND d.granularity IN ('episode','session')
                 ORDER BY e.document_id",
                session_id,
            ),
        }
    }

    fn consolidation_quote_nth(
        event: &ConsolidationEvent,
        needle: &str,
        occurrence: usize,
    ) -> ConsolidationQuote {
        let mut offset = 0_usize;
        let mut found = None;
        for _ in 0..=occurrence {
            let relative = event.content[offset..].find(needle).unwrap();
            let start_byte = offset + relative;
            found = Some(start_byte);
            offset = start_byte + needle.len();
        }
        let start_byte = found.unwrap();
        let start_char = event.content[..start_byte].chars().count();
        ConsolidationQuote {
            event_id: event.event_id.clone(),
            start_char,
            end_char: start_char + needle.chars().count(),
            content_sha256: content_sha256(needle),
        }
    }

    fn full_consolidation_quote(event: &ConsolidationEvent) -> ConsolidationQuote {
        ConsolidationQuote {
            event_id: event.event_id.clone(),
            start_char: 0,
            end_char: event.content.chars().count(),
            content_sha256: content_sha256(&event.content),
        }
    }

    fn seed_episode_projection(store: &SessionStore, session_id: &str) {
        let batch = store
            .retrieval()
            .next_consolidation_batch(session_id)
            .unwrap()
            .unwrap();
        let first_user = batch
            .events
            .iter()
            .find(|event| event.content == "Alice aka Al likes tea.")
            .unwrap();
        let boundary_user = batch
            .events
            .iter()
            .find(|event| event.content == "change topic now.")
            .unwrap();
        let mut entity = ConsolidatedEntityOutput {
            local_id: "local_alice".into(),
            name: "Alice".into(),
            kind: MemoryEntityKind::Person,
            resolution: EntityResolution::New,
            disambiguation: EntityDisambiguation::Resolved,
            basis: EntityResolutionBasis::FirstMention,
            existing_entity_id: None,
            name_evidence: consolidation_quote_nth(first_user, "Alice", 0),
            existing_identity_evidence: None,
            resolution_evidence: None,
            aliases: Vec::new(),
        };
        entity.aliases.push(EntityAliasOutput {
            text: "Al".into(),
            kind: MemoryAliasKind::ExplicitAlias,
            stable_identifier_kind: None,
            evidence: consolidation_quote_nth(first_user, "Al", 1),
            proof_evidence: full_consolidation_quote(first_user),
        });
        let subject = consolidation_quote_nth(first_user, "Alice", 0);
        let relation = consolidation_quote_nth(first_user, "likes", 0);
        let object = consolidation_quote_nth(first_user, "tea", 0);
        let output = StructuredConsolidationOutput {
            entities: vec![entity],
            claims: vec![ConsolidatedClaimOutput {
                local_id: "local_preference".into(),
                subject_ref: "local_alice".into(),
                predicate_key: "preference.drink".into(),
                object: ConsolidatedClaimObject {
                    kind: ConsolidationClaimObjectKind::Text,
                    text: Some("tea".into()),
                    entity_ref: None,
                    span: Some(object.clone()),
                },
                polarity: ClaimPolarity::Assert,
                cardinality: ClaimCardinality::Single,
                certainty: ClaimCertainty::Certain,
                disposition: ClaimDisposition::New,
                replaces_claim_ids: Vec::new(),
                conflicts_with_claim_ids: Vec::new(),
                event_time: None,
                valid_from: None,
                valid_to: None,
                evidence: vec![ConsolidationClaimEvidence {
                    kind: ConsolidationEvidenceKind::Assertion,
                    quote: full_consolidation_quote(first_user),
                    subject_span: subject,
                    relation_span: relation,
                    object_span: object,
                    speech_act_span: None,
                }],
            }],
            boundaries: vec![ConsolidationBoundaryOutput {
                before_event_id: boundary_user.event_id.clone(),
                reason: BoundarySuggestionReason::ExplicitTopicTransition,
                evidence: vec![consolidation_quote_nth(boundary_user, "change topic", 0)],
            }],
        };
        let candidates = store.retrieval().consolidation_candidates(1, 1).unwrap();
        assert!(candidates.entities.is_empty());
        assert!(candidates.claims.is_empty());
        let request_json = serde_json::to_string(
            &canonical_consolidation_request(
                "fixture-model".into(),
                &batch,
                &candidates,
                4096,
                1024,
            )
            .unwrap(),
        )
        .unwrap();
        let response_json = serde_json::to_string(&output).unwrap();
        let started_at = batch.events.last().unwrap().created_at.clone();
        let completed_at = (DateTime::parse_from_rfc3339(&started_at).unwrap()
            + chrono::Duration::seconds(1))
        .to_rfc3339();
        let attempt = ConsolidationAttemptRecord {
            attempt_id: format!("projection-attempt-{session_id}"),
            batch_key: batch.batch_key.clone(),
            session_id: batch.session_id.clone(),
            from_sequence: batch.from_sequence,
            through_sequence: batch.through_sequence,
            trigger: "test".into(),
            model: "fixture-model".into(),
            request_sha256: content_sha256(&request_json),
            request_json,
            input_event_ids: batch
                .events
                .iter()
                .map(|event| event.event_id.clone())
                .collect(),
            input_event_hashes: batch
                .events
                .iter()
                .map(|event| event.content_sha256.clone())
                .collect(),
            response_sha256: Some(content_sha256(&response_json)),
            response_json: Some(response_json),
            status: ConsolidationAttemptStatus::Applied,
            input_tokens: Some(1),
            output_tokens: Some(1),
            latency_ms: 1,
            started_at,
            completed_at,
            validation_json: Some("{\"valid\":true}".into()),
            error_json: None,
        };
        store
            .retrieval()
            .apply_consolidation_attempt(&batch, &candidates, &attempt)
            .unwrap();
    }

    fn aggregate_memory_config() -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            ..Default::default()
        }
    }

    fn canonical_message_writes(
        store: &SessionStore,
        session: &Session,
        spec: &VectorIndexSpec,
        value: f32,
    ) -> Vec<EmbeddingWrite> {
        store
            .retrieval()
            .replay_session(&session.id)
            .unwrap()
            .into_iter()
            .filter(|event| matches!(event.role, EventRole::User | EventRole::Assistant))
            .enumerate()
            .map(|(index, event)| EmbeddingWrite {
                document_id: format!("{}:0:{}", event.id, event.content.chars().count()),
                expected_source_sha256: content_sha256(&event.content),
                vector: vec![value + index as f32; spec.dimensions],
            })
            .collect()
    }

    fn materialize_with_canonical_embeddings(
        store: &SessionStore,
        session: &Session,
        config: &MemoryConfig,
    ) -> (VectorIndexSpec, EpisodeMaterializationReport) {
        store
            .retrieval()
            .materialize_episode_documents(&session.id, config)
            .unwrap();
        let spec = VectorIndexSpec::from_config(config).unwrap();
        let leaves = canonical_message_writes(store, session, &spec, 1.0);
        store.retrieval().upsert_embeddings(&spec, &leaves).unwrap();
        let report = store
            .retrieval()
            .materialize_episode_documents(&session.id, config)
            .unwrap();
        (spec, report)
    }

    fn aggregate_writes(
        store: &SessionStore,
        report: &EpisodeMaterializationReport,
        spec: &VectorIndexSpec,
        value: f32,
    ) -> Vec<EmbeddingWrite> {
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let fingerprint = spec.fingerprint().unwrap();
        let (messages, _, _, _) =
            load_episode_snapshot(&connection, &report.session_id, spec, &fingerprint).unwrap();
        let complete_message_coverage = messages.iter().all(|message| message.embedding.is_some());
        let direct_message_embeddings = messages
            .into_iter()
            .filter_map(|message| {
                message
                    .embedding
                    .map(|embedding| (message.member.document_id, embedding))
            })
            .collect::<HashMap<_, _>>();
        let documents = aggregate_documents_for_plan(report);
        if !complete_message_coverage {
            return documents
                .into_iter()
                .enumerate()
                .map(|(index, document)| EmbeddingWrite {
                    document_id: document.document_id,
                    expected_source_sha256: document.source_sha256,
                    vector: vec![value + index as f32; spec.dimensions],
                })
                .collect();
        }
        let canonical_vector_blobs = canonical_aggregate_vector_blobs(
            &documents,
            &direct_message_embeddings,
            spec.dimensions,
        )
        .unwrap();
        documents
            .into_iter()
            .map(|document| EmbeddingWrite {
                vector: decode_f32_le(
                    canonical_vector_blobs.get(&document.document_id).unwrap(),
                    spec.dimensions,
                )
                .unwrap(),
                document_id: document.document_id,
                expected_source_sha256: document.source_sha256,
            })
            .collect()
    }

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
            untrusted_history_wrapped: plan.untrusted_history_wrapped,
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
    fn rebuild_preserves_unknown_future_index_without_mutation() {
        let root = tempfile::tempdir().unwrap();
        let index = root.path().join(INDEX_FILENAME);
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE future_sentinel (
                     sentinel_key TEXT PRIMARY KEY,
                     sentinel_value BLOB NOT NULL
                 );
                 INSERT INTO future_sentinel VALUES ('keep-me', X'00FF1020');
                 CREATE TABLE consolidation_watermarks (
                     session_id TEXT PRIMARY KEY,
                     through_sequence INTEGER NOT NULL,
                     through_event_id TEXT,
                     through_event_sha256 TEXT,
                     updated_at TEXT,
                     future_column TEXT NOT NULL
                 );
                 INSERT INTO consolidation_watermarks VALUES
                     ('future-session', 42, 'future-event', 'future-hash',
                      '2026-01-01T00:00:00Z', 'future-watermark');
                 CREATE TABLE consolidation_batches (
                     attempt_id TEXT PRIMARY KEY,
                     status TEXT NOT NULL,
                     future_payload TEXT NOT NULL
                 );
                 INSERT INTO consolidation_batches VALUES
                     ('future-attempt', 'future-applied', '{\"future\":true}');
                 PRAGMA user_version=8;",
            )
            .unwrap();
        drop(connection);
        let original_bytes = fs::read(&index).unwrap();

        let store = RetrievalStore::new(root.path()).unwrap();
        assert!(matches!(
            store.rebuild(),
            Err(RetrievalError::UnsupportedIndexVersion(8))
        ));
        assert!(index.is_file());
        assert_eq!(fs::read(&index).unwrap(), original_bytes);

        let connection = Connection::open(&index).unwrap();
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            8
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT sentinel_key, hex(sentinel_value) FROM future_sentinel",
                    [],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                )
                .unwrap(),
            ("keep-me".to_owned(), "00FF1020".to_owned())
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT session_id, through_sequence, through_event_id,
                            through_event_sha256, updated_at, future_column
                     FROM consolidation_watermarks",
                    [],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                            row.get::<_, String>(5)?,
                        ))
                    }
                )
                .unwrap(),
            (
                "future-session".to_owned(),
                42,
                "future-event".to_owned(),
                "future-hash".to_owned(),
                "2026-01-01T00:00:00Z".to_owned(),
                "future-watermark".to_owned(),
            )
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT attempt_id, status, future_payload FROM consolidation_batches",
                    [],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    }
                )
                .unwrap(),
            (
                "future-attempt".to_owned(),
                "future-applied".to_owned(),
                "{\"future\":true}".to_owned(),
            )
        );
    }

    #[test]
    fn migrates_real_v4_vector_storage_transactionally() {
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
        let expected_context = store.retrieval().answer_context(&answer_id).unwrap();
        {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            connection
                .execute(
                    "INSERT INTO consolidation_watermarks
                     (session_id, through_sequence, through_event_id, through_event_sha256, updated_at)
                     VALUES (?1, 1, 'event', 'hash', '2026-01-01T00:00:00Z')",
                    [&session.id],
                )
                .unwrap();
            connection
                .execute_batch(
                    "DROP TABLE memory_embeddings;
                     DROP TABLE memory_document_members;
                     DROP TABLE memory_documents;
                     PRAGMA user_version=4;",
                )
                .unwrap();
        }

        let migrated = RetrievalStore::new(root.path()).unwrap();
        let connection = migrated.open_connection().unwrap();
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            INDEX_SCHEMA_VERSION
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM memory_documents", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            connection
                .query_row("SELECT count(*) FROM retrieval_documents", [], |row| row
                    .get::<_, i64>(0))
                .unwrap()
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM memory_document_members", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            connection
                .query_row("SELECT count(*) FROM retrieval_documents", [], |row| row
                    .get::<_, i64>(0))
                .unwrap()
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT member_count, start_sequence=end_sequence, source_sha256=content_sha256
                     FROM memory_documents m
                     JOIN retrieval_documents d ON d.document_id=m.document_id
                     ORDER BY m.document_id LIMIT 1",
                    [],
                    |row| Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?
                    )),
                )
                .unwrap(),
            (1, 1, 1)
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT mm.ordinal, mm.event_id=d.event_id,
                            mm.start_char=d.start_char, mm.end_char=d.end_char,
                            mm.content_sha256=d.content_sha256
                     FROM memory_document_members mm
                     JOIN retrieval_documents d ON d.document_id=mm.document_id
                     ORDER BY mm.document_id LIMIT 1",
                    [],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, i64>(2)?,
                            row.get::<_, i64>(3)?,
                            row.get::<_, i64>(4)?,
                        ))
                    },
                )
                .unwrap(),
            (0, 1, 1, 1, 1)
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT through_sequence FROM consolidation_watermarks WHERE session_id=?1",
                    [&session.id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
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
        assert_eq!(
            migrated.answer_context(&answer_id).unwrap().context_sha256,
            expected_context.context_sha256
        );
    }

    #[test]
    fn v4_vector_backfill_is_idempotent_and_preserves_compatible_embeddings() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "保留向量的原文", "回复", "");
        store.save(&mut session).unwrap();
        let spec = VectorIndexSpec {
            model: "qwen3-embedding:8b".into(),
            dimensions: 32,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
        };
        let (document_id, source_sha256): (String, String) = store
            .retrieval()
            .open_connection()
            .unwrap()
            .query_row(
                "SELECT document_id, source_sha256 FROM memory_documents ORDER BY document_id LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let write = EmbeddingWrite {
            document_id: document_id.clone(),
            expected_source_sha256: source_sha256,
            vector: vec![0.25; 32],
        };
        store
            .retrieval()
            .upsert_embeddings(&spec, std::slice::from_ref(&write))
            .unwrap();

        for _ in 0..2 {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            connection
                .pragma_update(None, "user_version", 4_i64)
                .unwrap();
            drop(connection);

            let migrated = RetrievalStore::new(root.path()).unwrap();
            assert_eq!(migrated.compatible_embeddings(&spec).unwrap().len(), 1);
            let connection = migrated.open_connection().unwrap();
            assert_eq!(
                connection
                    .query_row("SELECT count(*) FROM memory_documents", [], |row| row
                        .get::<_, i64>(0))
                    .unwrap(),
                connection
                    .query_row("SELECT count(*) FROM retrieval_documents", [], |row| row
                        .get::<_, i64>(0))
                    .unwrap()
            );
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM memory_documents d
                         LEFT JOIN memory_document_members m ON m.document_id=d.document_id
                         GROUP BY d.document_id HAVING count(m.ordinal) != d.member_count",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .optional()
                    .unwrap(),
                None
            );
        }
    }

    #[test]
    fn source_provenance_deletion_cleans_catalog_and_embeddings() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, &"用".repeat(241), "助手原文", "");
        store.save(&mut session).unwrap();
        let spec = VectorIndexSpec {
            model: "qwen3-embedding:8b".into(),
            dimensions: 32,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
        };
        let connection = store.retrieval().open_connection().unwrap();
        let mut statement = connection
            .prepare(
                "SELECT d.document_id, d.source_sha256, m.event_id, m.start_char, m.end_char, e.role,
                        d.granularity
                 FROM memory_documents d
                 JOIN memory_document_members m ON m.document_id=d.document_id
                 JOIN events e ON e.event_id=m.event_id
                 ORDER BY e.sequence, d.document_id",
            )
            .unwrap();
        let entries = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                ))
            })
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        drop(statement);
        drop(connection);
        let user = entries
            .iter()
            .find(|entry| entry.5 == "user" && entry.6 == "fragment")
            .unwrap();
        let assistant = entries.iter().find(|entry| entry.5 == "assistant").unwrap();
        let writes = [
            EmbeddingWrite {
                document_id: user.0.clone(),
                expected_source_sha256: user.1.clone(),
                vector: vec![0.1; 32],
            },
            EmbeddingWrite {
                document_id: assistant.0.clone(),
                expected_source_sha256: assistant.1.clone(),
                vector: vec![0.2; 32],
            },
        ];
        store.retrieval().upsert_embeddings(&spec, &writes).unwrap();

        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        connection
            .execute(
                "DELETE FROM source_spans WHERE event_id=?1 AND start_char=?2 AND end_char=?3",
                params![user.2, user.3, user.4],
            )
            .unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_documents WHERE document_id=?1",
                    [&user.0],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_embeddings WHERE document_id=?1",
                    [&user.0],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        connection
            .execute("DELETE FROM events WHERE event_id=?1", [&assistant.2])
            .unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_documents WHERE document_id=?1",
                    [&assistant.0],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_embeddings WHERE document_id=?1",
                    [&assistant.0],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM memory_documents d
                     LEFT JOIN memory_document_members m ON m.document_id=d.document_id
                     GROUP BY d.document_id HAVING count(m.ordinal) != d.member_count",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .optional()
                .unwrap(),
            None
        );
    }

    #[test]
    fn vector_storage_sync_rebuild_and_embedding_guards() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(
            &mut session,
            &format!("{}证据", "字".repeat(241)),
            "回复",
            "",
        );
        store.save(&mut session).unwrap();
        let spec = VectorIndexSpec {
            model: "qwen3-embedding:8b".into(),
            dimensions: 32,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
        };
        let connection = store.retrieval().open_connection().unwrap();
        let (document_id, source_sha256): (String, String) = connection
            .query_row(
                "SELECT document_id, source_sha256 FROM memory_documents ORDER BY document_id LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let lexical_count = connection
            .query_row("SELECT count(*) FROM retrieval_documents", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        let catalog_count = connection
            .query_row("SELECT count(*) FROM memory_documents", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        assert_eq!(catalog_count, lexical_count);
        drop(connection);

        let mut vector = vec![0.5; 32];
        vector[0] = 1.0;
        vector[1] = -0.0;
        let write = EmbeddingWrite {
            document_id: document_id.clone(),
            expected_source_sha256: source_sha256.clone(),
            vector,
        };
        store
            .retrieval()
            .upsert_embeddings(&spec, std::slice::from_ref(&write))
            .unwrap();
        assert_eq!(
            store
                .retrieval()
                .embedding_coverage(&spec)
                .unwrap()
                .compatible,
            1
        );
        let loaded = store.retrieval().compatible_embeddings(&spec).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].vector[1].to_bits(), (-0.0_f32).to_bits());
        store.save(&mut session).unwrap();
        assert_eq!(
            store
                .retrieval()
                .embedding_coverage(&spec)
                .unwrap()
                .compatible,
            1
        );
        assert_eq!(
            store
                .retrieval()
                .compatible_embeddings(&spec)
                .unwrap()
                .len(),
            1
        );
        append_complete_turn(&mut session, "追加问题", "追加回复", "");
        store.save(&mut session).unwrap();
        let after_append = store.retrieval().embedding_coverage(&spec).unwrap();
        assert_eq!(after_append.compatible, 1);
        assert!(after_append.total > after_append.compatible);
        assert_eq!(
            store
                .retrieval()
                .compatible_embeddings(&spec)
                .unwrap()
                .len(),
            1
        );

        let mut changed = spec.clone();
        changed.model = "different".into();
        assert_eq!(
            store
                .retrieval()
                .embedding_coverage(&changed)
                .unwrap()
                .compatible,
            0
        );
        changed = spec.clone();
        changed.dimensions = 64;
        assert_eq!(
            store
                .retrieval()
                .embedding_coverage(&changed)
                .unwrap()
                .compatible,
            0
        );
        changed = spec.clone();
        changed.hnsw_m = 24;
        assert_eq!(
            store
                .retrieval()
                .embedding_coverage(&changed)
                .unwrap()
                .compatible,
            0
        );
        changed = spec.clone();
        changed.hnsw_ef_construction = 300;
        assert_eq!(
            store
                .retrieval()
                .embedding_coverage(&changed)
                .unwrap()
                .compatible,
            0
        );
        changed = spec.clone();
        changed.hnsw_ef_search = 96;
        assert_eq!(
            store
                .retrieval()
                .embedding_coverage(&changed)
                .unwrap()
                .compatible,
            0
        );

        for invalid in [
            vec![EmbeddingWrite {
                expected_source_sha256: "wrong".into(),
                ..write.clone()
            }],
            vec![EmbeddingWrite {
                document_id: "unknown".into(),
                ..write.clone()
            }],
            vec![
                write.clone(),
                EmbeddingWrite {
                    document_id: "unknown".into(),
                    ..write.clone()
                },
            ],
            vec![EmbeddingWrite {
                vector: vec![1.0],
                ..write.clone()
            }],
            vec![EmbeddingWrite {
                vector: vec![f32::NAN, 0.0, 0.0],
                ..write.clone()
            }],
            vec![write.clone(), write.clone()],
        ] {
            assert!(
                store
                    .retrieval()
                    .upsert_embeddings(&spec, &invalid)
                    .is_err()
            );
            assert_eq!(
                store
                    .retrieval()
                    .compatible_embeddings(&spec)
                    .unwrap()
                    .len(),
                1
            );
        }

        {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            let mut corrupt_blob = f32::NAN.to_le_bytes().to_vec();
            corrupt_blob.resize(32 * std::mem::size_of::<f32>(), 0);
            connection
                .execute(
                    "UPDATE memory_embeddings SET vector_blob=?1 WHERE document_id=?2",
                    params![corrupt_blob, document_id],
                )
                .unwrap();
        }
        assert!(matches!(
            store.retrieval().compatible_embeddings(&spec),
            Err(RetrievalError::CorruptIndex(_))
        ));
        store.retrieval().rebuild().unwrap();
        let connection = store.retrieval().open_connection().unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM memory_embeddings", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM memory_documents", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            connection
                .query_row("SELECT count(*) FROM retrieval_documents", [], |row| row
                    .get::<_, i64>(0))
                .unwrap()
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
        let error = store.save(&mut b).unwrap_err();
        assert!(matches!(
            error
                .downcast_ref::<crate::store::IndexSyncAfterSourceCommit>()
                .map(|error| &error.source),
            Some(RetrievalError::InvalidSource { .. })
        ));
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
        assert!(store.save(&mut b).is_err());
        assert!(matches!(
            store.retrieval().rebuild(),
            Err(RetrievalError::InvalidSource { .. })
        ));
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

    #[test]
    fn lexically_equivalent_nonexistent_roots_share_the_same_lock() {
        use std::sync::mpsc;
        use std::thread;

        let parent = tempfile::tempdir().unwrap();
        let through_missing = parent.path().join("missing").join("..").join("root");
        let direct = parent.path().join("root");
        let expected = fs::canonicalize(parent.path()).unwrap().join("root");
        assert!(!direct.exists());
        let first = RetrievalStore::new(&through_missing).unwrap();
        assert!(expected.is_dir());
        let second = RetrievalStore::new(&direct).unwrap();
        assert_eq!(first.root(), expected);
        assert_eq!(second.root(), expected);
        assert_eq!(first.index_path(), expected.join(INDEX_FILENAME));
        assert!(Arc::ptr_eq(&first.root_lock, &second.root_lock));

        let guard = first.acquire_root_write().unwrap();
        let (sent, received) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _guard = second.acquire_root_write().unwrap();
            sent.send(()).unwrap();
        });
        assert!(received.recv_timeout(Duration::from_millis(100)).is_err());
        drop(guard);
        received.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_parent_nonexistent_roots_share_the_same_lock() {
        use std::os::unix::fs::symlink;
        use std::sync::mpsc;
        use std::thread;

        let parent = tempfile::tempdir().unwrap();
        let target_parent = parent.path().join("else");
        let target_child = target_parent.join("child");
        fs::create_dir_all(&target_child).unwrap();
        let link = parent.path().join("link");
        symlink(&target_child, &link).unwrap();

        let through_symlink_parent = link.join("..").join("shared");
        let direct = target_parent.join("shared");
        let expected = fs::canonicalize(&target_parent).unwrap().join("shared");
        assert!(!through_symlink_parent.exists());
        assert!(!direct.exists());
        let first = RetrievalStore::new(&through_symlink_parent).unwrap();
        assert!(expected.is_dir());
        let second = RetrievalStore::new(&direct).unwrap();
        assert_eq!(first.root(), expected);
        assert_eq!(second.root(), expected);
        assert_eq!(first.index_path(), expected.join(INDEX_FILENAME));
        assert!(Arc::ptr_eq(&first.root_lock, &second.root_lock));

        let guard = first.acquire_root_write().unwrap();
        let (sent, received) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _guard = second.acquire_root_write().unwrap();
            sent.send(()).unwrap();
        });
        assert!(received.recv_timeout(Duration::from_millis(100)).is_err());
        drop(guard);
        received.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn root_symlink_retarget_keeps_existing_store_pinned_to_original_target() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().unwrap();
        let target_a = parent.path().join("a");
        let target_b = parent.path().join("b");
        fs::create_dir_all(&target_a).unwrap();
        fs::create_dir_all(&target_b).unwrap();
        let canonical_a = fs::canonicalize(&target_a).unwrap();
        let canonical_b = fs::canonicalize(&target_b).unwrap();
        let link = parent.path().join("sessions");
        symlink(&target_a, &link).unwrap();

        let through_link = RetrievalStore::new(&link).unwrap();
        let direct_a = RetrievalStore::new(&target_a).unwrap();
        assert_eq!(through_link.root(), canonical_a);
        assert_eq!(through_link.index_path(), canonical_a.join(INDEX_FILENAME));
        assert!(Arc::ptr_eq(&through_link.root_lock, &direct_a.root_lock));

        fs::remove_file(&link).unwrap();
        symlink(&target_b, &link).unwrap();
        let direct_b = RetrievalStore::new(&target_b).unwrap();
        let fresh_link = RetrievalStore::new(&link).unwrap();
        assert_eq!(through_link.root(), canonical_a);
        assert_eq!(direct_b.root(), canonical_b);
        assert_eq!(fresh_link.root(), canonical_b);
        assert_eq!(direct_b.index_path(), canonical_b.join(INDEX_FILENAME));
        assert!(!Arc::ptr_eq(&through_link.root_lock, &direct_b.root_lock));
        assert!(Arc::ptr_eq(&fresh_link.root_lock, &direct_b.root_lock));

        let missing_link = parent.path().join("missing-sessions");
        symlink(&target_a, &missing_link).unwrap();
        let missing_root = missing_link.join("nested").join("sessions");
        let pinned_missing = RetrievalStore::new(&missing_root).unwrap();
        let pinned_target = canonical_a.join("nested").join("sessions");
        assert_eq!(pinned_missing.root(), pinned_target);
        assert!(pinned_target.is_dir());

        fs::remove_file(&missing_link).unwrap();
        symlink(&target_b, &missing_link).unwrap();
        pinned_missing.open_connection().unwrap();
        assert!(pinned_target.join(INDEX_FILENAME).is_file());
        assert!(
            !canonical_b
                .join("nested")
                .join("sessions")
                .join(INDEX_FILENAME)
                .exists()
        );

        let fresh_missing = RetrievalStore::new(&missing_root).unwrap();
        let fresh_target = canonical_b.join("nested").join("sessions");
        assert_eq!(fresh_missing.root(), fresh_target);
        assert!(!Arc::ptr_eq(
            &pinned_missing.root_lock,
            &fresh_missing.root_lock
        ));
        fresh_missing.open_connection().unwrap();
        assert!(fresh_target.join(INDEX_FILENAME).is_file());
    }

    #[test]
    fn retrieval_root_pins_missing_paths_and_rejects_regular_files() {
        let parent = tempfile::tempdir().unwrap();
        let missing = parent.path().join("missing").join("sessions");
        let expected = fs::canonicalize(parent.path())
            .unwrap()
            .join("missing")
            .join("sessions");
        let store = RetrievalStore::new(&missing).unwrap();
        assert_eq!(store.root(), expected);
        assert!(store.root().is_dir());
        assert!(!store.index_path().exists());
        store.open_connection().unwrap();
        assert!(store.index_path().is_file());

        let file = parent.path().join("not-a-directory");
        fs::write(&file, b"sentinel").unwrap();
        assert!(matches!(
            RetrievalStore::new(&file),
            Err(RetrievalError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::NotADirectory
        ));
        assert_eq!(fs::read(&file).unwrap(), b"sentinel");
    }

    #[test]
    fn memory_state_v1_rows_reset_atomically_but_raw_events_and_ledger_survive() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "legacy fact", "answer", "");
        store.save(&mut session).unwrap();
        let raw_counts = {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            (
                connection
                    .query_row("SELECT count(*) FROM events", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
                connection
                    .query_row("SELECT count(*) FROM source_spans", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
                connection
                    .query_row("SELECT count(*) FROM retrieval_documents", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
            )
        };
        {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            replace_with_old_memory_v1_schema(&connection);
            connection
                .execute_batch(
                    "INSERT INTO consolidation_batches
                     (attempt_id,batch_key,session_id,from_sequence,through_sequence,trigger,model,
                      request_json,request_sha256,input_event_ids,input_event_hashes,response_json,
                      response_sha256,status,input_tokens,output_tokens,latency_ms,started_at,
                      completed_at,validation_json,error_json)
                     VALUES ('old-attempt','old-batch','legacy-session',1,1,'test','model','{}',
                      '0000000000000000000000000000000000000000000000000000000000000000',
                      '[]','[]','{}',
                      '0000000000000000000000000000000000000000000000000000000000000000',
                      'applied',NULL,NULL,0,'2025-01-01T00:00:00Z','2025-01-01T00:00:01Z','{}',NULL);
                     INSERT INTO memory_entities
                     (entity_id,kind,canonical_name,normalized_name,disambiguation,
                      created_session_id,created_batch_key,created_event_id,created_start,
                      created_end,created_hash,created_at,updated_at)
                     VALUES ('old-entity','person','old','old','resolved','legacy-session',
                      'old-batch','old-event',0,3,
                      '0000000000000000000000000000000000000000000000000000000000000000',
                      '2025-01-01T00:00:01Z','2025-01-01T00:00:01Z');
                     INSERT INTO consolidation_watermarks
                     (session_id,through_sequence,through_event_id,through_event_sha256,updated_at)
                     VALUES ('legacy-session',1,'old-event',
                      '0000000000000000000000000000000000000000000000000000000000000000',
                      '2025-01-01T00:00:01Z');",
                )
                .unwrap();
        }

        let reopened = RetrievalStore::new(root.path()).unwrap();
        reopened.open_connection().unwrap();
        let connection = Connection::open(reopened.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT value FROM memory_schema_meta WHERE key='state_schema_version'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            4
        );
        assert_eq!(
            (
                connection
                    .query_row("SELECT count(*) FROM events", [], |row| row
                        .get::<_, i64>(0))
                    .unwrap(),
                connection
                    .query_row("SELECT count(*) FROM source_spans", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
                connection
                    .query_row("SELECT count(*) FROM retrieval_documents", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
            ),
            raw_counts
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM consolidation_batches", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT response_json FROM consolidation_batches WHERE attempt_id='old-attempt'",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "{}"
        );
        for table in [
            "memory_claim_evidence",
            "memory_claim_transitions",
            "memory_boundary_suggestions",
            "memory_claims",
            "memory_entity_aliases",
            "memory_entities",
            "consolidation_watermarks",
        ] {
            assert_eq!(
                connection
                    .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
                0,
                "{table} was not reset"
            );
        }
        let replay = reopened
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        assert_eq!(replay.watermark_before, 0);
        assert_ne!(replay.batch_key, "old-batch");
    }

    #[test]
    fn empty_real_memory_v1_schema_upgrades_to_v3_without_rows() {
        let root = tempfile::tempdir().unwrap();
        let store = RetrievalStore::new(root.path()).unwrap();
        store.open_connection().unwrap();
        let connection = Connection::open(store.index_path()).unwrap();
        replace_with_old_memory_v1_schema(&connection);
        drop(connection);

        store.open_connection().unwrap();
        let connection = Connection::open(store.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT value FROM memory_schema_meta WHERE key='state_schema_version'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            4
        );
        assert!(table_has_column(&connection, "memory_claims", "normalized_relation").unwrap());
        assert!(
            table_has_column(&connection, "memory_claim_evidence", "speech_act_sha256").unwrap()
        );
        assert!(table_has_column(&connection, "memory_claim_transitions", "ordinal").unwrap());
    }

    #[test]
    fn memory_state_v2_upgrades_to_v3_and_preserves_raw_events_and_ledger() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        append_complete_turn(&mut session, "v2 fact", "answer", "");
        store.save(&mut session).unwrap();
        let index_path = store.retrieval().index_path().to_owned();
        let (event_count, event_id, event_hash) = {
            let connection = Connection::open(&index_path).unwrap();
            let count = connection
                .query_row("SELECT count(*) FROM events", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap();
            let source = connection
                .query_row(
                    "SELECT event_id, content_sha256 FROM events WHERE role='user' LIMIT 1",
                    [],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .unwrap();
            (count, source.0, source.1)
        };
        {
            let connection = Connection::open(&index_path).unwrap();
            replace_transition_table_with_memory_v2_schema(&connection);
            connection
                .execute_batch(&format!(
                    "INSERT INTO consolidation_batches
                     (attempt_id,batch_key,session_id,from_sequence,through_sequence,trigger,model,
                      request_json,request_sha256,input_event_ids,input_event_hashes,response_json,
                      response_sha256,status,input_tokens,output_tokens,latency_ms,started_at,
                      completed_at,validation_json,error_json)
                     VALUES ('v2-attempt','v2-batch','v2-session',1,1,'test','model','{{}}',
                      '{zero}','[]','[]','{{}}','{zero}','applied',NULL,NULL,0,
                      '2025-01-01T00:00:00Z','2025-01-01T00:00:01Z','{{}}',NULL);
                     INSERT INTO memory_entities
                     (entity_id,kind,canonical_name,normalized_name,disambiguation,
                      created_session_id,created_batch_key,created_event_id,created_start,
                      created_end,created_hash,created_at,updated_at)
                     VALUES ('v2-entity','person','old','old','resolved','v2-session','v2-batch',
                      '{event_id}',0,1,'{event_hash}',
                      '2025-01-01T00:00:01Z','2025-01-01T00:00:01Z');
                     INSERT INTO consolidation_watermarks
                     (session_id,through_sequence,through_event_id,through_event_sha256,updated_at)
                     VALUES ('v2-session',1,'{event_id}','{event_hash}',
                      '2025-01-01T00:00:01Z');",
                    zero = "0".repeat(64),
                ))
                .unwrap();
        }

        let reopened = RetrievalStore::new(root.path()).unwrap();
        reopened.open_connection().unwrap();
        let connection = Connection::open(reopened.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT value FROM memory_schema_meta WHERE key='state_schema_version'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            4
        );
        assert!(table_has_column(&connection, "memory_claim_transitions", "ordinal").unwrap());
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM events", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            event_count
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM consolidation_batches WHERE attempt_id='v2-attempt'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        for table in [
            "memory_claim_evidence",
            "memory_claim_transitions",
            "memory_boundary_suggestions",
            "memory_claims",
            "memory_entity_aliases",
            "memory_entities",
            "consolidation_watermarks",
        ] {
            assert_eq!(
                connection
                    .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
                0,
                "{table} was not reset"
            );
        }
    }

    #[test]
    fn unknown_memory_state_version_fails_before_schema_mutation() {
        let root = tempfile::tempdir().unwrap();
        let store = RetrievalStore::new(root.path()).unwrap();
        store.open_connection().unwrap();
        let connection = Connection::open(store.index_path()).unwrap();
        connection
            .execute(
                "UPDATE memory_schema_meta SET value=99 WHERE key='state_schema_version'",
                [],
            )
            .unwrap();
        connection
            .execute_batch("CREATE TABLE memory_unknown_sentinel(value TEXT); INSERT INTO memory_unknown_sentinel VALUES ('keep');")
            .unwrap();
        drop(connection);

        assert!(matches!(
            store.open_connection(),
            Err(RetrievalError::UnsupportedMemoryStateVersion(99))
        ));
        let connection = Connection::open(store.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT value FROM memory_schema_meta", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            99
        );
        assert_eq!(
            connection
                .query_row("SELECT value FROM memory_unknown_sentinel", [], |row| {
                    row.get::<_, String>(0)
                })
                .unwrap(),
            "keep"
        );
    }
}
