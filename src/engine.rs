use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::config::AppConfig;
use crate::consolidation::{
    ConsolidationApplyError, ConsolidationAttemptRecord, ConsolidationAttemptStatus,
    ConsolidationRunReport, ConsolidationRunStatus, ConsolidationTrigger,
    StructuredConsolidationOutput, canonical_consolidation_request,
};
use crate::context::ContextAssembler;
use crate::knowledge::{KnowledgeRecall, KnowledgeTrace};
use crate::model::{
    BudgetBucket, BudgetExclusionTrace, BudgetProbeTrace, BudgetReflowTrace,
    BudgetStageLatencyTrace, BudgetTokenBreakdown, ChannelTrace, ChatEvent, ChatEventKind,
    ContextPlan, ContextTrace, EvidenceKind, ModelRequestTrace, ProvenanceQuality,
    RetrievalChannel, RetrievalDocumentGranularity, RetrievalTrace, Session, SessionStatus,
    TokenUsage, Turn, TurnStatus, content_sha256, utc_now,
};
#[cfg(test)]
use crate::ollama::StructuredChatRequest;
use crate::ollama::{ChatBackend, ChatRequest, EmbeddingRequest, OllamaError};
use crate::retrieval::{
    AggregateEmbeddingSnapshot, LeafEmbeddingSnapshot, PreparedQueryVectorRecall, RecallResult,
    RecalledEvidence, RetrievalError,
};
use crate::store::SessionStore;
use crate::vector::{
    EmbeddingWrite, VectorError, VectorIndexSpec, equal_mean, l2_normalize, weighted_pool,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitAction {
    ContinueWithTrim,
    EndSession,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparationStatus {
    Ready,
    LimitWarning,
    Blocked,
    Ended,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparationProgress {
    ExactContextCheckStarted {
        estimated_input_tokens: u64,
        probe_threshold: u64,
        input_budget: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingRefreshReport {
    pub leaf_documents: usize,
    pub leaf_reused: usize,
    pub leaf_embedded_inputs: usize,
    pub backend_batches: usize,
    pub aggregate_documents: usize,
    pub leaf_committed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsolidationProgress {
    AttemptStarted {
        session_id: String,
        attempt: usize,
        events: usize,
        from_sequence: usize,
        through_sequence: usize,
    },
    ValidationRetry {
        session_id: String,
        next_attempt: usize,
        max_attempts: usize,
    },
    BatchApplied {
        session_id: String,
        batches_applied: usize,
        watermark_after: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRefreshStage {
    LeafSnapshot,
    Planning,
    Backend,
    LeafPublish,
    Materialization,
    AggregateSnapshot,
    AggregatePublish,
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingRefreshError {
    #[error("记忆嵌入功能未启用")]
    Disabled,
    #[error("嵌入刷新在 {stage:?} 阶段被取消（批次 {batch:?}）")]
    Cancelled {
        stage: EmbeddingRefreshStage,
        batch: Option<usize>,
    },
    #[error("嵌入批次 {batch} 在 {timeout_secs} 秒后超时")]
    Timeout { batch: usize, timeout_secs: u64 },
    #[error("嵌入批次 {batch} 调用失败: {source}")]
    Backend {
        batch: usize,
        #[source]
        source: OllamaError,
    },
    #[error("嵌入批次 {batch} 返回无效响应: {message}")]
    InvalidResponse { batch: usize, message: String },
    #[error("嵌入配置无效: {source}")]
    InvalidConfig {
        #[source]
        source: VectorError,
    },
    #[error("文档 {document_id} 的向量无效: {source}")]
    Vector {
        document_id: String,
        #[source]
        source: VectorError,
    },
    #[error("嵌入刷新在 {stage:?} 阶段访问派生索引失败: {source}")]
    Retrieval {
        stage: EmbeddingRefreshStage,
        #[source]
        source: RetrievalError,
    },
    #[error("会话 {session_id} 的 episode materialization 失败: {source}")]
    Materialization {
        session_id: String,
        #[source]
        source: RetrievalError,
    },
    #[error("嵌入刷新在 {stage:?} 阶段的阻塞任务失败: {source}")]
    TaskJoin {
        stage: EmbeddingRefreshStage,
        #[source]
        source: tokio::task::JoinError,
    },
}

#[derive(Debug, Clone)]
pub struct PreparedTurn {
    pub session_id: String,
    pub turn_id: String,
    pub turn_index: usize,
    pub plan: ContextPlan,
    pub status: PreparationStatus,
    pub message: String,
    pub(crate) query_embedding: Option<Vec<f32>>,
}

impl PreparedTurn {
    pub fn ready(&self) -> bool {
        self.status == PreparationStatus::Ready
    }

    pub fn needs_limit_decision(&self) -> bool {
        self.status == PreparationStatus::LimitWarning
    }
}

#[derive(Debug, Clone)]
pub struct ChatEngine<B: ChatBackend> {
    store: SessionStore,
    client: B,
    assembler: ContextAssembler,
    config: AppConfig,
    background_embedding_gates: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    background_embedding_pending: Arc<AtomicUsize>,
    background_embedding_notify: Arc<Notify>,
    background_embedding_cancellation: Arc<Mutex<CancellationToken>>,
    #[cfg(test)]
    live_search_barrier: Arc<Mutex<Option<Arc<tokio::sync::Barrier>>>>,
    #[cfg(test)]
    live_search_fast_gate: Arc<Mutex<Option<Arc<Notify>>>>,
    #[cfg(test)]
    live_search_semantic_prepared: Arc<Mutex<Option<Arc<Notify>>>>,
}

struct StreamSnapshot<'a> {
    thinking: &'a str,
    content: &'a str,
    live_output_tokens: u64,
    final_usage: Option<TokenUsage>,
}

#[derive(Debug, Clone)]
struct BudgetEvidenceGroup {
    id: String,
    bucket: BudgetBucket,
    order: usize,
    evidence: Vec<RecalledEvidence>,
}

#[derive(Debug, Default)]
struct PreparationProbeCache {
    usages: HashMap<String, TokenUsage>,
    traces: Vec<BudgetProbeTrace>,
}

struct LiveSearchOutcome {
    recall: RecallResult,
    knowledge: KnowledgeRecall,
    query_embedding: Option<Vec<f32>>,
}

enum LiveSearchEvent {
    Fast {
        result: std::result::Result<RecallResult, String>,
        elapsed_ms: u64,
    },
    Knowledge {
        result: std::result::Result<KnowledgeRecall, String>,
        elapsed_ms: u64,
    },
    Embedding {
        result: std::result::Result<Vec<f32>, String>,
        elapsed_ms: u64,
        cache_hit: bool,
    },
    SemanticPrepared {
        result: std::result::Result<PreparedQueryVectorRecall, String>,
        elapsed_ms: u64,
    },
    Semantic {
        result: std::result::Result<RecallResult, String>,
        elapsed_ms: u64,
    },
}

#[derive(Debug, Clone)]
enum AcceptedBudgetUnit {
    Recent {
        index: usize,
        metric_before: u64,
        metric_after: u64,
        reflow: bool,
    },
    Evidence {
        id: String,
        bucket: BudgetBucket,
        metric_before: u64,
        metric_after: u64,
        reflow: bool,
    },
}

impl AcceptedBudgetUnit {
    fn bucket(&self) -> BudgetBucket {
        match self {
            Self::Recent { .. } => BudgetBucket::RecentHistory,
            Self::Evidence { bucket, .. } => *bucket,
        }
    }

    fn metrics(&self) -> (u64, u64) {
        match self {
            Self::Recent {
                metric_before,
                metric_after,
                ..
            }
            | Self::Evidence {
                metric_before,
                metric_after,
                ..
            } => (*metric_before, *metric_after),
        }
    }

    fn is_reflow(&self) -> bool {
        match self {
            Self::Recent { reflow, .. } | Self::Evidence { reflow, .. } => *reflow,
        }
    }
}

fn consolidation_report(
    session: &Session,
    trigger: ConsolidationTrigger,
) -> ConsolidationRunReport {
    ConsolidationRunReport {
        session_id: session.id.clone(),
        trigger,
        model: session.model.clone(),
        status: ConsolidationRunStatus::Failed,
        batches_attempted: 0,
        batches_applied: 0,
        events_attempted: 0,
        events_applied: 0,
        entities_attempted: 0,
        entities_applied: 0,
        claims_attempted: 0,
        claims_applied: 0,
        boundaries_attempted: 0,
        boundaries_applied: 0,
        watermark_before: 0,
        watermark_after: 0,
        warnings: Vec::new(),
    }
}

fn consolidation_failure_status(report: &ConsolidationRunReport) -> ConsolidationRunStatus {
    if report.batches_applied == 0 {
        ConsolidationRunStatus::Failed
    } else {
        ConsolidationRunStatus::Partial
    }
}

fn consolidation_elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(i64::MAX as u128) as u64
}

fn consolidation_failure_json(kind: &str, message: impl AsRef<str>) -> String {
    json!({"kind": kind, "message": message.as_ref(), "valid": false}).to_string()
}

const CONSOLIDATION_VALIDATION_ATTEMPTS: usize = 3;

fn consolidation_audit_response(raw: &str) -> String {
    if serde_json::from_str::<Value>(raw).is_ok() {
        raw.to_owned()
    } else {
        json!({
            "kind": "invalid_model_output",
            "raw_output": raw,
            "raw_sha256": content_sha256(raw),
            "valid": false,
        })
        .to_string()
    }
}

fn check_embedding_cancellation(
    cancellation: &CancellationToken,
    stage: EmbeddingRefreshStage,
    batch: Option<usize>,
) -> std::result::Result<(), EmbeddingRefreshError> {
    if cancellation.is_cancelled() {
        Err(EmbeddingRefreshError::Cancelled { stage, batch })
    } else {
        Ok(())
    }
}

fn embedding_catalog_error(stage: EmbeddingRefreshStage, message: String) -> EmbeddingRefreshError {
    EmbeddingRefreshError::Retrieval {
        stage,
        source: RetrievalError::CorruptIndex(message),
    }
}

fn derive_long_message_vectors(
    snapshot: &LeafEmbeddingSnapshot,
    vectors: &mut [Option<Vec<f32>>],
    cancellation: &CancellationToken,
) -> std::result::Result<(), EmbeddingRefreshError> {
    let mut fragments_by_message: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, document) in snapshot.documents.iter().enumerate() {
        if document.granularity == RetrievalDocumentGranularity::Fragment {
            fragments_by_message
                .entry(&document.message_document_id)
                .or_default()
                .push(index);
        }
    }
    for (message_index, message) in snapshot.documents.iter().enumerate() {
        check_embedding_cancellation(cancellation, EmbeddingRefreshStage::Planning, None)?;
        if message.granularity != RetrievalDocumentGranularity::Message
            || message.content.chars().count() <= 240
        {
            continue;
        }
        let message_len = message.content.chars().count();
        if message.start_char != 0 || message.end_char != message_len {
            return Err(embedding_catalog_error(
                EmbeddingRefreshStage::Planning,
                format!("长消息 {} 的规范范围无效", message.document_id),
            ));
        }
        let mut fragment_indices = fragments_by_message
            .get(message.document_id.as_str())
            .cloned()
            .unwrap_or_default();
        fragment_indices.sort_by(|left, right| {
            let left = &snapshot.documents[*left];
            let right = &snapshot.documents[*right];
            (left.start_char, left.end_char, left.document_id.as_str()).cmp(&(
                right.start_char,
                right.end_char,
                right.document_id.as_str(),
            ))
        });
        let mut expected_ranges = Vec::new();
        let mut start = 0_usize;
        while start < message_len {
            let end = start.saturating_add(240).min(message_len);
            expected_ranges.push((start, end));
            if end == message_len {
                break;
            }
            start = start.checked_add(200).ok_or_else(|| {
                embedding_catalog_error(
                    EmbeddingRefreshStage::Planning,
                    format!("长消息 {} 的分片范围溢出", message.document_id),
                )
            })?;
        }
        let actual_ranges = fragment_indices
            .iter()
            .map(|index| {
                let fragment = &snapshot.documents[*index];
                (fragment.start_char, fragment.end_char)
            })
            .collect::<Vec<_>>();
        if actual_ranges != expected_ranges {
            return Err(embedding_catalog_error(
                EmbeddingRefreshStage::Planning,
                format!("长消息 {} 的分片窗口不规范", message.document_id),
            ));
        }
        let mut covered_end = 0_usize;
        let mut weighted = Vec::with_capacity(fragment_indices.len());
        for fragment_index in fragment_indices {
            let fragment = &snapshot.documents[fragment_index];
            let new_start = covered_end.max(fragment.start_char);
            let weight = fragment.end_char.checked_sub(new_start).ok_or_else(|| {
                embedding_catalog_error(
                    EmbeddingRefreshStage::Planning,
                    format!("长消息 {} 的分片覆盖无效", message.document_id),
                )
            })?;
            if weight == 0 || fragment.start_char > covered_end {
                return Err(embedding_catalog_error(
                    EmbeddingRefreshStage::Planning,
                    format!("长消息 {} 的分片覆盖不连续", message.document_id),
                ));
            }
            covered_end = covered_end.max(fragment.end_char);
            let vector = vectors[fragment_index].as_deref().ok_or_else(|| {
                embedding_catalog_error(
                    EmbeddingRefreshStage::Planning,
                    format!("分片 {} 缺少规划向量", fragment.document_id),
                )
            })?;
            weighted.push((vector, weight));
        }
        if covered_end != message_len {
            return Err(embedding_catalog_error(
                EmbeddingRefreshStage::Planning,
                format!("长消息 {} 的分片未覆盖全文", message.document_id),
            ));
        }
        vectors[message_index] =
            Some(
                weighted_pool(&weighted).map_err(|source| EmbeddingRefreshError::Vector {
                    document_id: message.document_id.clone(),
                    source,
                })?,
            );
    }
    Ok(())
}

fn aggregate_embedding_writes(
    snapshot: &AggregateEmbeddingSnapshot,
) -> std::result::Result<Vec<EmbeddingWrite>, EmbeddingRefreshError> {
    snapshot
        .documents
        .iter()
        .map(|document| {
            let vectors = document
                .direct_messages
                .iter()
                .map(|message| message.vector.as_slice())
                .collect::<Vec<_>>();
            let vector = equal_mean(&vectors).map_err(|source| EmbeddingRefreshError::Vector {
                document_id: document.document_id.clone(),
                source,
            })?;
            Ok(EmbeddingWrite {
                document_id: document.document_id.clone(),
                expected_source_sha256: document.source_sha256.clone(),
                vector,
            })
        })
        .collect()
}

impl<B: ChatBackend> ChatEngine<B> {
    pub fn new(store: SessionStore, client: B) -> Self {
        Self::with_config(store, client, AppConfig::default())
    }

    pub fn with_config(store: SessionStore, client: B, config: AppConfig) -> Self {
        Self {
            store,
            client,
            assembler: ContextAssembler,
            config,
            background_embedding_gates: Arc::new(Mutex::new(HashMap::new())),
            background_embedding_pending: Arc::new(AtomicUsize::new(0)),
            background_embedding_notify: Arc::new(Notify::new()),
            background_embedding_cancellation: Arc::new(Mutex::new(CancellationToken::new())),
            #[cfg(test)]
            live_search_barrier: Arc::new(Mutex::new(None)),
            #[cfg(test)]
            live_search_fast_gate: Arc::new(Mutex::new(None)),
            #[cfg(test)]
            live_search_semantic_prepared: Arc::new(Mutex::new(None)),
        }
    }

    pub fn store(&self) -> &SessionStore {
        &self.store
    }

    pub fn client(&self) -> &B {
        &self.client
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    /// Refreshes the complete embedding catalog.
    ///
    /// Cancellation is observed until the final check immediately before leaf publication. After
    /// that commit point cancellation is deliberately ignored so the derived aggregate tail can
    /// be brought to a consistent result.
    pub async fn refresh_embeddings(
        &self,
        cancellation: CancellationToken,
    ) -> std::result::Result<EmbeddingRefreshReport, EmbeddingRefreshError> {
        if !self.config.memory.enabled {
            return Err(EmbeddingRefreshError::Disabled);
        }
        check_embedding_cancellation(&cancellation, EmbeddingRefreshStage::LeafSnapshot, None)?;

        let memory_config = self.config.memory.clone();
        let spec = VectorIndexSpec::from_config(&memory_config)
            .map_err(|source| EmbeddingRefreshError::InvalidConfig { source })?;
        let retrieval = self.store.retrieval().clone();
        let snapshot_spec = spec.clone();
        let leaf_snapshot =
            tokio::task::spawn_blocking(move || retrieval.leaf_embedding_snapshot(&snapshot_spec))
                .await
                .map_err(|source| EmbeddingRefreshError::TaskJoin {
                    stage: EmbeddingRefreshStage::LeafSnapshot,
                    source,
                })?
                .map_err(|source| EmbeddingRefreshError::Retrieval {
                    stage: EmbeddingRefreshStage::LeafSnapshot,
                    source,
                })?;
        let cached_vectors = self
            .store
            .retrieval()
            .cached_content_embeddings(&spec)
            .map_err(|source| EmbeddingRefreshError::Retrieval {
                stage: EmbeddingRefreshStage::LeafSnapshot,
                source,
            })?;

        let mut vectors = vec![None; leaf_snapshot.documents.len()];
        let mut pending = Vec::new();
        for (index, document) in leaf_snapshot.documents.iter().enumerate() {
            check_embedding_cancellation(&cancellation, EmbeddingRefreshStage::Planning, None)?;
            match document.granularity {
                RetrievalDocumentGranularity::Fragment => {
                    if let Some(vector) = &document.reusable_vector {
                        vectors[index] = Some(vector.clone());
                    } else if let Some(vector) = cached_vectors.get(&document.source_sha256) {
                        vectors[index] = Some(vector.clone());
                    } else {
                        pending.push((index, document.content.clone()));
                    }
                }
                RetrievalDocumentGranularity::Message => {
                    if document.content.chars().count() <= 240 {
                        if let Some(vector) = &document.reusable_vector {
                            vectors[index] = Some(vector.clone());
                        } else if let Some(vector) = cached_vectors.get(&document.source_sha256) {
                            vectors[index] = Some(vector.clone());
                        } else {
                            pending.push((index, document.content.clone()));
                        }
                    }
                }
                _ => {
                    return Err(embedding_catalog_error(
                        EmbeddingRefreshStage::Planning,
                        format!("leaf catalog 包含非 leaf 文档 {}", document.document_id),
                    ));
                }
            }
        }

        let mut backend_batches = 0_usize;
        for (batch_index, chunk) in pending
            .chunks(memory_config.embedding_batch_size)
            .enumerate()
        {
            let batch = batch_index + 1;
            check_embedding_cancellation(
                &cancellation,
                EmbeddingRefreshStage::Backend,
                Some(batch),
            )?;
            let request = EmbeddingRequest {
                model: spec.model.clone(),
                input: chunk.iter().map(|(_, content)| content.clone()).collect(),
                dimensions: Some(spec.dimensions),
                truncate: false,
            };
            let retrieval = self.store.retrieval().clone();
            let expected_generation = leaf_snapshot.control_generation_sha256.clone();
            tokio::task::spawn_blocking(move || {
                retrieval.verify_embedding_refresh_generation(&expected_generation)
            })
            .await
            .map_err(|source| EmbeddingRefreshError::TaskJoin {
                stage: EmbeddingRefreshStage::Backend,
                source,
            })?
            .map_err(|source| EmbeddingRefreshError::Retrieval {
                stage: EmbeddingRefreshStage::Backend,
                source,
            })?;
            backend_batches += 1;
            let call = self.client.embed(request);
            let response = tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    return Err(EmbeddingRefreshError::Cancelled {
                        stage: EmbeddingRefreshStage::Backend,
                        batch: Some(batch),
                    });
                }
                timed = tokio::time::timeout(
                    Duration::from_secs(memory_config.embedding_timeout_secs),
                    call,
                ) => match timed {
                    Ok(Ok(response)) => response,
                    Ok(Err(source)) => {
                        return Err(EmbeddingRefreshError::Backend { batch, source });
                    }
                    Err(_) => {
                        return Err(EmbeddingRefreshError::Timeout {
                            batch,
                            timeout_secs: memory_config.embedding_timeout_secs,
                        });
                    }
                },
            };
            if response.model != spec.model {
                return Err(EmbeddingRefreshError::InvalidResponse {
                    batch,
                    message: format!("模型不匹配：期望 {}，实际 {}", spec.model, response.model),
                });
            }
            if response.embeddings.len() != chunk.len() {
                return Err(EmbeddingRefreshError::InvalidResponse {
                    batch,
                    message: format!(
                        "向量数量不匹配：期望 {}，实际 {}",
                        chunk.len(),
                        response.embeddings.len()
                    ),
                });
            }
            for ((document_index, _), vector) in chunk.iter().zip(response.embeddings) {
                let document = &leaf_snapshot.documents[*document_index];
                if vector.len() != spec.dimensions {
                    return Err(EmbeddingRefreshError::Vector {
                        document_id: document.document_id.clone(),
                        source: VectorError::DimensionMismatch {
                            expected: spec.dimensions,
                            actual: vector.len(),
                        },
                    });
                }
                vectors[*document_index] = Some(l2_normalize(&vector).map_err(|source| {
                    EmbeddingRefreshError::Vector {
                        document_id: document.document_id.clone(),
                        source,
                    }
                })?);
            }
        }

        derive_long_message_vectors(&leaf_snapshot, &mut vectors, &cancellation)?;
        let writes = leaf_snapshot
            .documents
            .iter()
            .zip(vectors)
            .map(|(document, vector)| {
                vector
                    .map(|vector| EmbeddingWrite {
                        document_id: document.document_id.clone(),
                        expected_source_sha256: document.source_sha256.clone(),
                        vector,
                    })
                    .ok_or_else(|| {
                        embedding_catalog_error(
                            EmbeddingRefreshStage::Planning,
                            format!("文档 {} 缺少规划向量", document.document_id),
                        )
                    })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        check_embedding_cancellation(&cancellation, EmbeddingRefreshStage::LeafPublish, None)?;
        let retrieval = self.store.retrieval().clone();
        let publish_spec = spec.clone();
        let publish_snapshot = leaf_snapshot.clone();
        let leaf_publish = tokio::task::spawn_blocking(move || {
            retrieval.publish_leaf_embedding_catalog(&publish_spec, &publish_snapshot, &writes)
        })
        .await
        .map_err(|source| EmbeddingRefreshError::TaskJoin {
            stage: EmbeddingRefreshStage::LeafPublish,
            source,
        })?
        .map_err(|source| EmbeddingRefreshError::Retrieval {
            stage: EmbeddingRefreshStage::LeafPublish,
            source,
        })?;

        let mut session_ids = leaf_snapshot.session_ids.clone();
        session_ids.sort();
        let original_len = session_ids.len();
        session_ids.dedup();
        if session_ids.len() != original_len {
            return Err(embedding_catalog_error(
                EmbeddingRefreshStage::Materialization,
                "leaf snapshot 包含重复会话".into(),
            ));
        }
        for session_id in session_ids {
            let retrieval = self.store.retrieval().clone();
            let materialization_config = memory_config.clone();
            let error_session_id = session_id.clone();
            tokio::task::spawn_blocking(move || {
                retrieval.materialize_episode_documents(&session_id, &materialization_config)
            })
            .await
            .map_err(|source| EmbeddingRefreshError::TaskJoin {
                stage: EmbeddingRefreshStage::Materialization,
                source,
            })?
            .map_err(|source| EmbeddingRefreshError::Materialization {
                session_id: error_session_id,
                source,
            })?;
        }

        let retrieval = self.store.retrieval().clone();
        let aggregate_spec = spec.clone();
        let aggregate_snapshot = tokio::task::spawn_blocking(move || {
            retrieval.aggregate_embedding_snapshot(&aggregate_spec)
        })
        .await
        .map_err(|source| EmbeddingRefreshError::TaskJoin {
            stage: EmbeddingRefreshStage::AggregateSnapshot,
            source,
        })?
        .map_err(|source| EmbeddingRefreshError::Retrieval {
            stage: EmbeddingRefreshStage::AggregateSnapshot,
            source,
        })?;
        let aggregate_writes = aggregate_embedding_writes(&aggregate_snapshot)?;
        let aggregate_documents = aggregate_snapshot.documents.len();
        let retrieval = self.store.retrieval().clone();
        let aggregate_spec = spec.clone();
        tokio::task::spawn_blocking(move || {
            retrieval.publish_aggregate_embedding_catalog(
                &aggregate_spec,
                &aggregate_snapshot,
                &aggregate_writes,
            )
        })
        .await
        .map_err(|source| EmbeddingRefreshError::TaskJoin {
            stage: EmbeddingRefreshStage::AggregatePublish,
            source,
        })?
        .map_err(|source| EmbeddingRefreshError::Retrieval {
            stage: EmbeddingRefreshStage::AggregatePublish,
            source,
        })?;

        Ok(EmbeddingRefreshReport {
            leaf_documents: leaf_snapshot.documents.len(),
            leaf_reused: leaf_publish.reused,
            leaf_embedded_inputs: pending.len(),
            backend_batches,
            aggregate_documents,
            leaf_committed: leaf_publish.changed,
        })
    }

    /// Best-effort derived-memory extraction. Source session data is never mutated here, and a
    /// report is returned even when the model or derived index is unavailable.
    pub async fn consolidate_session(
        &self,
        session: &Session,
        trigger: ConsolidationTrigger,
        cancellation: CancellationToken,
    ) -> ConsolidationRunReport {
        self.consolidate_session_with_progress(session, trigger, cancellation, |_| {})
            .await
    }

    pub async fn consolidate_session_with_progress<F>(
        &self,
        session: &Session,
        trigger: ConsolidationTrigger,
        cancellation: CancellationToken,
        mut progress: F,
    ) -> ConsolidationRunReport
    where
        F: FnMut(ConsolidationProgress),
    {
        let mut report = consolidation_report(session, trigger);
        match self.store.control_state() {
            Ok(controls) if controls.allows_session(&session.id) => {}
            Ok(_) => {
                report.warnings.push(format!("会话已排除: {}", session.id));
                return report;
            }
            Err(error) => {
                report.warnings.push(format!("无法读取控制状态: {error}"));
                return report;
            }
        }
        if !self.config.memory.enabled {
            report.status = ConsolidationRunStatus::Disabled;
            return report;
        }
        if cancellation.is_cancelled() {
            report.status = ConsolidationRunStatus::Cancelled;
            return report;
        }

        let persisted = match self.store.load(&session.id) {
            Ok(persisted) => persisted,
            Err(error) => {
                report
                    .warnings
                    .push(format!("无法加载会话 {}: {error}", session.id));
                return report;
            }
        };
        report.model = persisted.model.clone();
        if cancellation.is_cancelled() {
            report.status = ConsolidationRunStatus::Cancelled;
            return report;
        }

        let retrieval = self.store.retrieval();
        let watermark = match retrieval.consolidation_watermark(&persisted.id) {
            Ok(watermark) => watermark,
            Err(error) => {
                report.warnings.push(format!("无法读取巩固水位: {error}"));
                return report;
            }
        };
        report.watermark_before = watermark.through_sequence;
        report.watermark_after = watermark.through_sequence;

        loop {
            if cancellation.is_cancelled() {
                report.status = ConsolidationRunStatus::Cancelled;
                return report;
            }
            let batch = match retrieval.next_consolidation_batch(&persisted.id) {
                Ok(Some(batch)) => batch,
                Ok(None) => {
                    report.status = if report.batches_applied == 0 {
                        ConsolidationRunStatus::UpToDate
                    } else {
                        ConsolidationRunStatus::Completed
                    };
                    return report;
                }
                Err(error) => {
                    report.status = consolidation_failure_status(&report);
                    report.warnings.push(format!("无法读取待巩固批次: {error}"));
                    return report;
                }
            };
            if cancellation.is_cancelled() {
                report.status = ConsolidationRunStatus::Cancelled;
                return report;
            }
            let candidates = match retrieval.consolidation_candidates(
                self.config.memory.candidate_limit,
                self.config.memory.candidate_limit,
            ) {
                Ok(candidates) => candidates,
                Err(error) => {
                    report.status = consolidation_failure_status(&report);
                    report.warnings.push(format!("无法读取巩固候选: {error}"));
                    return report;
                }
            };
            if cancellation.is_cancelled() {
                report.status = ConsolidationRunStatus::Cancelled;
                return report;
            }
            let request = match canonical_consolidation_request(
                persisted.model.clone(),
                &batch,
                &candidates,
                persisted.budget.context_window,
                persisted.budget.max_output_tokens,
            ) {
                Ok(request) => request,
                Err(error) => {
                    report.status = consolidation_failure_status(&report);
                    report.warnings.push(format!("无法序列化巩固请求: {error}"));
                    return report;
                }
            };
            let request_json = match serde_json::to_string(&request) {
                Ok(request_json) => request_json,
                Err(error) => {
                    report.status = consolidation_failure_status(&report);
                    report.warnings.push(format!("无法序列化巩固请求: {error}"));
                    return report;
                }
            };
            if cancellation.is_cancelled() {
                report.status = ConsolidationRunStatus::Cancelled;
                return report;
            }

            report.batches_attempted += 1;
            report.events_attempted += batch.events.len();
            let mut validation_attempt = 0_usize;
            loop {
                validation_attempt += 1;
                progress(ConsolidationProgress::AttemptStarted {
                    session_id: persisted.id.clone(),
                    attempt: validation_attempt,
                    events: batch.events.len(),
                    from_sequence: batch.from_sequence,
                    through_sequence: batch.through_sequence,
                });
                let started_at = utc_now();
                let started = Instant::now();
                let call = self.client.structured_chat(request.clone());
                let result = tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(("cancellation", "caller cancelled consolidation".to_owned(), ConsolidationAttemptStatus::Cancelled)),
                timed = tokio::time::timeout(Duration::from_secs(self.config.memory.consolidation_timeout_secs), call) => match timed {
                    Ok(Ok(response)) => Ok(response),
                    Ok(Err(error)) => Err(("backend", error.to_string(), ConsolidationAttemptStatus::ModelError)),
                    Err(_) => Err(("timeout", "structured consolidation model call timed out".to_owned(), ConsolidationAttemptStatus::ModelError)),
                },
                };
                let completed_at = utc_now();
                let latency_ms = consolidation_elapsed_ms(started);
                let base_attempt = |status,
                                    response_json,
                                    response_sha256,
                                    input_tokens,
                                    output_tokens,
                                    validation_json,
                                    error_json| {
                    ConsolidationAttemptRecord {
                        attempt_id: Uuid::new_v4().to_string(),
                        batch_key: batch.batch_key.clone(),
                        session_id: persisted.id.clone(),
                        from_sequence: batch.from_sequence,
                        through_sequence: batch.through_sequence,
                        trigger: trigger.as_str().into(),
                        model: persisted.model.clone(),
                        request_json: request_json.clone(),
                        request_sha256: content_sha256(&request_json),
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
                        response_json,
                        response_sha256,
                        status,
                        input_tokens,
                        output_tokens,
                        latency_ms,
                        started_at: started_at.clone(),
                        completed_at: completed_at.clone(),
                        validation_json,
                        error_json,
                    }
                };

                let response = match result {
                    Ok(response) => response,
                    Err((kind, message, status)) => {
                        let primary = format!("{kind}: {message}");
                        let attempt = base_attempt(
                            status,
                            None,
                            None,
                            None,
                            None,
                            None,
                            Some(consolidation_failure_json(kind, &message)),
                        );
                        if let Err(audit_error) = retrieval.record_consolidation_failure(&attempt) {
                            report.warnings.push(format!(
                                "巩固失败 ({primary}); 失败审计无法写入: {audit_error}"
                            ));
                        } else {
                            report.warnings.push(format!("巩固失败: {primary}"));
                        }
                        report.status = if status == ConsolidationAttemptStatus::Cancelled {
                            ConsolidationRunStatus::Cancelled
                        } else {
                            consolidation_failure_status(&report)
                        };
                        return report;
                    }
                };

                let response_json = response.content;
                let response_sha256 = content_sha256(&response_json);
                let response_input_tokens = response.usage.input_tokens;
                let response_output_tokens = response.usage.output_tokens;
                let decoded = serde_json::from_str::<StructuredConsolidationOutput>(&response_json);
                if let Ok(output) = &decoded {
                    report.entities_attempted += output.entities.len();
                    report.claims_attempted += output.claims.len();
                    report.boundaries_attempted += output.boundaries.len();
                }
                if cancellation.is_cancelled() {
                    let message = "caller cancelled consolidation";
                    let attempt = base_attempt(
                        ConsolidationAttemptStatus::Cancelled,
                        Some(response_json),
                        Some(response_sha256),
                        response_input_tokens,
                        response_output_tokens,
                        None,
                        Some(consolidation_failure_json("cancellation", message)),
                    );
                    if let Err(audit_error) = retrieval.record_consolidation_failure(&attempt) {
                        report.warnings.push(format!(
                            "巩固取消 ({message}); 失败审计无法写入: {audit_error}"
                        ));
                    }
                    report.status = ConsolidationRunStatus::Cancelled;
                    return report;
                }
                let attempt = base_attempt(
                    ConsolidationAttemptStatus::Applied,
                    Some(response_json.clone()),
                    Some(response_sha256),
                    response_input_tokens,
                    response_output_tokens,
                    Some(json!({"valid": true, "batch_key": batch.batch_key, "watermark_after": batch.through_sequence}).to_string()),
                    None,
                );
                match retrieval.apply_consolidation_attempt(&batch, &candidates, &attempt) {
                    Ok(applied) => {
                        report.batches_applied += 1;
                        report.events_applied += batch.events.len();
                        if let Ok(output) = decoded {
                            report.entities_applied += output.entities.len();
                            report.claims_applied += output.claims.len();
                            report.boundaries_applied += output.boundaries.len();
                        }
                        report.watermark_after = applied.watermark_after;
                        progress(ConsolidationProgress::BatchApplied {
                            session_id: persisted.id.clone(),
                            batches_applied: report.batches_applied,
                            watermark_after: report.watermark_after,
                        });
                        break;
                    }
                    Err(error) => {
                        let (validation_json, message, retryable) = match error {
                            ConsolidationApplyError::Rejected {
                                validation_json,
                                message,
                            } => (validation_json, message, true),
                            ConsolidationApplyError::Stale { message } => (
                                consolidation_failure_json("stale", &message),
                                message,
                                false,
                            ),
                            ConsolidationApplyError::Retrieval(error) => {
                                let message = error.to_string();
                                (
                                    consolidation_failure_json("retrieval", &message),
                                    message,
                                    false,
                                )
                            }
                        };
                        let audited_response = consolidation_audit_response(&response_json);
                        let rejected = base_attempt(
                            ConsolidationAttemptStatus::Rejected,
                            Some(audited_response.clone()),
                            Some(content_sha256(&audited_response)),
                            attempt.input_tokens,
                            attempt.output_tokens,
                            Some(validation_json),
                            Some(consolidation_failure_json("apply", &message)),
                        );
                        if let Err(audit_error) = retrieval.record_consolidation_failure(&rejected)
                        {
                            report.warnings.push(format!(
                                "巩固应用被拒绝 ({message}); 失败审计无法写入: {audit_error}"
                            ));
                        } else {
                            report.warnings.push(format!("巩固应用被拒绝: {message}"));
                        }
                        if retryable && validation_attempt < CONSOLIDATION_VALIDATION_ATTEMPTS {
                            report.warnings.push(format!(
                                "巩固输出校验失败，正在进行第 {}/{} 次重试",
                                validation_attempt + 1,
                                CONSOLIDATION_VALIDATION_ATTEMPTS
                            ));
                            progress(ConsolidationProgress::ValidationRetry {
                                session_id: persisted.id.clone(),
                                next_attempt: validation_attempt + 1,
                                max_attempts: CONSOLIDATION_VALIDATION_ATTEMPTS,
                            });
                            continue;
                        }
                        report.status = consolidation_failure_status(&report);
                        return report;
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn live_search(
        &self,
        user_content: &str,
        current_event_id: &str,
        recent_event_ids: &[String],
        retrieval_config: crate::model::RetrievalConfig,
        memory_config: crate::config::MemoryConfig,
    ) -> LiveSearchOutcome {
        let started = Instant::now();
        let deadline_ms = memory_config.search_timeout_ms;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(deadline_ms);
        log::info!(
            target: "hippocampus::retrieval",
            "live search started event_id={} memory_enabled={} recent_events={} deadline_ms={}",
            current_event_id,
            memory_config.enabled,
            recent_event_ids.len(),
            deadline_ms,
        );
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut tasks = Vec::new();

        let fast_store = self.store.retrieval().clone();
        let fast_query = user_content.to_owned();
        let fast_current = current_event_id.to_owned();
        let fast_recent = recent_event_ids.to_vec();
        let fast_config = retrieval_config.clone();
        let fast_sender = sender.clone();
        #[cfg(test)]
        let fast_barrier = self
            .live_search_barrier
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        #[cfg(test)]
        let fast_gate = self
            .live_search_fast_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        tasks.push(tokio::spawn(async move {
            #[cfg(test)]
            if let Some(barrier) = fast_barrier {
                barrier.wait().await;
            }
            #[cfg(test)]
            if let Some(gate) = fast_gate {
                gate.notified().await;
            }
            let task_started = Instant::now();
            let result = tokio::task::spawn_blocking(move || {
                fast_store.keyword_recall(&fast_query, &fast_current, &fast_recent, fast_config)
            })
            .await
            .map_err(|error| format!("快速搜索任务失败: {error}"))
            .and_then(|result| result.map_err(|error| error.to_string()));
            let _ = fast_sender.send(LiveSearchEvent::Fast {
                result,
                elapsed_ms: elapsed_millis(task_started),
            });
        }));

        let knowledge_store = self.store.knowledge().clone();
        let knowledge_query = user_content.to_owned();
        let knowledge_config = self.config.knowledge.clone();
        let knowledge_sender = sender.clone();
        #[cfg(test)]
        let knowledge_barrier = self
            .live_search_barrier
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        tasks.push(tokio::spawn(async move {
            #[cfg(test)]
            if let Some(barrier) = knowledge_barrier {
                barrier.wait().await;
            }
            let task_started = Instant::now();
            let result = tokio::task::spawn_blocking(move || {
                knowledge_store.recall(&knowledge_query, &knowledge_config)
            })
            .await
            .map_err(|error| format!("知识搜索任务失败: {error}"))
            .and_then(|result| result.map_err(|error| format!("{error:#}")));
            let _ = knowledge_sender.send(LiveSearchEvent::Knowledge {
                result,
                elapsed_ms: elapsed_millis(task_started),
            });
        }));

        if memory_config.enabled {
            let client = self.client.clone();
            let embedding_query = user_content.to_owned();
            let embedding_model = memory_config.embedding_model.clone();
            let embedding_dimensions = memory_config.embedding_dimensions;
            let embedding_spec = VectorIndexSpec::from_config(&memory_config);
            let embedding_store = self.store.retrieval().clone();
            let embedding_sender = sender.clone();
            #[cfg(test)]
            let embedding_barrier = self
                .live_search_barrier
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            tasks.push(tokio::spawn(async move {
                #[cfg(test)]
                if let Some(barrier) = embedding_barrier {
                    barrier.wait().await;
                }
                let task_started = Instant::now();
                let query_hash = content_sha256(&embedding_query);
                let cached = match embedding_spec {
                    Ok(spec) => tokio::task::spawn_blocking(move || {
                        embedding_store.cached_content_embedding(&spec, &query_hash)
                    })
                    .await
                    .map_err(|error| format!("查询向量缓存任务失败: {error}"))
                    .and_then(|result| result.map_err(|error| error.to_string())),
                    Err(error) => Err(error.to_string()),
                };
                let (result, cache_hit) = match cached {
                    Ok(Some(vector)) => (Ok(vector), true),
                    Ok(None) => {
                        let result = client
                            .embed(EmbeddingRequest {
                                model: embedding_model.clone(),
                                input: vec![embedding_query],
                                dimensions: Some(embedding_dimensions),
                                truncate: false,
                            })
                            .await
                            .map_err(|error| error.to_string())
                            .and_then(|response| {
                                if response.model != embedding_model
                                    || response.embeddings.len() != 1
                                {
                                    return Err("查询 Embedding 响应与请求不匹配".into());
                                }
                                let vector = response
                                    .embeddings
                                    .into_iter()
                                    .next()
                                    .expect("validated one query embedding");
                                if vector.len() != embedding_dimensions {
                                    return Err(format!(
                                        "查询向量维度不匹配：期望 {embedding_dimensions}，实际 {}",
                                        vector.len()
                                    ));
                                }
                                l2_normalize(&vector).map_err(|error| error.to_string())
                            });
                        (result, false)
                    }
                    Err(error) => (Err(error), false),
                };
                let _ = embedding_sender.send(LiveSearchEvent::Embedding {
                    result,
                    elapsed_ms: elapsed_millis(task_started),
                    cache_hit,
                });
            }));
        }
        let mut fast: Option<RecallResult> = None;
        let mut fast_finished = false;
        let mut fast_error = None;
        let mut fast_elapsed_ms = 0;
        let mut knowledge: Option<KnowledgeRecall> = None;
        let mut knowledge_finished = false;
        let mut knowledge_error = None;
        let mut knowledge_elapsed_ms = 0;
        let mut query_embedding: Option<Vec<f32>> = None;
        let mut embedding_finished = !memory_config.enabled;
        let mut embedding_error = None;
        let mut embedding_elapsed_ms = 0;
        let mut semantic: Option<RecallResult> = None;
        let mut semantic_prepared: Option<PreparedQueryVectorRecall> = None;
        let mut semantic_finished = !memory_config.enabled;
        let mut semantic_error = None;
        let mut semantic_prepare_elapsed_ms = 0;
        let mut semantic_fusion_elapsed_ms = 0;
        let mut semantic_preparation_started = false;
        let mut semantic_fusion_started = false;
        let mut deadline_exceeded = false;

        loop {
            if fast_finished && knowledge_finished && semantic_finished {
                break;
            }
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    deadline_exceeded = true;
                    break;
                }
                event = receiver.recv() => {
                    let Some(event) = event else { break; };
                    match event {
                        LiveSearchEvent::Fast { result, elapsed_ms } => {
                            fast_finished = true;
                            fast_elapsed_ms = elapsed_ms;
                            match result {
                                Ok(result) => {
                                    log::debug!(
                                        target: "hippocampus::retrieval",
                                        "bm25 completed event_id={} status={} candidates={} selected={} elapsed_ms={}",
                                        current_event_id,
                                        result.trace.status,
                                        result.trace.candidates.len(),
                                        result.evidence.len(),
                                        elapsed_ms,
                                    );
                                    fast = Some(result);
                                }
                                Err(error) => {
                                    log::warn!(
                                        target: "hippocampus::retrieval",
                                        "bm25 failed event_id={} elapsed_ms={} error={}",
                                        current_event_id,
                                        elapsed_ms,
                                        error,
                                    );
                                    fast_error = Some(error);
                                }
                            }
                        }
                        LiveSearchEvent::Knowledge { result, elapsed_ms } => {
                            knowledge_finished = true;
                            knowledge_elapsed_ms = elapsed_ms;
                            match result {
                                Ok(result) => {
                                    log::debug!(
                                        target: "hippocampus::retrieval",
                                        "knowledge recall completed event_id={} status={} candidates={} selected={} elapsed_ms={}",
                                        current_event_id,
                                        result.trace.status,
                                        result.trace.candidates.len(),
                                        result.trace.selected_evidence.len(),
                                        elapsed_ms,
                                    );
                                    knowledge = Some(result);
                                }
                                Err(error) => {
                                    log::warn!(
                                        target: "hippocampus::retrieval",
                                        "knowledge recall failed event_id={} elapsed_ms={} error={}",
                                        current_event_id,
                                        elapsed_ms,
                                        error,
                                    );
                                    knowledge_error = Some(error);
                                }
                            }
                        }
                        LiveSearchEvent::Embedding {
                            result,
                            elapsed_ms,
                            cache_hit,
                        } => {
                            embedding_finished = true;
                            embedding_elapsed_ms = elapsed_ms;
                            match result {
                                Ok(vector) => {
                                    log::debug!(
                                        target: "hippocampus::retrieval",
                                        "query embedding completed event_id={} dimensions={} cache_hit={} elapsed_ms={}",
                                        current_event_id,
                                        vector.len(),
                                        cache_hit,
                                        elapsed_ms,
                                    );
                                    query_embedding = Some(vector);
                                }
                                Err(error) => {
                                    log::warn!(
                                        target: "hippocampus::retrieval",
                                        "query embedding failed event_id={} elapsed_ms={} error={}",
                                        current_event_id,
                                        elapsed_ms,
                                        error,
                                    );
                                    embedding_error = Some(error);
                                }
                            }
                        }
                        LiveSearchEvent::SemanticPrepared { result, elapsed_ms } => {
                            semantic_prepare_elapsed_ms = elapsed_ms;
                            #[cfg(test)]
                            if let Some(notify) = self
                                .live_search_semantic_prepared
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .clone()
                            {
                                notify.notify_one();
                            }
                            match result {
                                Ok(prepared) => {
                                    log::debug!(
                                        target: "hippocampus::retrieval",
                                        "semantic preparation completed event_id={} elapsed_ms={}",
                                        current_event_id,
                                        elapsed_ms,
                                    );
                                    semantic_prepared = Some(prepared);
                                }
                                Err(error) => {
                                    log::warn!(
                                        target: "hippocampus::retrieval",
                                        "semantic preparation failed event_id={} elapsed_ms={} error={}",
                                        current_event_id,
                                        elapsed_ms,
                                        error,
                                    );
                                    semantic_error = Some(error);
                                    semantic_finished = true;
                                }
                            }
                        }
                        LiveSearchEvent::Semantic { result, elapsed_ms } => {
                            semantic_finished = true;
                            semantic_fusion_elapsed_ms = elapsed_ms;
                            match result {
                                Ok(result) => {
                                    log::debug!(
                                        target: "hippocampus::retrieval",
                                        "semantic fusion completed event_id={} status={} selected={} elapsed_ms={}",
                                        current_event_id,
                                        result.trace.status,
                                        result.evidence.len(),
                                        elapsed_ms,
                                    );
                                    semantic = Some(result);
                                }
                                Err(error) => {
                                    log::warn!(
                                        target: "hippocampus::retrieval",
                                        "semantic fusion failed event_id={} elapsed_ms={} error={}",
                                        current_event_id,
                                        elapsed_ms,
                                        error,
                                    );
                                    semantic_error = Some(error);
                                }
                            }
                        }
                    }
                }
            }

            if !semantic_preparation_started && let Some(vector) = query_embedding.as_ref() {
                semantic_preparation_started = true;
                let semantic_store = self.store.retrieval().clone();
                let semantic_query = user_content.to_owned();
                let semantic_current = current_event_id.to_owned();
                let semantic_recent = recent_event_ids.to_vec();
                let semantic_retrieval = retrieval_config.clone();
                let semantic_memory = memory_config.clone();
                let semantic_vector = vector.clone();
                let semantic_sender = sender.clone();
                tasks.push(tokio::spawn(async move {
                    let task_started = Instant::now();
                    let result = semantic_store
                        .prepare_query_vector_recall(
                            &semantic_query,
                            &semantic_current,
                            &semantic_recent,
                            None,
                            semantic_retrieval,
                            &semantic_memory,
                            semantic_vector,
                        )
                        .await
                        .map_err(|error| error.to_string());
                    let _ = semantic_sender.send(LiveSearchEvent::SemanticPrepared {
                        result,
                        elapsed_ms: elapsed_millis(task_started),
                    });
                }));
            }

            if !semantic_fusion_started
                && let Some(semantic_fast) = fast.as_ref()
                && let Some(prepared) = semantic_prepared.take()
            {
                semantic_fusion_started = true;
                let semantic_store = self.store.retrieval().clone();
                let semantic_query = user_content.to_owned();
                let semantic_current = current_event_id.to_owned();
                let semantic_recent = recent_event_ids.to_vec();
                let semantic_retrieval = retrieval_config.clone();
                let semantic_memory = memory_config.clone();
                let semantic_fast = semantic_fast.clone();
                let semantic_sender = sender.clone();
                tasks.push(tokio::spawn(async move {
                    let task_started = Instant::now();
                    let result = semantic_store
                        .hybrid_recall_from_prepared_query_vector(
                            &semantic_query,
                            &semantic_current,
                            &semantic_recent,
                            None,
                            semantic_retrieval,
                            &semantic_memory,
                            prepared,
                            embedding_elapsed_ms,
                            semantic_fast,
                            fast_elapsed_ms,
                        )
                        .await
                        .map_err(|error| error.to_string());
                    let _ = semantic_sender.send(LiveSearchEvent::Semantic {
                        result,
                        elapsed_ms: elapsed_millis(task_started),
                    });
                }));
            }

            if memory_config.enabled
                && !semantic_finished
                && ((embedding_finished && embedding_error.is_some())
                    || (fast_finished && fast.is_none()))
            {
                semantic_finished = true;
            }
        }

        for task in tasks {
            task.abort();
        }

        let mut missing = Vec::new();
        if fast.is_none() {
            missing.push("bm25");
        }
        if memory_config.enabled && semantic.is_none() {
            missing.extend(["vector", "entity", "state", "episode", "graph"]);
        }
        if knowledge.is_none() {
            missing.push("knowledge");
        }

        let fast_available = fast.is_some();
        let fast_fallback_used = memory_config.enabled
            && (semantic.is_none()
                || semantic
                    .as_ref()
                    .is_some_and(|result| result.trace.status == "bm25_fallback"));
        let fallback_reason = fast_fallback_used.then(|| {
            if deadline_exceeded {
                format!(
                    "deadline_exceeded: deadline_ms={deadline_ms}, missing_channels={}",
                    missing.join(",")
                )
            } else if let Some(error) = embedding_error.as_deref() {
                format!("embedding_failed: {error}")
            } else if let Some(error) = semantic_error.as_deref() {
                format!("semantic_failed: {error}")
            } else if let Some(reason) = semantic
                .as_ref()
                .and_then(|result| result.trace.fallback_reason.as_deref())
            {
                reason.to_owned()
            } else if let Some(detail) = semantic.as_ref().and_then(|result| {
                result
                    .trace
                    .channels
                    .iter()
                    .find_map(|channel| channel.error.as_deref())
                    .or_else(|| result.trace.warnings.first().map(String::as_str))
            }) {
                format!("semantic_pipeline_fallback: {detail}")
            } else if !fast_available {
                "bm25_and_semantic_unavailable".into()
            } else if !semantic_preparation_started {
                "semantic_preparation_not_started".into()
            } else if !semantic_fusion_started {
                "semantic_fusion_not_started".into()
            } else {
                "semantic_result_unavailable".into()
            }
        });
        let mut recall = semantic.unwrap_or_else(|| {
            fast.unwrap_or_else(|| {
                empty_live_recall(
                    current_event_id,
                    retrieval_config.clone(),
                    &memory_config,
                    fast_error.as_deref(),
                )
            })
        });
        for message in [embedding_error.as_ref(), semantic_error.as_ref()]
            .into_iter()
            .flatten()
        {
            recall.trace.warnings.push(message.clone());
        }
        if deadline_exceeded {
            recall.trace.warnings.push(format!(
                "搜索超过 {deadline_ms}ms，已采用快速搜索结果；未完成通道：{}",
                missing.join(", ")
            ));
        }
        recall.trace.deadline_ms = deadline_ms;
        recall.trace.deadline_exceeded = deadline_exceeded;
        recall.trace.fast_fallback_used = fast_fallback_used;
        recall.trace.fallback_reason = fallback_reason;
        recall.trace.elapsed_ms = elapsed_millis(started);
        ensure_live_channel_statuses(
            &mut recall.trace,
            memory_config.enabled,
            deadline_exceeded,
            fast_available,
            fast_elapsed_ms,
            embedding_elapsed_ms,
        );
        if recall.trace.fast_fallback_used {
            log::warn!(
                target: "hippocampus::retrieval",
                "live search fell back to bm25 event_id={} reason={} total_ms={} bm25_ms={} embedding_ms={} semantic_prepare_ms={} semantic_fusion_ms={} knowledge_ms={}",
                current_event_id,
                recall.trace.fallback_reason.as_deref().unwrap_or("unknown"),
                recall.trace.elapsed_ms,
                fast_elapsed_ms,
                embedding_elapsed_ms,
                semantic_prepare_elapsed_ms,
                semantic_fusion_elapsed_ms,
                knowledge_elapsed_ms,
            );
        } else {
            log::info!(
                target: "hippocampus::retrieval",
                "live search completed event_id={} status={} selected={} total_ms={} bm25_ms={} embedding_ms={} semantic_prepare_ms={} semantic_fusion_ms={} knowledge_ms={}",
                current_event_id,
                recall.trace.status,
                recall.evidence.len(),
                recall.trace.elapsed_ms,
                fast_elapsed_ms,
                embedding_elapsed_ms,
                semantic_prepare_elapsed_ms,
                semantic_fusion_elapsed_ms,
                knowledge_elapsed_ms,
            );
        }
        for channel in &recall.trace.channels {
            log::debug!(
                target: "hippocampus::retrieval",
                "retrieval channel event_id={} channel={:?} status={} candidates={} elapsed_ms={} error={:?}",
                current_event_id,
                channel.channel,
                channel.status,
                channel.candidate_count,
                channel.elapsed_ms,
                channel.error,
            );
        }

        let knowledge = knowledge.unwrap_or_else(|| KnowledgeRecall {
            trace: KnowledgeTrace {
                status: if deadline_exceeded {
                    "timeout"
                } else {
                    "failed"
                }
                .into(),
                candidate_limit: self.config.knowledge.candidate_limit,
                max_selected: self.config.knowledge.max_selected,
                evidence_char_budget: self.config.knowledge.evidence_char_budget,
                error: knowledge_error,
                warnings: if deadline_exceeded {
                    vec![format!("知识搜索超过 {deadline_ms}ms，本轮未注入知识结果")]
                } else {
                    Vec::new()
                },
                ..Default::default()
            },
        });

        LiveSearchOutcome {
            recall,
            knowledge,
            query_embedding,
        }
    }

    fn cache_prepared_query_embedding(&self, session: &Session, prepared: &PreparedTurn) {
        if !self.config.memory.enabled {
            return;
        }
        let Some(vector) = prepared.query_embedding.as_deref() else {
            return;
        };
        let Some(turn) = session.turns.get(prepared.turn_index) else {
            return;
        };
        if turn.assistant_content.is_empty() {
            return;
        }
        let Ok(spec) = VectorIndexSpec::from_config(&self.config.memory) else {
            return;
        };
        let _ = self.store.retrieval().cache_content_embedding(
            &spec,
            &content_sha256(&turn.user_content),
            vector,
        );
    }

    fn pause_background_embeddings(&self) {
        let mut cancellation = self
            .background_embedding_cancellation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cancellation.cancel();
        *cancellation = CancellationToken::new();
    }

    fn enqueue_turn_embedding_cache(&self, session: &Session, turn_index: usize) {
        if !self.config.memory.enabled {
            return;
        }
        let Some(turn) = session.turns.get(turn_index) else {
            return;
        };
        if turn.assistant_content.is_empty() {
            return;
        }
        let inputs = turn_leaf_embedding_inputs(&turn.user_content, &turn.assistant_content);
        if inputs.is_empty() {
            return;
        }
        let cancellation = self
            .background_embedding_cancellation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let gate = self
            .background_embedding_gates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(session.id.clone())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone();
        let engine = self.clone();
        self.background_embedding_pending
            .fetch_add(1, Ordering::SeqCst);
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            let _gate = gate.lock().await;
            if !cancellation.is_cancelled() {
                let _ = engine
                    .cache_embedding_inputs(inputs, cancellation.clone())
                    .await;
            }
            engine
                .background_embedding_pending
                .fetch_sub(1, Ordering::SeqCst);
            engine.background_embedding_notify.notify_waiters();
        });
    }

    async fn cache_embedding_inputs(
        &self,
        inputs: Vec<String>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let spec = VectorIndexSpec::from_config(&self.config.memory)?;
        let cached = self.store.retrieval().cached_content_embeddings(&spec)?;
        let mut pending = HashMap::<String, String>::new();
        for input in inputs {
            let hash = content_sha256(&input);
            if !cached.contains_key(&hash) {
                pending.entry(hash).or_insert(input);
            }
        }
        let pending = pending.into_iter().collect::<Vec<_>>();
        for chunk in pending.chunks(self.config.memory.embedding_batch_size) {
            if cancellation.is_cancelled() {
                break;
            }
            let request = EmbeddingRequest {
                model: spec.model.clone(),
                input: chunk.iter().map(|(_, input)| input.clone()).collect(),
                dimensions: Some(spec.dimensions),
                truncate: false,
            };
            let response = tokio::select! {
                _ = cancellation.cancelled() => break,
                response = self.client.embed(request) => response?,
            };
            if response.model != spec.model || response.embeddings.len() != chunk.len() {
                bail!("后台 Embedding 响应与请求不匹配");
            }
            for ((hash, _), vector) in chunk.iter().zip(response.embeddings) {
                let vector = l2_normalize(&vector)?;
                self.store
                    .retrieval()
                    .cache_content_embedding(&spec, hash, &vector)?;
            }
        }
        Ok(())
    }

    pub async fn wait_for_background_embeddings(&self) {
        loop {
            let notified = self.background_embedding_notify.notified();
            if self.background_embedding_pending.load(Ordering::SeqCst) == 0 {
                return;
            }
            notified.await;
        }
    }

    pub async fn prepare_turn(
        &self,
        session: &mut Session,
        user_content: String,
    ) -> Result<PreparedTurn> {
        self.prepare_turn_with_progress(session, user_content, |_| {})
            .await
    }

    pub async fn prepare_turn_with_progress<F>(
        &self,
        session: &mut Session,
        user_content: String,
        mut progress: F,
    ) -> Result<PreparedTurn>
    where
        F: FnMut(PreparationProgress) + Send,
    {
        self.pause_background_embeddings();
        let controls = self.store.control_state()?;
        if !controls.allows_session(&session.id) {
            bail!("会话已排除: {}", session.id);
        }
        if user_content.trim().is_empty() {
            bail!("用户输入不能为空");
        }
        self.recover_stale_pending(session)?;
        session.status = SessionStatus::Active;
        if session.turns.is_empty() {
            let compact = user_content
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let mut title = compact.chars().take(40).collect::<String>();
            if compact.chars().count() > 40 {
                title.push('…');
            }
            session.title = title;
        }

        let start_before = session.active_context_start_index;
        session.turns.push(Turn::pending(user_content.clone()));
        let turn_index = session.turns.len() - 1;
        self.store.save(session)?;

        let prepared = self
            .prepare_persisted_turn(
                session,
                turn_index,
                user_content,
                start_before,
                &controls,
                &mut progress,
            )
            .await;
        if let Err(error) = &prepared
            && session.turns[turn_index].status == TurnStatus::Pending
        {
            let turn = &mut session.turns[turn_index];
            turn.status = TurnStatus::Failed;
            turn.error = Some(error.to_string());
            turn.touch();
            self.store.save(session)?;
        }
        prepared
    }

    #[allow(clippy::collapsible_if)]
    async fn prepare_persisted_turn(
        &self,
        session: &mut Session,
        turn_index: usize,
        user_content: String,
        start_before: usize,
        controls: &crate::control::ControlState,
        progress: &mut (dyn FnMut(PreparationProgress) + Send),
    ) -> Result<PreparedTurn> {
        let history = session
            .eligible_turns(Some(turn_index), true)
            .into_iter()
            .filter(|(_, turn)| controls.allows_turn(&session.id, &turn.id))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let current_event_id = crate::model::event_id(
            &session.id,
            Some(&session.turns[turn_index].id),
            crate::model::EventRole::User,
        );
        let recent_event_ids = history
            .iter()
            .flat_map(|index| {
                let turn = &session.turns[*index];
                [
                    crate::model::event_id(
                        &session.id,
                        Some(&turn.id),
                        crate::model::EventRole::User,
                    ),
                    crate::model::event_id(
                        &session.id,
                        Some(&turn.id),
                        crate::model::EventRole::Assistant,
                    ),
                ]
            })
            .collect::<Vec<_>>();
        let classify_started = Instant::now();
        let memory_query_kind = crate::retrieval::classify_query(&user_content);
        let classify_elapsed = elapsed_millis(classify_started);
        let retrieval_pool_config = crate::retrieval::candidate_pool_config(&session.retrieval);
        let memory_pool_config = self.config.memory.clone();
        let LiveSearchOutcome {
            mut recall,
            knowledge,
            query_embedding,
        } = self
            .live_search(
                &user_content,
                &current_event_id,
                &recent_event_ids,
                retrieval_pool_config.clone(),
                memory_pool_config.clone(),
            )
            .await;
        recall.trace.config = session.retrieval.clone();
        recall.trace.query_kind = memory_query_kind;
        recall.trace.budget_allocation.query_kind = memory_query_kind;
        // Retrieval has succeeded independently of rendering/probing. Persist
        // it now so a later planning failure remains diagnosable.
        session.turns[turn_index].context_trace.retrieval = recall.trace.clone();
        session.turns[turn_index].context_trace.knowledge = knowledge.trace.clone();
        session.turns[turn_index].context_trace.decision = "retrieval_completed".into();
        session.turns[turn_index].touch();
        self.store.save(session)?;
        let mut probe_cache = PreparationProbeCache::default();
        let mut full_groups = Vec::new();
        for group in budget_evidence_groups(&recall) {
            if hard_exclusion_reason(
                &full_groups,
                &group,
                &session.retrieval,
                self.config.memory.graph_candidate_limit,
            )
            .is_none()
            {
                full_groups.push(group);
            }
        }
        let full_recall = recall_for_groups(&recall, &full_groups);
        let mut full_plan = self.assemble_estimated(
            session,
            turn_index,
            &history,
            &user_content,
            &full_recall,
            &knowledge,
        );
        let estimated_full_metric = plan_metric(&full_plan)?;
        if estimated_full_metric >= session.budget.probe_threshold() {
            progress(PreparationProgress::ExactContextCheckStarted {
                estimated_input_tokens: estimated_full_metric,
                probe_threshold: session.budget.probe_threshold(),
                input_budget: session.budget.input_budget(),
            });
            if let Err(error) = self
                .budget_probe_plan(
                    session,
                    &mut full_plan,
                    &mut probe_cache,
                    "near_limit_probe",
                )
                .await
            {
                session.turns[turn_index].context_trace.decision = "probe_failed".into();
                session.turns[turn_index].touch();
                self.store.save(session)?;
                return Err(error);
            }
        }
        let full_metric = plan_metric(&full_plan)?;
        let mut plan = match self
            .allocate_adaptive_plan(
                session,
                turn_index,
                &history,
                &user_content,
                &recall,
                &knowledge,
                classify_elapsed,
                &mut probe_cache,
                if full_metric >= session.budget.warning_threshold() {
                    session.budget.trim_target().saturating_sub(1)
                } else {
                    session.budget.input_budget()
                },
            )
            .await
        {
            Ok(plan) => plan,
            Err(error) => {
                session.turns[turn_index].context_trace.decision = "planning_failed".into();
                let mut failed_trace = recall.trace.clone();
                failed_trace
                    .budget_allocation
                    .stage_latencies
                    .push(BudgetStageLatencyTrace {
                        stage: "classify".into(),
                        elapsed_ms: classify_elapsed,
                    });
                failed_trace
                    .budget_allocation
                    .stage_latencies
                    .push(BudgetStageLatencyTrace {
                        stage: "retrieval_candidates".into(),
                        elapsed_ms: recall.trace.elapsed_ms,
                    });
                failed_trace
                    .budget_allocation
                    .exclusions
                    .push(BudgetExclusionTrace {
                        bucket: BudgetBucket::RecentHistory,
                        candidate_group_id: "local_budget".into(),
                        stage: "error".into(),
                        reason: "planning_failed".into(),
                        exact_increment_tokens: None,
                    });
                session.turns[turn_index].context_trace.retrieval = failed_trace;
                session.turns[turn_index].touch();
                self.store.save(session)?;
                return Err(error);
            }
        };
        if plan.context_sha256 == full_plan.context_sha256 {
            plan.exact_input_tokens = full_plan.exact_input_tokens;
        }
        if let Some(probe) = probe_cache.traces.last() {
            session.turns[turn_index].probe_usage.add(probe.usage);
        }

        if plan
            .retrieval_trace
            .budget_allocation
            .mandatory_input_tokens
            > session.budget.input_budget()
        {
            let mut prepared = self.block_mandatory(session, turn_index, plan, start_before)?;
            prepared.query_embedding = query_embedding;
            return Ok(prepared);
        }

        if full_metric >= session.budget.warning_threshold() {
            apply_trace(
                session,
                turn_index,
                &plan,
                "limit_warning",
                start_before,
                start_before,
            );
            session.turns[turn_index].touch();
            self.store.save(session)?;
            let mut prepared = self.prepared(
                session,
                turn_index,
                plan,
                PreparationStatus::LimitWarning,
                "上下文已达到临界阈值；请选择丢弃最旧完整轮次后继续，或暂停当前会话。",
            );
            prepared.query_embedding = query_embedding;
            return Ok(prepared);
        }

        apply_trace(
            session,
            turn_index,
            &plan,
            "ready",
            start_before,
            start_before,
        );
        session.turns[turn_index].touch();
        self.store.save(session)?;
        let mut prepared = self.prepared(session, turn_index, plan, PreparationStatus::Ready, "");
        prepared.query_embedding = query_embedding;
        Ok(prepared)
    }

    pub async fn resolve_limit(
        &self,
        session: &mut Session,
        prepared: PreparedTurn,
        action: LimitAction,
    ) -> Result<PreparedTurn> {
        if !prepared.needs_limit_decision() {
            bail!("当前轮次不需要上下文临界决策");
        }
        self.pending_turn(session, &prepared)?;
        let start_before = session.active_context_start_index;

        if action == LimitAction::EndSession {
            let turn = &mut session.turns[prepared.turn_index];
            turn.status = TurnStatus::Blocked;
            turn.error = Some("用户选择在上下文临界点暂停会话；消息未发送给模型".into());
            turn.context_trace.decision = "paused_by_user".into();
            turn.touch();
            session.status = SessionStatus::Paused;
            self.store.save(session)?;
            let mut ended = prepared;
            ended.status = PreparationStatus::Ended;
            ended.message = session.turns[ended.turn_index]
                .error
                .clone()
                .unwrap_or_default();
            return Ok(ended);
        }

        let selected_plan = prepared.plan.clone();
        let query_embedding = prepared.query_embedding.clone();
        let budget_allocation = &selected_plan.retrieval_trace.budget_allocation;
        let mandatory_tokens = budget_allocation.mandatory_input_tokens;
        if mandatory_tokens >= session.budget.trim_target() {
            let message = "系统提示与当前输入超过 80% 安全裁剪目标，请缩短系统提示或当前输入";
            apply_trace(
                session,
                prepared.turn_index,
                &selected_plan,
                "mandatory_above_trim_target",
                start_before,
                start_before,
            );
            let turn = &mut session.turns[prepared.turn_index];
            turn.status = TurnStatus::Blocked;
            turn.error = Some(message.into());
            turn.touch();
            session.status = SessionStatus::Paused;
            self.store.save(session)?;
            return Ok(self.prepared(
                session,
                prepared.turn_index,
                selected_plan,
                PreparationStatus::Blocked,
                message,
            ));
        }
        let retained = selected_plan.selected_history_indices.len();
        let new_start = if retained > 0 {
            *selected_plan
                .selected_history_indices
                .first()
                .ok_or_else(|| anyhow!("内部错误：保留上下文没有索引"))?
        } else {
            prepared.turn_index
        };
        session.active_context_start_index = new_start;
        session.status = SessionStatus::Active;
        apply_trace(
            session,
            prepared.turn_index,
            &selected_plan,
            "trimmed_and_continued",
            start_before,
            new_start,
        );
        session.turns[prepared.turn_index].touch();
        self.store.save(session)?;
        let mut resolved = self.prepared(
            session,
            prepared.turn_index,
            selected_plan,
            PreparationStatus::Ready,
            &format!("已保留最近 {retained} 个完整轮次并继续。"),
        );
        resolved.query_embedding = query_embedding;
        Ok(resolved)
    }

    pub async fn stream_turn<F>(
        &self,
        session: &mut Session,
        prepared: &PreparedTurn,
        cancellation: CancellationToken,
        mut emit: F,
    ) -> Result<()>
    where
        F: FnMut(ChatEvent) + Send,
    {
        if !prepared.ready() {
            bail!("轮次尚未准备好，不能生成");
        }
        self.pending_turn(session, prepared)?;
        let turn = &mut session.turns[prepared.turn_index];
        turn.request_started_at = Some(utc_now());
        turn.touch();
        self.store.save(session)?;
        let mut thinking = String::new();
        let mut content = String::new();
        let mut live_output_tokens = 0_u64;
        let mut final_usage = None;
        let mut done_reason = None;
        let request = ChatRequest {
            model: session.model.clone(),
            messages: prepared.plan.messages.clone(),
            think: session.think,
            num_ctx: session.budget.context_window,
            num_predict: session.budget.max_output_tokens,
        };
        let result = self
            .client
            .stream_chat(request, cancellation, &mut |event| {
                if let Some(value) = event.live_output_tokens {
                    live_output_tokens = value;
                }
                match event.kind {
                    ChatEventKind::Thinking => thinking.push_str(&event.text),
                    ChatEventKind::Content => content.push_str(&event.text),
                    ChatEventKind::Completed => {
                        final_usage = event.usage;
                        done_reason.clone_from(&event.done_reason);
                    }
                    ChatEventKind::Usage => {}
                }
                emit(event);
            })
            .await;

        if let Err(error) = result {
            self.persist_stream_error(
                session,
                prepared.turn_index,
                &error,
                StreamSnapshot {
                    thinking: &thinking,
                    content: &content,
                    live_output_tokens: error.live_output_tokens().unwrap_or(live_output_tokens),
                    final_usage,
                },
            )?;
            return Err(error.into());
        }

        let Some(usage) = final_usage else {
            let error = OllamaError::Protocol("模型流在完成事件之前结束".into());
            self.persist_stream_error(
                session,
                prepared.turn_index,
                &error,
                StreamSnapshot {
                    thinking: &thinking,
                    content: &content,
                    live_output_tokens,
                    final_usage: None,
                },
            )?;
            return Err(error.into());
        };
        if prepared.plan.exact_input_tokens.is_some()
            && prepared.plan.exact_input_tokens != usage.input_tokens
        {
            let error = OllamaError::Protocol(
                "精确探测与正式请求的输入 token 不一致；拒绝将该轮加入上下文".into(),
            );
            self.persist_stream_error(
                session,
                prepared.turn_index,
                &error,
                StreamSnapshot {
                    thinking: &thinking,
                    content: &content,
                    live_output_tokens,
                    final_usage: Some(usage),
                },
            )?;
            return Err(error.into());
        }

        let turn = &mut session.turns[prepared.turn_index];
        turn.thinking = thinking;
        turn.assistant_content = content;
        turn.usage = usage;
        turn.done_reason = done_reason;
        turn.context_trace.exact_input_tokens = usage.input_tokens;
        if turn.assistant_content.is_empty() {
            turn.status = TurnStatus::NoAnswer;
            turn.error = Some("模型未返回可作为后续上下文的正文".into());
        } else if turn.done_reason.as_deref() == Some("length") {
            turn.status = TurnStatus::Truncated;
            turn.error = Some("回答达到输出 token 上限，正文可能不完整".into());
        } else {
            turn.status = TurnStatus::Complete;
            turn.error = None;
        }
        turn.touch();
        session.status = SessionStatus::Active;
        self.store.save(session)?;
        self.cache_prepared_query_embedding(session, prepared);
        self.enqueue_turn_embedding_cache(session, prepared.turn_index);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_estimated(
        &self,
        session: &Session,
        turn_index: usize,
        history: &[usize],
        user_content: &str,
        recall: &RecallResult,
        knowledge: &KnowledgeRecall,
    ) -> ContextPlan {
        self.assembler.assemble_with_recall_and_knowledge(
            session,
            user_content,
            Some(history),
            Some(turn_index),
            Some(recall),
            Some(knowledge),
        )
    }

    #[allow(clippy::too_many_arguments)]
    async fn allocate_adaptive_plan(
        &self,
        session: &mut Session,
        turn_index: usize,
        history: &[usize],
        user_content: &str,
        recall: &RecallResult,
        knowledge: &KnowledgeRecall,
        classify_elapsed: u64,
        probe_cache: &mut PreparationProbeCache,
        allocation_limit: u64,
    ) -> Result<ContextPlan> {
        let empty_recall = RecallResult {
            trace: recall.trace.clone(),
            evidence: Vec::new(),
        };
        let mandatory_started = Instant::now();
        let mandatory = self.assemble_estimated(
            session,
            turn_index,
            &[],
            user_content,
            &empty_recall,
            knowledge,
        );
        let mandatory_tokens = plan_metric(&mandatory)?;
        let input_budget = session.budget.input_budget();
        let available = allocation_limit.saturating_sub(mandatory_tokens);
        let mut budget = recall.trace.budget_allocation.clone();
        budget.mandatory_input_tokens = mandatory_tokens;
        budget.available_input_tokens = available;
        budget.initial_tokens = BudgetTokenBreakdown {
            recent_history: available * u64::from(budget.recent_history_percent) / 100,
            exact_or_state: available * u64::from(budget.exact_or_state_percent) / 100,
            episode: available * u64::from(budget.episode_percent) / 100,
            graph: available * u64::from(budget.graph_percent) / 100,
        };
        budget.stage_latencies.push(BudgetStageLatencyTrace {
            stage: "classify".into(),
            elapsed_ms: classify_elapsed,
        });
        budget.stage_latencies.push(BudgetStageLatencyTrace {
            stage: "retrieval_candidates".into(),
            elapsed_ms: recall.trace.elapsed_ms,
        });
        budget.stage_latencies.push(BudgetStageLatencyTrace {
            stage: "mandatory_estimate".into(),
            elapsed_ms: elapsed_millis(mandatory_started),
        });
        if mandatory_tokens > input_budget {
            let mut blocked = mandatory;
            budget.probes = probe_cache.traces.clone();
            blocked.retrieval_trace.budget_allocation = budget;
            return Ok(blocked);
        }

        let groups = budget_evidence_groups(recall);
        for evidence in recall
            .evidence
            .iter()
            .filter(|evidence| evidence.selected.kind == EvidenceKind::Context)
        {
            if !groups.iter().any(|group| {
                group
                    .evidence
                    .iter()
                    .any(|candidate| candidate.selected.span == evidence.selected.span)
            }) {
                budget.exclusions.push(BudgetExclusionTrace {
                    bucket: BudgetBucket::ExactOrState,
                    candidate_group_id: format!(
                        "grp_{}",
                        content_sha256(&format!(
                            "hippocampus:budget-orphan-context:v1\0{}\0{}\0{}\0{}",
                            evidence.selected.span.event_id,
                            evidence.selected.span.start_char,
                            evidence.selected.span.end_char,
                            evidence.selected.content_sha256
                        ))
                    ),
                    stage: "hard".into(),
                    reason: "dependency_not_selected".into(),
                    exact_increment_tokens: None,
                });
            }
        }
        let mut accepted_history = Vec::<usize>::new();
        let mut accepted_groups = Vec::<BudgetEvidenceGroup>::new();
        let mut current_tokens = mandatory_tokens;
        let mut deferred_history = Vec::<usize>::new();
        let mut deferred_groups = Vec::<BudgetEvidenceGroup>::new();
        let mut rejected_increments = HashMap::<String, u64>::new();
        let mut acceptance_log = Vec::<AcceptedBudgetUnit>::new();

        let recent_started = Instant::now();
        let recent_limit = mandatory_tokens.saturating_add(budget.initial_tokens.recent_history);
        for index in history.iter().rev() {
            let mut proposed = accepted_history.clone();
            proposed.insert(0, *index);
            let candidate = self.assemble_estimated(
                session,
                turn_index,
                &proposed,
                user_content,
                &empty_recall,
                knowledge,
            );
            let tokens = plan_metric(&candidate)?;
            if tokens <= recent_limit && tokens <= allocation_limit {
                accepted_history = proposed;
                budget.actual_tokens.recent_history = budget
                    .actual_tokens
                    .recent_history
                    .saturating_add(tokens.saturating_sub(current_tokens));
                acceptance_log.push(AcceptedBudgetUnit::Recent {
                    index: *index,
                    metric_before: current_tokens,
                    metric_after: tokens,
                    reflow: false,
                });
                current_tokens = tokens;
            } else {
                rejected_increments.insert(
                    format!("turn:{}", session.turns[*index].id),
                    tokens.saturating_sub(current_tokens),
                );
                let position = history
                    .iter()
                    .position(|candidate| candidate == index)
                    .expect("history index came from history");
                deferred_history.extend_from_slice(&history[..=position]);
                break;
            }
        }
        budget.stage_latencies.push(BudgetStageLatencyTrace {
            stage: "initial_recent".into(),
            elapsed_ms: elapsed_millis(recent_started),
        });

        for bucket in [
            BudgetBucket::ExactOrState,
            BudgetBucket::Episode,
            BudgetBucket::Graph,
        ] {
            let started = Instant::now();
            let share = breakdown_value(&budget.initial_tokens, bucket);
            let bucket_base = current_tokens;
            for group in groups.iter().filter(|group| group.bucket == bucket) {
                if hard_exclusion_reason(
                    &accepted_groups,
                    group,
                    &session.retrieval,
                    self.config.memory.graph_candidate_limit,
                )
                .is_some()
                {
                    deferred_groups.push(group.clone());
                    continue;
                }
                let mut proposed_groups = accepted_groups.clone();
                proposed_groups.push(group.clone());
                let proposed_recall = recall_for_groups(recall, &proposed_groups);
                let candidate = self.assemble_estimated(
                    session,
                    turn_index,
                    &accepted_history,
                    user_content,
                    &proposed_recall,
                    knowledge,
                );
                let tokens = plan_metric(&candidate)?;
                let increment = tokens.saturating_sub(current_tokens);
                if tokens <= allocation_limit && tokens.saturating_sub(bucket_base) <= share {
                    accepted_groups = proposed_groups;
                    add_breakdown(&mut budget.actual_tokens, bucket, increment);
                    acceptance_log.push(AcceptedBudgetUnit::Evidence {
                        id: group.id.clone(),
                        bucket,
                        metric_before: current_tokens,
                        metric_after: tokens,
                        reflow: false,
                    });
                    current_tokens = tokens;
                } else {
                    rejected_increments.insert(group.id.clone(), increment);
                    deferred_groups.push(group.clone());
                }
            }
            budget.stage_latencies.push(BudgetStageLatencyTrace {
                stage: format!("initial_{}", budget_bucket_name(bucket)),
                elapsed_ms: elapsed_millis(started),
            });
        }

        let initially_consumed = breakdown_sum(&budget.actual_tokens);
        let mut reflow_pool = available.saturating_sub(initially_consumed);
        for bucket in [
            BudgetBucket::RecentHistory,
            BudgetBucket::ExactOrState,
            BudgetBucket::Episode,
            BudgetBucket::Graph,
        ] {
            let started = Instant::now();
            let offered = reflow_pool;
            let before = current_tokens;
            if bucket == BudgetBucket::RecentHistory {
                for index in deferred_history.iter().rev() {
                    let mut proposed = accepted_history.clone();
                    proposed.insert(0, *index);
                    let proposed_recall = recall_for_groups(recall, &accepted_groups);
                    let candidate = self.assemble_estimated(
                        session,
                        turn_index,
                        &proposed,
                        user_content,
                        &proposed_recall,
                        knowledge,
                    );
                    let tokens = plan_metric(&candidate)?;
                    let increment = tokens.saturating_sub(current_tokens);
                    if tokens <= allocation_limit && increment <= reflow_pool {
                        accepted_history = proposed;
                        add_breakdown(&mut budget.actual_tokens, bucket, increment);
                        acceptance_log.push(AcceptedBudgetUnit::Recent {
                            index: *index,
                            metric_before: current_tokens,
                            metric_after: tokens,
                            reflow: true,
                        });
                        reflow_pool = reflow_pool.saturating_sub(increment);
                        current_tokens = tokens;
                    } else {
                        rejected_increments
                            .insert(format!("turn:{}", session.turns[*index].id), increment);
                        break;
                    }
                }
            } else {
                let candidates = deferred_groups
                    .iter()
                    .filter(|group| group.bucket == bucket)
                    .cloned()
                    .collect::<Vec<_>>();
                for group in candidates {
                    if hard_exclusion_reason(
                        &accepted_groups,
                        &group,
                        &session.retrieval,
                        self.config.memory.graph_candidate_limit,
                    )
                    .is_some()
                    {
                        continue;
                    }
                    let mut proposed_groups = accepted_groups.clone();
                    proposed_groups.push(group.clone());
                    let proposed_recall = recall_for_groups(recall, &proposed_groups);
                    let candidate = self.assemble_estimated(
                        session,
                        turn_index,
                        &accepted_history,
                        user_content,
                        &proposed_recall,
                        knowledge,
                    );
                    let tokens = plan_metric(&candidate)?;
                    let increment = tokens.saturating_sub(current_tokens);
                    if tokens <= allocation_limit && increment <= reflow_pool {
                        accepted_groups = proposed_groups;
                        add_breakdown(&mut budget.actual_tokens, bucket, increment);
                        acceptance_log.push(AcceptedBudgetUnit::Evidence {
                            id: group.id.clone(),
                            bucket,
                            metric_before: current_tokens,
                            metric_after: tokens,
                            reflow: true,
                        });
                        reflow_pool = reflow_pool.saturating_sub(increment);
                        current_tokens = tokens;
                    } else {
                        rejected_increments.insert(group.id.clone(), increment);
                    }
                }
            }
            budget.reflow.push(BudgetReflowTrace {
                bucket,
                offered_tokens: offered,
                consumed_tokens: current_tokens.saturating_sub(before),
                remaining_tokens: reflow_pool,
            });
            budget.stage_latencies.push(BudgetStageLatencyTrace {
                stage: format!("reflow_{}", budget_bucket_name(bucket)),
                elapsed_ms: elapsed_millis(started),
            });
        }

        for group in &groups {
            if !accepted_groups.iter().any(|value| value.id == group.id) {
                let hard_reason = hard_exclusion_reason(
                    &accepted_groups,
                    group,
                    &session.retrieval,
                    self.config.memory.graph_candidate_limit,
                );
                budget.exclusions.push(BudgetExclusionTrace {
                    bucket: group.bucket,
                    candidate_group_id: group.id.clone(),
                    stage: if hard_reason.is_some() {
                        "hard"
                    } else {
                        "reflow"
                    }
                    .into(),
                    reason: hard_reason.unwrap_or("token_budget").into(),
                    exact_increment_tokens: rejected_increments.get(&group.id).copied(),
                });
            }
        }
        for index in history {
            if !accepted_history.contains(index) {
                let group_id = format!("turn:{}", session.turns[*index].id);
                budget.exclusions.push(BudgetExclusionTrace {
                    bucket: BudgetBucket::RecentHistory,
                    candidate_group_id: group_id.clone(),
                    stage: "reflow".into(),
                    reason: "token_budget".into(),
                    exact_increment_tokens: rejected_increments.get(&group_id).copied(),
                });
            }
        }

        let final_started = Instant::now();
        let mut final_recall = recall_for_groups(recall, &accepted_groups);
        let mut current = self.assemble_estimated(
            session,
            turn_index,
            &accepted_history,
            user_content,
            &final_recall,
            knowledge,
        );
        current_tokens = plan_metric(&current)?;
        while current_tokens > allocation_limit {
            let Some(removed) = acceptance_log.pop() else {
                budget.mandatory_input_tokens = current_tokens;
                budget.available_input_tokens = 0;
                break;
            };
            let (bucket, group_id, removed_increment) = match removed {
                AcceptedBudgetUnit::Recent {
                    index,
                    metric_before,
                    metric_after,
                    ..
                } => {
                    accepted_history.retain(|candidate| *candidate != index);
                    (
                        BudgetBucket::RecentHistory,
                        format!("turn:{}", session.turns[index].id),
                        metric_after.saturating_sub(metric_before),
                    )
                }
                AcceptedBudgetUnit::Evidence {
                    id,
                    bucket,
                    metric_before,
                    metric_after,
                    ..
                } => {
                    accepted_groups.retain(|group| group.id != id);
                    (bucket, id, metric_after.saturating_sub(metric_before))
                }
            };
            budget.exclusions.retain(|exclusion| {
                exclusion.candidate_group_id != group_id || exclusion.stage == "final_estimate"
            });
            if !budget.exclusions.iter().any(|exclusion| {
                exclusion.candidate_group_id == group_id && exclusion.stage == "final_estimate"
            }) {
                budget.exclusions.push(BudgetExclusionTrace {
                    bucket,
                    candidate_group_id: group_id,
                    stage: "final_estimate".into(),
                    reason: "final_estimate_over_budget".into(),
                    exact_increment_tokens: Some(removed_increment),
                });
            }
            final_recall = recall_for_groups(recall, &accepted_groups);
            current = self.assemble_estimated(
                session,
                turn_index,
                &accepted_history,
                user_content,
                &final_recall,
                knowledge,
            );
            current_tokens = plan_metric(&current)?;
        }
        budget.actual_tokens = BudgetTokenBreakdown::default();
        let mut attributed = Vec::<(BudgetBucket, bool, u64)>::new();
        for accepted in &acceptance_log {
            let (before, after) = accepted.metrics();
            if after >= before {
                attributed.push((accepted.bucket(), accepted.is_reflow(), after - before));
            } else {
                let mut credit = before - after;
                for (_, _, amount) in attributed.iter_mut().rev() {
                    let applied = credit.min(*amount);
                    *amount -= applied;
                    credit -= applied;
                    if credit == 0 {
                        break;
                    }
                }
            }
        }
        let target_actual = current_tokens.saturating_sub(budget.mandatory_input_tokens);
        let attributed_total = attributed.iter().map(|(_, _, amount)| *amount).sum::<u64>();
        if attributed_total > target_actual {
            let mut credit = attributed_total - target_actual;
            for (_, _, amount) in attributed.iter_mut().rev() {
                let applied = credit.min(*amount);
                *amount -= applied;
                credit -= applied;
                if credit == 0 {
                    break;
                }
            }
        } else if attributed_total < target_actual
            && let Some(last) = attributed.last_mut()
        {
            last.2 = last.2.saturating_add(target_actual - attributed_total);
        }
        for (bucket, _, amount) in &attributed {
            add_breakdown(&mut budget.actual_tokens, *bucket, *amount);
        }
        let initial_consumed = attributed
            .iter()
            .filter(|(_, reflow, _)| !reflow)
            .map(|(_, _, amount)| *amount)
            .sum::<u64>();
        let mut remaining = available.saturating_sub(initial_consumed);
        budget.reflow.clear();
        for bucket in [
            BudgetBucket::RecentHistory,
            BudgetBucket::ExactOrState,
            BudgetBucket::Episode,
            BudgetBucket::Graph,
        ] {
            let offered = remaining;
            let consumed = attributed
                .iter()
                .filter(|(accepted_bucket, reflow, _)| *reflow && *accepted_bucket == bucket)
                .map(|(_, _, amount)| *amount)
                .sum::<u64>();
            remaining = remaining.saturating_sub(consumed);
            budget.reflow.push(BudgetReflowTrace {
                bucket,
                offered_tokens: offered,
                consumed_tokens: consumed,
                remaining_tokens: remaining,
            });
        }
        budget.final_input_tokens = Some(current_tokens);
        budget.stage_latencies.push(BudgetStageLatencyTrace {
            stage: "final_estimate".into(),
            elapsed_ms: elapsed_millis(final_started),
        });
        apply_budget_exclusion_reasons(&mut current.retrieval_trace, &groups, &budget.exclusions);
        budget.probes = probe_cache.traces.clone();
        current.retrieval_trace.budget_allocation = budget;
        Ok(current)
    }

    async fn budget_probe_plan(
        &self,
        session: &Session,
        plan: &mut ContextPlan,
        cache: &mut PreparationProbeCache,
        stage: &str,
    ) -> Result<()> {
        let (request_sha256, kind, usage) = {
            let request_sha256 = content_sha256(
                &serde_json::to_string(&json!({
                    "kind": "normal",
                    "model": session.model,
                    "messages": plan.messages,
                    "think": session.think,
                    "num_ctx": session.budget.context_window,
                    "num_predict": session.budget.max_output_tokens,
                }))
                .expect("normal request is serializable"),
            );
            if let Some(usage) = cache.usages.get(&request_sha256).copied() {
                (request_sha256, "normal", usage)
            } else {
                let usage = match self
                    .client
                    .probe(
                        &session.model,
                        &plan.messages,
                        session.think,
                        session.budget.context_window,
                    )
                    .await
                {
                    Ok(usage) => usage,
                    Err(OllamaError::ContextLength { prompt_tokens, .. }) => TokenUsage::new(
                        Some(prompt_tokens.unwrap_or(session.budget.context_window + 1)),
                        Some(0),
                    ),
                    Err(error) => return Err(error.into()),
                };
                cache.usages.insert(request_sha256.clone(), usage);
                (request_sha256, "normal", usage)
            }
        };
        let cache_hit = cache
            .traces
            .iter()
            .any(|trace| trace.request_sha256 == request_sha256);
        cache.traces.push(BudgetProbeTrace {
            stage: stage.into(),
            request_sha256,
            kind: kind.into(),
            usage,
            cache_hit,
        });
        plan.exact_input_tokens = usage.input_tokens;
        Ok(())
    }

    fn block_mandatory(
        &self,
        session: &mut Session,
        turn_index: usize,
        plan: ContextPlan,
        start_before: usize,
    ) -> Result<PreparedTurn> {
        let message = "系统提示与当前输入本身已超过输入预算；请缩短输入或提高上下文配置";
        apply_trace(
            session,
            turn_index,
            &plan,
            "mandatory_input_exceeded",
            start_before,
            start_before,
        );
        let turn = &mut session.turns[turn_index];
        turn.status = TurnStatus::Blocked;
        turn.error = Some(message.into());
        turn.touch();
        session.status = SessionStatus::Paused;
        self.store.save(session)?;
        Ok(self.prepared(
            session,
            turn_index,
            plan,
            PreparationStatus::Blocked,
            message,
        ))
    }

    fn persist_stream_error(
        &self,
        session: &mut Session,
        turn_index: usize,
        error: &OllamaError,
        snapshot: StreamSnapshot<'_>,
    ) -> Result<()> {
        let turn = &mut session.turns[turn_index];
        turn.thinking = snapshot.thinking.to_owned();
        turn.assistant_content = snapshot.content.to_owned();
        turn.usage = snapshot.final_usage.unwrap_or_else(|| {
            TokenUsage::new(
                None,
                (snapshot.live_output_tokens > 0).then_some(snapshot.live_output_tokens),
            )
        });
        match error {
            OllamaError::ContextLength { .. } => {
                turn.status = TurnStatus::Blocked;
                session.status = SessionStatus::Paused;
            }
            OllamaError::Cancelled { .. } => {
                turn.status = TurnStatus::Interrupted;
                session.status = SessionStatus::Paused;
            }
            _ if !snapshot.thinking.is_empty()
                || !snapshot.content.is_empty()
                || snapshot.live_output_tokens > 0 =>
            {
                turn.status = TurnStatus::Interrupted;
            }
            _ => turn.status = TurnStatus::Failed,
        }
        turn.error = Some(error.to_string());
        turn.touch();
        self.store.save(session)?;
        Ok(())
    }

    fn pending_turn(&self, session: &Session, prepared: &PreparedTurn) -> Result<()> {
        if session.id != prepared.session_id {
            bail!("prepared turn belongs to a different session");
        }
        let turn = session
            .turns
            .get(prepared.turn_index)
            .ok_or_else(|| anyhow!("prepared turn index is no longer valid"))?;
        if turn.id != prepared.turn_id || turn.status != TurnStatus::Pending {
            bail!("prepared turn no longer references a pending turn");
        }
        Ok(())
    }

    fn recover_stale_pending(&self, session: &mut Session) -> Result<()> {
        let mut changed = false;
        for turn in &mut session.turns {
            if turn.status == TurnStatus::Pending {
                turn.status = TurnStatus::Interrupted;
                turn.error = Some("上次进程在该轮完成前退出".into());
                turn.touch();
                changed = true;
            }
        }
        if changed {
            self.store.save(session)?;
        }
        Ok(())
    }

    fn prepared(
        &self,
        session: &Session,
        turn_index: usize,
        plan: ContextPlan,
        status: PreparationStatus,
        message: &str,
    ) -> PreparedTurn {
        PreparedTurn {
            session_id: session.id.clone(),
            turn_id: session.turns[turn_index].id.clone(),
            turn_index,
            plan,
            status,
            message: message.to_owned(),
            query_embedding: None,
        }
    }
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn empty_live_recall(
    current_event_id: &str,
    retrieval_config: crate::model::RetrievalConfig,
    memory_config: &crate::config::MemoryConfig,
    error: Option<&str>,
) -> RecallResult {
    let query_kind = crate::model::QueryKind::GeneralSemantic;
    RecallResult {
        trace: RetrievalTrace {
            status: "fast_search_unavailable".into(),
            current_query_event_id: current_event_id.into(),
            config: retrieval_config,
            error: error.map(str::to_owned),
            query_kind,
            budget_allocation: crate::retrieval::memory_budget_trace(memory_config, query_kind),
            ..Default::default()
        },
        evidence: Vec::new(),
    }
}

fn turn_leaf_embedding_inputs(user: &str, assistant: &str) -> Vec<String> {
    [user, assistant]
        .into_iter()
        .filter(|content| !content.trim().is_empty())
        .flat_map(|content| {
            let characters = content.chars().collect::<Vec<_>>();
            if characters.len() <= 240 {
                return vec![content.to_owned()];
            }
            let mut fragments = Vec::new();
            let mut start = 0usize;
            while start < characters.len() {
                let end = start.saturating_add(240).min(characters.len());
                fragments.push(characters[start..end].iter().collect());
                if end == characters.len() {
                    break;
                }
                start = start.saturating_add(200);
            }
            fragments
        })
        .collect()
}

fn ensure_live_channel_statuses(
    trace: &mut RetrievalTrace,
    memory_enabled: bool,
    deadline_exceeded: bool,
    fast_available: bool,
    bm25_elapsed_ms: u64,
    vector_elapsed_ms: u64,
) {
    let semantic_status = if !memory_enabled {
        "disabled"
    } else if trace.fast_fallback_used && deadline_exceeded {
        "timeout"
    } else if trace.fast_fallback_used {
        "error"
    } else {
        "ok"
    };
    let expected = [
        (
            RetrievalChannel::Bm25,
            if fast_available {
                "ok"
            } else if deadline_exceeded {
                "timeout"
            } else {
                "error"
            },
            bm25_elapsed_ms,
        ),
        (RetrievalChannel::Vector, semantic_status, vector_elapsed_ms),
        (RetrievalChannel::Entity, semantic_status, 0),
        (RetrievalChannel::State, semantic_status, 0),
        (RetrievalChannel::Episode, semantic_status, 0),
        (RetrievalChannel::Graph, semantic_status, 0),
    ];
    for (channel, status, elapsed_ms) in expected {
        if let Some(existing) = trace
            .channels
            .iter_mut()
            .find(|existing| existing.channel == channel)
        {
            if channel == RetrievalChannel::Bm25 {
                existing.status = status.into();
                if existing.elapsed_ms == 0 {
                    existing.elapsed_ms = elapsed_ms;
                }
            } else if !memory_enabled || trace.fast_fallback_used {
                existing.status = status.into();
                if status == "timeout" {
                    existing.error = Some("live search deadline exceeded".into());
                } else if status == "error" && existing.error.is_none() {
                    existing.error = Some("live semantic search unavailable".into());
                }
            } else if existing.status == "empty" {
                existing.status = "ok".into();
            }
            continue;
        }
        trace.channels.push(ChannelTrace {
            channel,
            status: status.into(),
            candidate_count: if channel == RetrievalChannel::Bm25 {
                trace.candidates.len()
            } else {
                0
            },
            elapsed_ms,
            error: match status {
                "timeout" => Some("live search deadline exceeded".into()),
                "error" => Some("live search unavailable".into()),
                _ => None,
            },
        });
    }
}

fn plan_metric(plan: &ContextPlan) -> Result<u64> {
    plan.exact_input_tokens
        .or(plan.estimated_upper_tokens)
        .ok_or_else(|| anyhow!("上下文计划缺少精确或估计 token 数"))
}

fn budget_bucket_name(bucket: BudgetBucket) -> &'static str {
    match bucket {
        BudgetBucket::RecentHistory => "recent",
        BudgetBucket::ExactOrState => "exact_or_state",
        BudgetBucket::Episode => "episode",
        BudgetBucket::Graph => "graph",
    }
}

fn breakdown_value(value: &BudgetTokenBreakdown, bucket: BudgetBucket) -> u64 {
    match bucket {
        BudgetBucket::RecentHistory => value.recent_history,
        BudgetBucket::ExactOrState => value.exact_or_state,
        BudgetBucket::Episode => value.episode,
        BudgetBucket::Graph => value.graph,
    }
}

fn add_breakdown(value: &mut BudgetTokenBreakdown, bucket: BudgetBucket, tokens: u64) {
    match bucket {
        BudgetBucket::RecentHistory => {
            value.recent_history = value.recent_history.saturating_add(tokens)
        }
        BudgetBucket::ExactOrState => {
            value.exact_or_state = value.exact_or_state.saturating_add(tokens)
        }
        BudgetBucket::Episode => value.episode = value.episode.saturating_add(tokens),
        BudgetBucket::Graph => value.graph = value.graph.saturating_add(tokens),
    }
}

fn breakdown_sum(value: &BudgetTokenBreakdown) -> u64 {
    value
        .recent_history
        .saturating_add(value.exact_or_state)
        .saturating_add(value.episode)
        .saturating_add(value.graph)
}

fn hard_exclusion_reason(
    accepted: &[BudgetEvidenceGroup],
    candidate: &BudgetEvidenceGroup,
    config: &crate::model::RetrievalConfig,
    graph_candidate_limit: usize,
) -> Option<&'static str> {
    let all = accepted.iter().chain(std::iter::once(candidate));
    let core_slots = all
        .clone()
        .flat_map(|group| &group.evidence)
        .filter(|item| item.selected.kind == EvidenceKind::Core)
        .count();
    if core_slots > config.max_selected {
        return Some("hard_max_selected");
    }
    let core_chars = all
        .clone()
        .flat_map(|group| &group.evidence)
        .filter(|item| item.selected.kind == EvidenceKind::Core)
        .map(|item| item.content.chars().count())
        .sum::<usize>();
    if core_chars > config.evidence_char_budget {
        return Some("hard_core_chars");
    }
    let context_chars = all
        .clone()
        .flat_map(|group| &group.evidence)
        .filter(|item| item.selected.kind == EvidenceKind::Context)
        .map(|item| item.content.chars().count())
        .sum::<usize>();
    if context_chars > config.expansion_char_budget {
        return Some("hard_context_chars");
    }
    let graph_groups = all
        .filter(|group| group.bucket == BudgetBucket::Graph)
        .count();
    (graph_groups > graph_candidate_limit).then_some("hard_max_selected")
}

fn budget_evidence_groups(recall: &RecallResult) -> Vec<BudgetEvidenceGroup> {
    let state_spans = recall
        .trace
        .state_selections
        .iter()
        .filter_map(|state| {
            state.evidence_span.as_ref().map(|span| {
                let mut claims = state.related_claim_ids.clone();
                claims.push(state.claim_id.clone());
                claims.sort();
                claims.dedup();
                (span, format!("state:{}", claims.join(",")))
            })
        })
        .collect::<HashMap<_, _>>();
    let mut groups = Vec::<BudgetEvidenceGroup>::new();
    for evidence in &recall.evidence {
        if evidence.selected.kind == EvidenceKind::Context {
            let matching = groups
                .iter()
                .enumerate()
                .filter(|(_, group)| {
                    group.evidence.first().is_some_and(|core| {
                        core.selected.kind == EvidenceKind::Core
                            && core.selected.originating_candidate_rank
                                == evidence.selected.originating_candidate_rank
                    })
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            if let [parent] = matching.as_slice() {
                groups[*parent].evidence.push(evidence.clone());
            }
            continue;
        }
        let fusion_matches = recall
            .trace
            .fusion_candidates
            .iter()
            .filter(|candidate| candidate.span == evidence.selected.span)
            .collect::<Vec<_>>();
        let fusion = match fusion_matches.as_slice() {
            [candidate] => Some(*candidate),
            _ => None,
        };
        let graph_path = recall.trace.graph_paths.iter().find(|path| {
            path.span.as_ref() == Some(&evidence.selected.span)
                || fusion.is_some_and(|candidate| {
                    path.target_document_id == candidate.document_id
                        || candidate
                            .source_document_ids
                            .contains(&path.target_document_id)
                })
        });
        let bucket = if state_spans.contains_key(&evidence.selected.span) {
            BudgetBucket::ExactOrState
        } else if graph_path.is_some() {
            BudgetBucket::Graph
        } else if fusion.is_some_and(|candidate| {
            matches!(
                candidate.granularity,
                RetrievalDocumentGranularity::Episode | RetrievalDocumentGranularity::Session
            ) || candidate.episode_id.is_some()
                && candidate
                    .source_document_ids
                    .iter()
                    .any(|source| source.starts_with("episode:") || source.starts_with("session:"))
        }) {
            BudgetBucket::Episode
        } else {
            BudgetBucket::ExactOrState
        };
        let state_identity = state_spans
            .get(&evidence.selected.span)
            .cloned()
            .unwrap_or_default();
        let graph_identity = graph_path
            .map(|path| {
                format!(
                    "{}:{}:{}",
                    path.target_document_id,
                    path.node_ids.join(","),
                    path.edge_ids.join(",")
                )
            })
            .unwrap_or_default();
        let identity = format!(
            "hippocampus:budget-group:v1\0{}\0{}\0{}\0{}\0{}\0{}",
            budget_bucket_name(bucket),
            evidence.selected.span.event_id,
            evidence.selected.span.start_char,
            evidence.selected.span.end_char,
            evidence.selected.content_sha256,
            evidence.selected.role.as_str(),
        );
        let id = if !state_identity.is_empty() {
            format!(
                "grp_{}",
                content_sha256(&format!(
                    "hippocampus:budget-state-group:v1\0{state_identity}"
                ))
            )
        } else if !graph_identity.is_empty() {
            format!(
                "grp_{}",
                content_sha256(&format!("{identity}\0{graph_identity}"))
            )
        } else {
            format!("grp_{}", content_sha256(&identity))
        };
        if let Some(index) = groups.iter().position(|group| group.id == id) {
            groups[index].evidence.push(evidence.clone());
            continue;
        }
        groups.push(BudgetEvidenceGroup {
            id,
            bucket,
            order: groups.len(),
            evidence: vec![evidence.clone()],
        });
    }
    groups
}

fn recall_for_groups(recall: &RecallResult, groups: &[BudgetEvidenceGroup]) -> RecallResult {
    let mut evidence = Vec::new();
    for bucket in [
        BudgetBucket::ExactOrState,
        BudgetBucket::Episode,
        BudgetBucket::Graph,
    ] {
        let mut ordered = groups
            .iter()
            .filter(|group| group.bucket == bucket)
            .collect::<Vec<_>>();
        ordered.sort_by_key(|group| group.order);
        for group in ordered {
            evidence.extend(group.evidence.clone());
        }
    }
    let selected_spans = evidence
        .iter()
        .map(|item| &item.selected.span)
        .collect::<HashSet<_>>();
    let mut trace = recall.trace.clone();
    trace.selected_evidence = evidence.iter().map(|item| item.selected.clone()).collect();
    for candidate in &mut trace.candidates {
        candidate.selected = selected_spans.contains(&candidate.span);
        if !candidate.selected && candidate.reason == "selected_core" {
            candidate.reason = "token_budget".into();
        }
    }
    for candidate in &mut trace.fusion_candidates {
        candidate.selected = selected_spans.contains(&candidate.span);
        if !candidate.selected && candidate.reason.starts_with("selected") {
            candidate.reason = "token_budget".into();
        }
    }
    for state in &mut trace.state_selections {
        state.selected = state
            .evidence_span
            .as_ref()
            .is_some_and(|span| selected_spans.contains(span));
        if !state.selected && state.reason.starts_with("selected") {
            state.reason = "token_budget".into();
        }
    }
    for path in &mut trace.graph_paths {
        path.selected = trace.fusion_candidates.iter().any(|candidate| {
            candidate.selected
                && (candidate.document_id == path.target_document_id
                    || candidate
                        .source_document_ids
                        .contains(&path.target_document_id))
        }) || path
            .span
            .as_ref()
            .is_some_and(|span| selected_spans.contains(span));
        if !path.selected && path.reason.starts_with("selected") {
            path.reason = "token_budget".into();
        }
    }
    RecallResult { trace, evidence }
}

fn apply_budget_exclusion_reasons(
    trace: &mut crate::model::RetrievalTrace,
    groups: &[BudgetEvidenceGroup],
    exclusions: &[BudgetExclusionTrace],
) {
    for exclusion in exclusions {
        let Some(group) = groups
            .iter()
            .find(|group| group.id == exclusion.candidate_group_id)
        else {
            continue;
        };
        let spans = group
            .evidence
            .iter()
            .map(|item| &item.selected.span)
            .collect::<HashSet<_>>();
        for candidate in &mut trace.candidates {
            if spans.contains(&candidate.span) {
                candidate.selected = false;
                candidate.reason = exclusion.reason.clone();
            }
        }
        for candidate in &mut trace.fusion_candidates {
            if spans.contains(&candidate.span) {
                candidate.selected = false;
                candidate.reason = exclusion.reason.clone();
            }
        }
        for state in &mut trace.state_selections {
            if state
                .evidence_span
                .as_ref()
                .is_some_and(|span| spans.contains(span))
            {
                state.selected = false;
                state.reason = exclusion.reason.clone();
            }
        }
        for path in &mut trace.graph_paths {
            if path.span.as_ref().is_some_and(|span| spans.contains(span)) {
                path.selected = false;
                path.reason = exclusion.reason.clone();
            }
        }
    }
}

fn apply_trace(
    session: &mut Session,
    turn_index: usize,
    plan: &ContextPlan,
    decision: &str,
    start_before: usize,
    start_after: usize,
) {
    let request = ModelRequestTrace {
        model: session.model.clone(),
        think: session.think,
        context_window: session.budget.context_window,
        max_output_tokens: session.budget.max_output_tokens,
    };
    let turn = &mut session.turns[turn_index];
    turn.context_trace = ContextTrace {
        included_turn_ids: plan.included_turn_ids.clone(),
        omitted_turn_ids: plan.omitted_turn_ids.clone(),
        estimated_upper_tokens: plan.estimated_upper_tokens,
        exact_input_tokens: plan.exact_input_tokens,
        input_budget: plan.input_budget,
        decision: decision.to_owned(),
        active_context_start_before: start_before,
        active_context_start_after: start_after,
        context_items: plan.context_items.clone(),
        context_sha256: Some(plan.context_sha256.clone()),
        request: Some(request),
        identity_instruction: Some(plan.identity_instruction.clone()),
        untrusted_history_wrapped: plan.untrusted_history_wrapped,
        provenance_quality: ProvenanceQuality::Exact,
        retrieval: plan.retrieval_trace.clone(),
        knowledge: plan.knowledge_trace.clone(),
    };
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use rusqlite::Connection;
    use tokio::sync::Notify;

    use super::*;
    use crate::model::{BudgetConfig, ChatMessage};
    use crate::ollama::ModelInfo;

    #[derive(Clone, Default)]
    struct StructuredTestControl {
        block_call: Option<usize>,
        cancel_on_call: Option<usize>,
        call_started: Option<Arc<Notify>>,
        release_response: Option<Arc<Notify>>,
        cancellation: Option<CancellationToken>,
    }

    #[derive(Clone)]
    struct FakeClient {
        count: u64,
        history_cost: Option<(u64, u64)>,
        render_supported: bool,
        probes: Arc<Mutex<usize>>,
        events: Vec<ChatEvent>,
        stream_error: Option<OllamaError>,
        observe_source: Option<(PathBuf, Arc<Mutex<bool>>)>,
        captured_requests: Arc<Mutex<Vec<ChatRequest>>>,
        stream_calls: Arc<Mutex<usize>>,
        render_error: Option<OllamaError>,
        probe_error: Option<OllamaError>,
        embed_requests: Arc<Mutex<Vec<EmbeddingRequest>>>,
        embed_delay: Arc<Mutex<Option<Duration>>>,
        embed_error: Option<OllamaError>,
        structured_responses:
            Arc<Mutex<VecDeque<Result<crate::ollama::StructuredChatResponse, OllamaError>>>>,
        structured_requests: Arc<Mutex<Vec<StructuredChatRequest>>>,
        structured_delay: Arc<Mutex<Option<Duration>>>,
        structured_control: Arc<Mutex<StructuredTestControl>>,
    }

    impl FakeClient {
        fn new(count: u64) -> Self {
            Self {
                count,
                history_cost: None,
                render_supported: true,
                probes: Arc::new(Mutex::new(0)),
                events: vec![
                    ChatEvent::text(ChatEventKind::Thinking, "reason".into(), 1),
                    ChatEvent::text(ChatEventKind::Content, "answer".into(), 2),
                    ChatEvent {
                        kind: ChatEventKind::Completed,
                        text: String::new(),
                        live_output_tokens: Some(2),
                        usage: Some(TokenUsage::new(Some(count), Some(3))),
                        done_reason: Some("stop".into()),
                    },
                ],
                stream_error: None,
                observe_source: None,
                captured_requests: Arc::new(Mutex::new(Vec::new())),
                stream_calls: Arc::new(Mutex::new(0)),
                render_error: None,
                probe_error: None,
                embed_requests: Arc::new(Mutex::new(Vec::new())),
                embed_delay: Arc::new(Mutex::new(None)),
                embed_error: None,
                structured_responses: Arc::new(Mutex::new(VecDeque::new())),
                structured_requests: Arc::new(Mutex::new(Vec::new())),
                structured_delay: Arc::new(Mutex::new(None)),
                structured_control: Arc::new(Mutex::new(StructuredTestControl::default())),
            }
        }

        fn count_for(&self, messages: &[ChatMessage]) -> u64 {
            if let Some((base, per_history)) = self.history_cost {
                base + per_history
                    * messages
                        .iter()
                        .filter(|message| message.role == "assistant")
                        .count() as u64
            } else {
                self.count
            }
        }
    }

    #[async_trait]
    impl ChatBackend for FakeClient {
        async fn check_model(&self, model: &str, _: u64) -> Result<ModelInfo, OllamaError> {
            Ok(ModelInfo {
                version: "test".into(),
                name: model.into(),
                context_length: 65_536,
            })
        }

        async fn render_prompt(
            &self,
            _: &str,
            messages: &[ChatMessage],
            _: bool,
            _: u64,
        ) -> Result<Option<String>, OllamaError> {
            if let Some(error) = &self.render_error {
                return Err(error.clone());
            }
            Ok(self
                .render_supported
                .then(|| "x".repeat(self.count_for(messages) as usize)))
        }

        async fn probe(
            &self,
            _: &str,
            messages: &[ChatMessage],
            _: bool,
            _: u64,
        ) -> Result<TokenUsage, OllamaError> {
            *self.probes.lock().unwrap() += 1;
            if let Some(error) = &self.probe_error {
                return Err(error.clone());
            }
            Ok(TokenUsage::new(Some(self.count_for(messages)), Some(1)))
        }

        async fn embed(
            &self,
            request: EmbeddingRequest,
        ) -> Result<crate::ollama::EmbeddingResponse, OllamaError> {
            self.embed_requests.lock().unwrap().push(request.clone());
            let delay = *self.embed_delay.lock().unwrap();
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            if let Some(error) = &self.embed_error {
                return Err(error.clone());
            }
            let dimensions = request.dimensions.unwrap_or(4);
            Ok(crate::ollama::EmbeddingResponse {
                model: request.model,
                embeddings: request
                    .input
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        let mut vector = vec![0.0; dimensions];
                        vector[index % dimensions] = 1.0;
                        vector
                    })
                    .collect(),
                prompt_eval_count: None,
                total_duration: None,
                load_duration: None,
            })
        }

        async fn stream_chat(
            &self,
            request: ChatRequest,
            _: CancellationToken,
            emit: &mut (dyn FnMut(ChatEvent) + Send),
        ) -> Result<(), OllamaError> {
            *self.stream_calls.lock().unwrap() += 1;
            self.captured_requests.lock().unwrap().push(request);
            if let Some((path, observed)) = &self.observe_source {
                let raw = std::fs::read(path).unwrap();
                let persisted: Session = serde_json::from_slice(&raw).unwrap();
                *observed.lock().unwrap() = persisted
                    .turns
                    .last()
                    .is_some_and(|turn| turn.request_started_at.is_some());
            }
            for event in &self.events {
                emit(event.clone());
            }
            self.stream_error.clone().map_or(Ok(()), Err)
        }

        async fn structured_chat(
            &self,
            request: StructuredChatRequest,
        ) -> Result<crate::ollama::StructuredChatResponse, OllamaError> {
            self.structured_requests.lock().unwrap().push(request);
            let control = self.structured_control.lock().unwrap().clone();
            let call_number = self.structured_requests.lock().unwrap().len();
            let delay = *self.structured_delay.lock().unwrap();
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            if control.block_call == Some(call_number) {
                if let Some(notify) = control.call_started {
                    notify.notify_one();
                }
                if let Some(notify) = control.release_response {
                    notify.notified().await;
                }
            }
            if control.cancel_on_call == Some(call_number)
                && let Some(cancellation) = control.cancellation
            {
                cancellation.cancel();
            }
            self.structured_responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    Err(OllamaError::Other(
                        "missing fake structured response".into(),
                    ))
                })
        }
    }

    fn budget() -> BudgetConfig {
        BudgetConfig {
            context_window: 1_000,
            max_output_tokens: 100,
            safety_margin_tokens: 0,
            probe_ratio: 0.8,
            warning_ratio: 0.9,
            trim_target_ratio: 0.8,
        }
    }

    fn roomy_budget() -> BudgetConfig {
        BudgetConfig::default()
    }

    fn enabled_memory_config(search_timeout_ms: u64) -> AppConfig {
        let mut config = AppConfig::default();
        config.memory.enabled = true;
        config.memory.search_timeout_ms = search_timeout_ms;
        config
    }

    fn input_reaching_estimate(session: &Session, target: u64) -> String {
        let mut input = String::new();
        while ContextAssembler
            .assemble(session, &input, Some(&[]), None)
            .estimated_upper_tokens
            .unwrap_or_default()
            < target
        {
            input.push('x');
        }
        input
    }

    #[tokio::test]
    async fn consolidation_disabled_and_up_to_date_skip_model() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let client = FakeClient::new(100);
        let disabled = ChatEngine::new(store.clone(), client.clone());
        let report = disabled
            .consolidate_session(
                &session,
                ConsolidationTrigger::Manual,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Disabled);
        assert_eq!(
            store
                .retrieval()
                .consolidation_attempts(&session.id)
                .unwrap(),
            vec![]
        );

        let mut config = AppConfig::default();
        config.memory.enabled = true;
        let enabled = ChatEngine::with_config(store.clone(), client, config);
        let report = enabled
            .consolidate_session(
                &session,
                ConsolidationTrigger::Manual,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::UpToDate);
        assert_eq!(
            store
                .retrieval()
                .consolidation_attempts(&session.id)
                .unwrap(),
            vec![]
        );
    }

    #[tokio::test]
    async fn consolidation_uses_persisted_session_contract_and_applies_batches() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("persisted-model", "http://localhost", None, budget(), true)
            .unwrap();
        let client = FakeClient::new(100);
        for index in 0..17 {
            let mut turn = Turn::pending(format!("turn {index}"));
            turn.status = TurnStatus::Complete;
            turn.assistant_content = format!("answer {index}");
            turn.usage = TokenUsage::new(Some(100), Some(3));
            turn.done_reason = Some("stop".into());
            session.turns.push(turn);
        }
        store.save(&mut session).unwrap();
        *client.structured_responses.lock().unwrap() = VecDeque::from([
            Ok(crate::ollama::StructuredChatResponse {
                content: "{\"entities\":[],\"claims\":[],\"boundaries\":[]}".into(),
                usage: TokenUsage::new(Some(11), Some(7)),
                done_reason: Some("stop".into()),
            }),
            Ok(crate::ollama::StructuredChatResponse {
                content: "{\"entities\":[],\"claims\":[],\"boundaries\":[]}".into(),
                usage: TokenUsage::new(Some(11), Some(7)),
                done_reason: Some("stop".into()),
            }),
        ]);
        let before = serde_json::to_vec(&session).unwrap();
        let mut stale = session.clone();
        stale.model = "caller-stale-model".into();
        stale.budget.context_window = 9_999;
        stale.budget.max_output_tokens = 777;
        let mut config = AppConfig::default();
        config.memory.enabled = true;
        let engine = ChatEngine::with_config(store.clone(), client.clone(), config);
        let report = engine
            .consolidate_session(
                &stale,
                ConsolidationTrigger::Manual,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Completed);
        assert_eq!(report.batches_applied, 2);
        assert_eq!(report.watermark_after, 33);
        assert_eq!(serde_json::to_vec(&session).unwrap(), before);
        let requests = client.structured_requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        for request in requests.iter() {
            assert_eq!(request.model, "persisted-model");
            assert_eq!(request.num_ctx, budget().context_window);
            assert_eq!(request.num_predict, budget().max_output_tokens);
            assert_eq!(request.messages.len(), 2);
            let payload: Value = serde_json::from_str(&request.messages[1].content).unwrap();
            assert!(payload.get("batch").is_some());
            assert!(payload.get("candidate_snapshot").is_some());
        }
        let attempts = store
            .retrieval()
            .consolidation_attempts(&session.id)
            .unwrap();
        assert_eq!(attempts.len(), 2);
        for (attempt, captured) in attempts.into_iter().zip(requests.iter()) {
            let request: StructuredChatRequest =
                serde_json::from_str(&attempt.request_json).unwrap();
            assert_eq!(request, *captured);
            assert_eq!(
                attempt.request_sha256,
                content_sha256(&attempt.request_json)
            );
        }
    }

    #[tokio::test]
    async fn consolidation_failures_are_audited_and_retryable() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let mut turn = Turn::pending("fact".into());
        turn.status = TurnStatus::Complete;
        turn.assistant_content = "answer".into();
        turn.usage = TokenUsage::new(Some(3), Some(2));
        session.turns.push(turn);
        store.save(&mut session).unwrap();
        let client = FakeClient::new(100);
        let mut config = AppConfig::default();
        config.memory.enabled = true;
        let engine = ChatEngine::with_config(store.clone(), client.clone(), config);
        *client.structured_responses.lock().unwrap() = VecDeque::from([
            Ok(crate::ollama::StructuredChatResponse {
                content: "malformed raw output".into(),
                usage: TokenUsage::new(Some(8), Some(4)),
                done_reason: None,
            }),
            Ok(crate::ollama::StructuredChatResponse {
                content: "{\"entities\":[],\"claims\":[],\"boundaries\":[]}".into(),
                usage: TokenUsage::new(Some(8), Some(4)),
                done_reason: None,
            }),
        ]);
        let progress = Arc::new(Mutex::new(Vec::new()));
        let observed_progress = progress.clone();
        let report = engine
            .consolidate_session_with_progress(
                &session,
                ConsolidationTrigger::Manual,
                CancellationToken::new(),
                move |event| observed_progress.lock().unwrap().push(event),
            )
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Completed);
        assert!(report.watermark_after > 0);
        let attempts = store
            .retrieval()
            .consolidation_attempts(&session.id)
            .unwrap();
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].status, ConsolidationAttemptStatus::Rejected);
        assert_eq!(attempts[1].status, ConsolidationAttemptStatus::Applied);
        let wrapped: Value =
            serde_json::from_str(attempts[0].response_json.as_deref().unwrap()).unwrap();
        assert_eq!(
            wrapped["raw_output"],
            Value::String("malformed raw output".into())
        );
        assert_eq!(
            wrapped["raw_sha256"],
            Value::String(content_sha256("malformed raw output"))
        );
        assert_eq!(
            attempts[0].response_sha256.as_deref(),
            Some(content_sha256(attempts[0].response_json.as_deref().unwrap()).as_str())
        );
        assert!(
            serde_json::from_str::<Value>(attempts[0].validation_json.as_deref().unwrap()).is_ok()
        );
        assert!(matches!(
            progress.lock().unwrap().as_slice(),
            [
                ConsolidationProgress::AttemptStarted { attempt: 1, .. },
                ConsolidationProgress::ValidationRetry {
                    next_attempt: 2,
                    max_attempts: CONSOLIDATION_VALIDATION_ATTEMPTS,
                    ..
                },
                ConsolidationProgress::AttemptStarted { attempt: 2, .. },
                ConsolidationProgress::BatchApplied { .. },
            ]
        ));

        let mut second = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let mut turn = Turn::pending("fact".into());
        turn.status = TurnStatus::Complete;
        turn.assistant_content = "answer".into();
        second.turns.push(turn);
        store.save(&mut second).unwrap();
        client
            .structured_responses
            .lock()
            .unwrap()
            .push_back(Err(OllamaError::Other("backend down".into())));
        let report = engine
            .consolidate_session(
                &second,
                ConsolidationTrigger::Manual,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Failed);
        let failure = store
            .retrieval()
            .consolidation_attempts(&second.id)
            .unwrap();
        assert_eq!(failure[0].status, ConsolidationAttemptStatus::ModelError);
        assert!(failure[0].response_json.is_none());
        assert_eq!(report.watermark_after, 0);

        let mut exhausted = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let mut turn = Turn::pending("fact".into());
        turn.status = TurnStatus::Complete;
        turn.assistant_content = "answer".into();
        exhausted.turns.push(turn);
        store.save(&mut exhausted).unwrap();
        *client.structured_responses.lock().unwrap() = (0..CONSOLIDATION_VALIDATION_ATTEMPTS)
            .map(|attempt| {
                Ok(crate::ollama::StructuredChatResponse {
                    content: format!("malformed {attempt}"),
                    usage: TokenUsage::new(Some(8), Some(4)),
                    done_reason: None,
                })
            })
            .collect();
        let report = engine
            .consolidate_session(
                &exhausted,
                ConsolidationTrigger::Manual,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Failed);
        assert_eq!(report.batches_attempted, 1);
        assert_eq!(report.watermark_after, 0);
        let failures = store
            .retrieval()
            .consolidation_attempts(&exhausted.id)
            .unwrap();
        assert_eq!(failures.len(), CONSOLIDATION_VALIDATION_ATTEMPTS);
        assert!(
            failures
                .iter()
                .all(|attempt| attempt.status == ConsolidationAttemptStatus::Rejected)
        );
    }

    #[tokio::test]
    async fn consolidation_cancellation_timeout_and_partial_progress() {
        fn valid_response() -> crate::ollama::StructuredChatResponse {
            crate::ollama::StructuredChatResponse {
                content: "{\"entities\":[],\"claims\":[],\"boundaries\":[]}".into(),
                usage: TokenUsage::new(Some(8), Some(4)),
                done_reason: Some("stop".into()),
            }
        }
        fn seeded(store: &SessionStore, turns: usize) -> Session {
            let mut session = store
                .create("model", "http://localhost", None, budget(), true)
                .unwrap();
            for index in 0..turns {
                let mut turn = Turn::pending(format!("fact {index}"));
                turn.status = TurnStatus::Complete;
                turn.assistant_content = format!("answer {index}");
                turn.usage = TokenUsage::new(Some(3), Some(2));
                turn.done_reason = Some("stop".into());
                session.turns.push(turn);
            }
            store.save(&mut session).unwrap();
            session
        }
        fn engine(store: SessionStore, client: FakeClient) -> ChatEngine<FakeClient> {
            let mut config = AppConfig::default();
            config.memory.enabled = true;
            config.memory.consolidation_timeout_secs = 1;
            ChatEngine::with_config(store, client, config)
        }

        // A: pre-cancel has no backend call or audit.
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let session = seeded(&store, 1);
        let client = FakeClient::new(100);
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let report = engine(store.clone(), client.clone())
            .consolidate_session(&session, ConsolidationTrigger::Manual, cancelled)
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Cancelled);
        assert!(client.structured_requests.lock().unwrap().is_empty());
        assert!(
            store
                .retrieval()
                .consolidation_attempts(&session.id)
                .unwrap()
                .is_empty()
        );

        // B: cancellation during an already-started call produces one response-less audit.
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let session = seeded(&store, 1);
        let client = FakeClient::new(100);
        let call_started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        *client.structured_control.lock().unwrap() = StructuredTestControl {
            block_call: Some(1),
            call_started: Some(call_started.clone()),
            release_response: Some(release),
            ..StructuredTestControl::default()
        };
        let cancellation = CancellationToken::new();
        let task_engine = engine(store.clone(), client.clone());
        let task_session = session.clone();
        let task_cancellation = cancellation.clone();
        let task = tokio::spawn(async move {
            task_engine
                .consolidate_session(
                    &task_session,
                    ConsolidationTrigger::Manual,
                    task_cancellation,
                )
                .await
        });
        call_started.notified().await;
        cancellation.cancel();
        let report = task.await.unwrap();
        assert_eq!(report.status, ConsolidationRunStatus::Cancelled);
        assert_eq!(report.watermark_after, 0);
        let attempts = store
            .retrieval()
            .consolidation_attempts(&session.id)
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].status, ConsolidationAttemptStatus::Cancelled);
        assert!(attempts[0].response_json.is_none());
        assert!(attempts[0].input_tokens.is_none() && attempts[0].output_tokens.is_none());

        // C: timeout is a model error with no response and leaves the batch retryable.
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let session = seeded(&store, 1);
        let client = FakeClient::new(100);
        *client.structured_delay.lock().unwrap() = Some(Duration::from_secs(2));
        let report = engine(store.clone(), client.clone())
            .consolidate_session(
                &session,
                ConsolidationTrigger::Manual,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Failed);
        let timeout = store
            .retrieval()
            .consolidation_attempts(&session.id)
            .unwrap();
        assert_eq!(timeout.len(), 1);
        assert_eq!(timeout[0].status, ConsolidationAttemptStatus::ModelError);
        assert!(timeout[0].response_json.is_none());
        assert_eq!(report.watermark_after, 0);

        // D: the first of two bounded batches persists, and the rejected second batch stays retryable.
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let session = seeded(&store, 17);
        let first = store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        let client = FakeClient::new(100);
        let mut responses = VecDeque::from([Ok(valid_response())]);
        responses.extend((0..CONSOLIDATION_VALIDATION_ATTEMPTS).map(|attempt| {
            Ok(crate::ollama::StructuredChatResponse {
                content: format!("malformed {attempt}"),
                usage: TokenUsage::new(Some(9), Some(5)),
                done_reason: None,
            })
        }));
        *client.structured_responses.lock().unwrap() = responses;
        let report = engine(store.clone(), client.clone())
            .consolidate_session(
                &session,
                ConsolidationTrigger::Manual,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Partial);
        assert_eq!(report.batches_attempted, 2);
        assert_eq!(report.batches_applied, 1);
        assert_eq!(report.events_applied, first.events.len());
        assert!(report.events_attempted > report.events_applied);
        assert_eq!(report.watermark_after, first.through_sequence);
        let attempts = store
            .retrieval()
            .consolidation_attempts(&session.id)
            .unwrap();
        assert_eq!(
            attempts
                .iter()
                .map(|attempt| attempt.status)
                .collect::<Vec<_>>(),
            vec![
                ConsolidationAttemptStatus::Applied,
                ConsolidationAttemptStatus::Rejected,
                ConsolidationAttemptStatus::Rejected,
                ConsolidationAttemptStatus::Rejected
            ]
        );
        assert_eq!(attempts[0].through_sequence, first.through_sequence);
        assert!(
            attempts[1..]
                .iter()
                .all(|attempt| attempt.through_sequence > first.through_sequence)
        );

        // E: cancellation after the second call starts keeps exactly the first batch.
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let session = seeded(&store, 17);
        let first = store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        let client = FakeClient::new(100);
        *client.structured_responses.lock().unwrap() = VecDeque::from([Ok(valid_response())]);
        let call_started = Arc::new(Notify::new());
        *client.structured_control.lock().unwrap() = StructuredTestControl {
            block_call: Some(2),
            call_started: Some(call_started.clone()),
            release_response: Some(Arc::new(Notify::new())),
            ..StructuredTestControl::default()
        };
        let cancellation = CancellationToken::new();
        let task_engine = engine(store.clone(), client.clone());
        let task_session = session.clone();
        let task_cancellation = cancellation.clone();
        let task = tokio::spawn(async move {
            task_engine
                .consolidate_session(
                    &task_session,
                    ConsolidationTrigger::Manual,
                    task_cancellation,
                )
                .await
        });
        call_started.notified().await;
        cancellation.cancel();
        let report = task.await.unwrap();
        assert_eq!(report.status, ConsolidationRunStatus::Cancelled);
        assert_eq!(report.batches_attempted, 2);
        assert_eq!(report.batches_applied, 1);
        assert_eq!(report.watermark_after, first.through_sequence);
        let attempts = store
            .retrieval()
            .consolidation_attempts(&session.id)
            .unwrap();
        assert_eq!(
            attempts
                .iter()
                .map(|attempt| attempt.status)
                .collect::<Vec<_>>(),
            vec![
                ConsolidationAttemptStatus::Applied,
                ConsolidationAttemptStatus::Cancelled
            ]
        );
        assert!(attempts[1].response_json.is_none());
        assert!(attempts[1].input_tokens.is_none() && attempts[1].output_tokens.is_none());

        // F: cancellation observed after a response returns retains that exact response and usage.
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let session = seeded(&store, 1);
        let client = FakeClient::new(100);
        let cancellation = CancellationToken::new();
        let raw = "{\"entities\":[],\"claims\":[],\"boundaries\":[]}";
        client.structured_responses.lock().unwrap().push_back(Ok(
            crate::ollama::StructuredChatResponse {
                content: raw.into(),
                usage: TokenUsage::new(Some(12), None),
                done_reason: None,
            },
        ));
        *client.structured_control.lock().unwrap() = StructuredTestControl {
            cancel_on_call: Some(1),
            cancellation: Some(cancellation.clone()),
            ..StructuredTestControl::default()
        };
        let report = engine(store.clone(), client)
            .consolidate_session(&session, ConsolidationTrigger::Manual, cancellation)
            .await;
        assert_eq!(report.status, ConsolidationRunStatus::Cancelled);
        assert_eq!(report.watermark_after, 0);
        let attempts = store
            .retrieval()
            .consolidation_attempts(&session.id)
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].status, ConsolidationAttemptStatus::Cancelled);
        assert_eq!(attempts[0].response_json.as_deref(), Some(raw));
        assert_eq!(
            attempts[0].response_sha256.as_deref(),
            Some(content_sha256(raw).as_str())
        );
        assert_eq!(attempts[0].input_tokens, Some(12));
        assert_eq!(attempts[0].output_tokens, None);
    }

    #[tokio::test]
    async fn below_threshold_streams_and_persists_exact_usage() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let client = FakeClient::new(100);
        let engine = ChatEngine::new(store.clone(), client.clone());
        let prepared = engine
            .prepare_turn(&mut session, "hello".into())
            .await
            .unwrap();
        assert!(prepared.ready());
        assert_eq!(*client.probes.lock().unwrap(), 0);
        assert_eq!(session.turns[0].probe_usage, TokenUsage::zero());
        engine
            .stream_turn(&mut session, &prepared, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        assert_eq!(session.turns[0].status, TurnStatus::Complete);
        assert_eq!(session.turns[0].thinking, "reason");
        assert_eq!(session.turns[0].usage.total_tokens, Some(103));
        let answer_id = crate::model::event_id(
            &session.id,
            Some(&session.turns[0].id),
            crate::model::EventRole::Assistant,
        );
        let trace = engine
            .store()
            .retrieval()
            .answer_context(&answer_id)
            .unwrap();
        assert_eq!(trace.provenance_quality, ProvenanceQuality::Exact);
        assert_eq!(
            trace
                .items
                .iter()
                .map(|item| item.resolved.content.as_str())
                .collect::<Vec<_>>(),
            vec![session.system_prompt.as_str(), "hello"]
        );
        let next = ContextAssembler.assemble(&session, "next", None, None);
        assert!(!format!("{:?}", next.messages).contains("reason"));
    }

    #[tokio::test]
    async fn request_start_is_durable_before_backend_stream_begins() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let observed = Arc::new(Mutex::new(false));
        let mut client = FakeClient::new(100);
        client.observe_source = Some((
            root.path().join(format!("{}.json", session.id)),
            observed.clone(),
        ));
        let engine = ChatEngine::new(store, client);
        let prepared = engine
            .prepare_turn(&mut session, "hello".into())
            .await
            .unwrap();
        engine
            .stream_turn(&mut session, &prepared, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        assert!(*observed.lock().unwrap());
    }

    #[tokio::test]
    async fn warning_can_end_without_streaming() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let input = input_reaching_estimate(&session, session.budget.probe_threshold());
        let engine = ChatEngine::new(store, FakeClient::new(850));
        let prepared = engine.prepare_turn(&mut session, input).await.unwrap();
        assert!(prepared.needs_limit_decision());
        let ended = engine
            .resolve_limit(&mut session, prepared, LimitAction::EndSession)
            .await
            .unwrap();
        assert_eq!(ended.status, PreparationStatus::Ended);
        assert_eq!(session.status, SessionStatus::Paused);
        assert_eq!(session.turns[0].status, TurnStatus::Blocked);
        let answer_id = crate::model::event_id(
            &session.id,
            Some(&session.turns[0].id),
            crate::model::EventRole::Assistant,
        );
        assert!(matches!(
            engine.store().retrieval().get_event(&answer_id),
            Err(crate::retrieval::RetrievalError::EventNotFound(_))
        ));
    }

    #[tokio::test]
    async fn ordinary_context_skips_render_and_exact_probe() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let mut client = FakeClient::new(100);
        client.render_supported = false;
        let probes = client.probes.clone();
        let prepared = ChatEngine::new(store, client)
            .prepare_turn(&mut session, "hello".into())
            .await
            .unwrap();
        assert_eq!(prepared.plan.exact_input_tokens, None);
        assert_eq!(*probes.lock().unwrap(), 0);
        assert_eq!(session.turns[0].probe_usage, TokenUsage::zero());
    }

    #[tokio::test]
    async fn live_search_deadline_uses_fast_fallback_and_marks_timeout_channels() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, roomy_budget(), false)
            .unwrap();
        let client = FakeClient::new(100);
        *client.embed_delay.lock().unwrap() = Some(Duration::from_millis(1_500));
        let engine = ChatEngine::with_config(store, client, enabled_memory_config(1_000));
        let prepared = engine
            .prepare_turn(&mut session, "deadline query".into())
            .await
            .unwrap();
        let trace = &prepared.plan.retrieval_trace;
        assert!((900..1_300).contains(&trace.elapsed_ms));
        assert_eq!(trace.deadline_ms, 1_000);
        assert!(trace.deadline_exceeded);
        assert!(trace.fast_fallback_used);
        assert!(
            trace
                .fallback_reason
                .as_deref()
                .is_some_and(|reason| reason.starts_with("deadline_exceeded:"))
        );
        assert!(trace.warnings.iter().any(|warning| {
            warning.contains("搜索超过 1000ms，已采用快速搜索结果；未完成通道：")
        }));
        assert!(trace.channels.iter().any(|channel| {
            channel.channel == RetrievalChannel::Bm25 && channel.status == "ok"
        }));
        assert!(trace.channels.iter().any(|channel| {
            channel.channel == RetrievalChannel::Vector && channel.status == "timeout"
        }));
        assert!(prepared.query_embedding.is_none());
    }

    #[tokio::test]
    async fn live_search_starts_bm25_embedding_and_knowledge_concurrently() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, roomy_budget(), false)
            .unwrap();
        let engine =
            ChatEngine::with_config(store, FakeClient::new(100), enabled_memory_config(1_000));
        *engine
            .live_search_barrier
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(Arc::new(tokio::sync::Barrier::new(3)));
        let prepared = tokio::time::timeout(
            Duration::from_millis(1_500),
            engine.prepare_turn(&mut session, "parallel query".into()),
        )
        .await
        .expect("all three search workers must reach the shared barrier")
        .unwrap();
        assert!(!prepared.plan.retrieval_trace.deadline_exceeded);
    }

    #[tokio::test]
    async fn semantic_channels_prepare_before_delayed_bm25_finishes() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, roomy_budget(), false)
            .unwrap();
        let engine =
            ChatEngine::with_config(store, FakeClient::new(100), enabled_memory_config(1_000));
        let fast_gate = Arc::new(Notify::new());
        let semantic_prepared = Arc::new(Notify::new());
        *engine
            .live_search_fast_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&fast_gate));
        *engine
            .live_search_semantic_prepared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(Arc::clone(&semantic_prepared));

        let task_engine = engine.clone();
        let preparation = tokio::spawn(async move {
            task_engine
                .prepare_turn(&mut session, "delayed bm25 query".into())
                .await
        });
        tokio::time::timeout(Duration::from_millis(1_500), semantic_prepared.notified())
            .await
            .expect("semantic preparation must finish while BM25 is still blocked");
        fast_gate.notify_one();
        let prepared = tokio::time::timeout(Duration::from_millis(1_500), preparation)
            .await
            .expect("live search should finish within its shared deadline")
            .unwrap()
            .unwrap();
        assert!(!prepared.plan.retrieval_trace.deadline_exceeded);
        assert!(
            prepared
                .plan
                .retrieval_trace
                .channels
                .iter()
                .any(|channel| {
                    channel.channel == RetrievalChannel::Vector
                        && matches!(channel.status.as_str(), "ok" | "stale")
                }),
            "semantic channels: {:#?}; warnings: {:#?}",
            prepared.plan.retrieval_trace.channels,
            prepared.plan.retrieval_trace.warnings
        );
    }

    #[tokio::test]
    async fn knowledge_failure_does_not_discard_fast_memory_results() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        std::fs::create_dir_all(store.knowledge().index_path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, roomy_budget(), false)
            .unwrap();
        let prepared = ChatEngine::new(store, FakeClient::new(100))
            .prepare_turn(&mut session, "knowledge failure".into())
            .await
            .unwrap();
        assert!(prepared.ready());
        assert!(
            prepared
                .plan
                .retrieval_trace
                .channels
                .iter()
                .any(|channel| {
                    channel.channel == RetrievalChannel::Bm25 && channel.status == "ok"
                })
        );
        assert_eq!(prepared.plan.knowledge_trace.status, "failed");
        assert!(prepared.plan.knowledge_trace.error.is_some());
    }

    #[tokio::test]
    async fn short_query_embedding_is_reused_by_exit_refresh() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, roomy_budget(), false)
            .unwrap();
        let client = FakeClient::new(100);
        let requests = client.embed_requests.clone();
        let config = enabled_memory_config(5_000);
        let engine = ChatEngine::with_config(store.clone(), client, config.clone());
        let user = "short cache query".to_owned();
        let prepared = engine
            .prepare_turn(&mut session, user.clone())
            .await
            .unwrap();
        assert!(prepared.query_embedding.is_some());
        assert_eq!(
            prepared
                .plan
                .retrieval_trace
                .channels
                .iter()
                .map(|channel| (
                    channel.channel,
                    channel.status.as_str(),
                    channel.error.is_some(),
                ))
                .collect::<Vec<_>>(),
            vec![
                (RetrievalChannel::Bm25, "ok", false),
                (RetrievalChannel::Vector, "stale", false),
                (RetrievalChannel::Entity, "stale", false),
                (RetrievalChannel::State, "stale", false),
                (RetrievalChannel::Episode, "stale", false),
                (RetrievalChannel::Graph, "error", true),
            ],
            "warnings: {:#?}",
            prepared.plan.retrieval_trace.warnings
        );
        let before_answer = Connection::open(store.retrieval().index_path()).unwrap();
        let cached_before: i64 = before_answer
            .query_row("SELECT count(*) FROM memory_embedding_cache", [], |row| {
                row.get(0)
            })
            .unwrap();
        let published_before: i64 = before_answer
            .query_row("SELECT count(*) FROM memory_embeddings", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!((cached_before, published_before), (0, 0));
        drop(before_answer);
        engine
            .stream_turn(&mut session, &prepared, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        engine.wait_for_background_embeddings().await;
        let spec = VectorIndexSpec::from_config(&config.memory).unwrap();
        let cached = store.retrieval().cached_content_embeddings(&spec).unwrap();
        assert!(cached.contains_key(&content_sha256(&user)));
        let user_calls_before_refresh = requests
            .lock()
            .unwrap()
            .iter()
            .flat_map(|request| request.input.iter())
            .filter(|input| *input == &user)
            .count();
        assert_eq!(user_calls_before_refresh, 1);
        let report = engine
            .refresh_embeddings(CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(report.leaf_embedded_inputs, 0);
        let user_calls_after_refresh = requests
            .lock()
            .unwrap()
            .iter()
            .flat_map(|request| request.input.iter())
            .filter(|input| *input == &user)
            .count();
        assert_eq!(user_calls_after_refresh, 1);

        let mut second_session = store
            .create("model", "http://localhost", None, roomy_budget(), false)
            .unwrap();
        let second = engine
            .prepare_turn(&mut second_session, user.clone())
            .await
            .unwrap();
        let vector_channel = second
            .plan
            .retrieval_trace
            .channels
            .iter()
            .find(|channel| channel.channel == RetrievalChannel::Vector)
            .expect("live search records vector status");
        assert_eq!(
            vector_channel.status, "stale",
            "warnings: {:#?}",
            second.plan.retrieval_trace.warnings
        );
        assert!(!second.plan.retrieval_trace.fusion_candidates.is_empty());
        assert_eq!(
            requests
                .lock()
                .unwrap()
                .iter()
                .flat_map(|request| request.input.iter())
                .filter(|input| *input == &user)
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn long_query_uses_cached_overlapping_fragments_not_whole_query_vector() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, roomy_budget(), false)
            .unwrap();
        let client = FakeClient::new(100);
        let requests = client.embed_requests.clone();
        let config = enabled_memory_config(5_000);
        let engine = ChatEngine::with_config(store.clone(), client, config.clone());
        let user = "长消息".repeat(100);
        assert!(user.chars().count() > 240);
        let prepared = engine
            .prepare_turn(&mut session, user.clone())
            .await
            .unwrap();
        engine
            .stream_turn(&mut session, &prepared, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        engine.wait_for_background_embeddings().await;
        let spec = VectorIndexSpec::from_config(&config.memory).unwrap();
        let cached = store.retrieval().cached_content_embeddings(&spec).unwrap();
        let query_vector = cached
            .get(&content_sha256(&user))
            .expect("completed query vector is cached")
            .clone();
        let fragments = turn_leaf_embedding_inputs(&user, "answer")
            .into_iter()
            .filter(|input| input != "answer")
            .collect::<Vec<_>>();
        assert!(fragments.len() > 1);
        assert!(
            fragments
                .iter()
                .all(|fragment| cached.contains_key(&content_sha256(fragment)))
        );
        assert_eq!(
            requests
                .lock()
                .unwrap()
                .iter()
                .flat_map(|request| request.input.iter())
                .filter(|input| *input == &user)
                .count(),
            1
        );
        let report = engine
            .refresh_embeddings(CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(report.leaf_embedded_inputs, 0);
        let message_vector = store
            .retrieval()
            .compatible_embeddings(&spec)
            .unwrap()
            .into_iter()
            .find(|embedding| {
                embedding.granularity == RetrievalDocumentGranularity::Message
                    && embedding.source_sha256 == content_sha256(&user)
            })
            .expect("long message embedding was published from fragments")
            .vector;
        assert_ne!(message_vector, query_vector);
    }

    #[tokio::test]
    async fn organized_graph_prefix_survives_append_and_is_reported_stale() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, roomy_budget(), false)
            .unwrap();
        let config = enabled_memory_config(5_000);
        let engine = ChatEngine::with_config(store.clone(), FakeClient::new(100), config.clone());
        let first = engine
            .prepare_turn(&mut session, "first organized fact".into())
            .await
            .unwrap();
        engine
            .stream_turn(&mut session, &first, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        engine.wait_for_background_embeddings().await;
        engine
            .refresh_embeddings(CancellationToken::new())
            .await
            .unwrap();
        store.retrieval().refresh_graph(&config.memory).unwrap();
        store.retrieval().mark_graph_organized().unwrap();

        let second = engine
            .prepare_turn(&mut session, "second pending fact".into())
            .await
            .unwrap();
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        let materializations: i64 = connection
            .query_row(
                "SELECT count(*) FROM memory_graph_materializations",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(materializations, 1);
        let graph_channel = second
            .plan
            .retrieval_trace
            .channels
            .iter()
            .find(|channel| channel.channel == RetrievalChannel::Graph)
            .expect("live search records graph status");
        assert_eq!(
            (
                graph_channel.status.as_str(),
                graph_channel.error.as_deref()
            ),
            ("stale", None)
        );
    }

    #[tokio::test]
    async fn two_session_fragment_evidence_survives_source_resync() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut a = store
            .create(
                "model",
                "http://localhost",
                Some("a-system"),
                roomy_budget(),
                false,
            )
            .unwrap();
        let client_a = FakeClient::new(100);
        let engine_a = ChatEngine::new(store.clone(), client_a);
        let long = format!(
            "{} 海棠计划暗号是青瓷月亮。 {}",
            "填充".repeat(110),
            "尾部".repeat(30)
        );
        let prepared_a = engine_a.prepare_turn(&mut a, long.clone()).await.unwrap();
        engine_a
            .stream_turn(&mut a, &prepared_a, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        let mut b = store
            .create(
                "model",
                "http://localhost",
                Some("b-system"),
                roomy_budget(),
                false,
            )
            .unwrap();
        let client_b = FakeClient::new(100);
        let requests = client_b.captured_requests.clone();
        let calls = client_b.stream_calls.clone();
        let engine_b = ChatEngine::new(store.clone(), client_b);
        let prepared_b = engine_b
            .prepare_turn(&mut b, "海棠计划暗号是什么".into())
            .await
            .unwrap();
        let core = prepared_b
            .plan
            .evidence
            .iter()
            .find(|item| item.kind == crate::model::EvidenceKind::Core)
            .unwrap();
        assert_eq!(
            core.span.event_id,
            crate::model::event_id(&a.id, Some(&a.turns[0].id), crate::model::EventRole::User)
        );
        let selected = prepared_b
            .plan
            .retrieval_trace
            .candidates
            .iter()
            .find(|candidate| candidate.selected && candidate.span == core.span)
            .unwrap();
        assert_eq!(
            selected.granularity,
            crate::model::RetrievalDocumentGranularity::Fragment
        );
        let resolved = store.retrieval().resolve_span(&core.span).unwrap();
        assert!(
            prepared_b
                .plan
                .messages
                .iter()
                .any(|message| message.content == resolved.content)
        );
        assert_eq!(prepared_b.plan.messages[0].content, b.system_prompt);
        engine_b
            .stream_turn(&mut b, &prepared_b, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        assert_eq!(*calls.lock().unwrap(), 1);
        assert_eq!(
            requests.lock().unwrap()[0].messages,
            prepared_b.plan.messages
        );
        let answer_id = crate::model::event_id(
            &b.id,
            Some(&b.turns[0].id),
            crate::model::EventRole::Assistant,
        );
        let before = store.retrieval().answer_context(&answer_id).unwrap();
        let mut messages = before
            .items
            .iter()
            .map(|item| ChatMessage {
                role: item.role.as_str().into(),
                content: item.resolved.content.clone(),
            })
            .collect::<Vec<_>>();
        if let Some(identity) = &before.identity_instruction {
            let position = messages
                .iter()
                .position(|message| message.role == "system")
                .map_or(0, |index| index + 1);
            messages.insert(
                position,
                ChatMessage {
                    role: "system".into(),
                    content: identity.clone(),
                },
            );
        }
        assert_eq!(messages, requests.lock().unwrap()[0].messages);
        assert_eq!(before.context_sha256, prepared_b.plan.context_sha256);
        assert_eq!(
            before.retrieval_trace.selected_evidence,
            prepared_b.plan.retrieval_trace.selected_evidence
        );
        assert_eq!(
            before
                .retrieval_trace
                .candidates
                .iter()
                .map(|c| (c.raw_rank, &c.document_id, &c.reason, c.selected))
                .collect::<Vec<_>>(),
            prepared_b
                .plan
                .retrieval_trace
                .candidates
                .iter()
                .map(|c| (c.raw_rank, &c.document_id, &c.reason, c.selected))
                .collect::<Vec<_>>()
        );
        let prepared_a2 = engine_a
            .prepare_turn(&mut a, "A的合法新增事件".into())
            .await
            .unwrap();
        engine_a
            .stream_turn(&mut a, &prepared_a2, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        let after = store.retrieval().answer_context(&answer_id).unwrap();
        assert_eq!(after.context_sha256, before.context_sha256);
        assert_eq!(
            after.retrieval_trace.selected_evidence,
            before.retrieval_trace.selected_evidence
        );
        let path = root.path().join(format!("{}.json", a.id));
        let raw = std::fs::read(&path).unwrap();
        std::fs::write(&path, [raw, b" ".to_vec()].concat()).unwrap();
        assert!(
            matches!(store.retrieval().answer_context(&answer_id), Err(crate::retrieval::RetrievalError::StaleIndex { session_id }) if session_id == a.id)
        );
    }

    #[tokio::test]
    async fn bm25_corruption_falls_back_to_recent_context_without_streaming() {
        for (column, value, table) in [
            ("exact_content", "tampered", "retrieval_documents"),
            ("content_sha256", "bad-hash", "retrieval_documents"),
            ("content_sha256", "bad-hash", "source_spans"),
        ] {
            let root = tempfile::tempdir().unwrap();
            let store = SessionStore::new(root.path()).unwrap();
            let mut a = store
                .create("model", "http://localhost", Some("a"), budget(), false)
                .unwrap();
            let engine_a = ChatEngine::new(store.clone(), FakeClient::new(100));
            let prepared_a = engine_a
                .prepare_turn(&mut a, "唯一暗号是青瓷月亮".into())
                .await
                .unwrap();
            engine_a
                .stream_turn(&mut a, &prepared_a, CancellationToken::new(), |_| {})
                .await
                .unwrap();
            let source_bytes = std::fs::read(root.path().join(format!("{}.json", a.id))).unwrap();
            let event =
                crate::model::event_id(&a.id, Some(&a.turns[0].id), crate::model::EventRole::User);
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            if table == "retrieval_documents" {
                connection
                    .execute(
                        &format!("UPDATE retrieval_documents SET {column}=?1 WHERE event_id=?2"),
                        rusqlite::params![value, event],
                    )
                    .unwrap();
            } else {
                connection
                    .execute(
                        &format!(
                            "UPDATE source_spans SET {column}=?1 WHERE event_id=?2 AND start_char=0"
                        ),
                        rusqlite::params![value, event],
                    )
                    .unwrap();
            }
            drop(connection);
            let mut b = store
                .create(
                    "model",
                    "http://localhost",
                    Some("b"),
                    roomy_budget(),
                    false,
                )
                .unwrap();
            b.turns.push(completed_turn(0));
            store.save(&mut b).unwrap();
            let recent_turn_id = b.turns[0].id.clone();
            let client = FakeClient::new(100);
            let calls = client.stream_calls.clone();
            let requests = client.captured_requests.clone();
            let engine_b = ChatEngine::new(store.clone(), client);
            let prepared = engine_b
                .prepare_turn(&mut b, "青瓷月亮暗号".into())
                .await
                .unwrap();
            assert!(prepared.ready());
            assert!(prepared.plan.included_turn_ids.contains(&recent_turn_id));
            assert_eq!(
                prepared.plan.retrieval_trace.status,
                "fast_search_unavailable"
            );
            assert!(
                prepared
                    .plan
                    .retrieval_trace
                    .channels
                    .iter()
                    .any(|channel| channel.channel == RetrievalChannel::Bm25
                        && channel.status == "error")
            );
            assert_eq!(*calls.lock().unwrap(), 0);
            assert!(requests.lock().unwrap().is_empty());
            assert!(
                b.turns
                    .last()
                    .is_some_and(|turn| turn.status == TurnStatus::Pending
                        && turn.request_started_at.is_none()
                        && turn.context_trace.retrieval.status == "fast_search_unavailable")
            );
            assert_eq!(
                std::fs::read(root.path().join(format!("{}.json", a.id))).unwrap(),
                source_bytes
            );
        }
    }

    #[tokio::test]
    async fn ordinary_recall_does_not_depend_on_render_or_probe() {
        for probe_case in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let store = SessionStore::new(root.path()).unwrap();
            let mut a = store
                .create(
                    "model",
                    "http://localhost",
                    Some("a"),
                    roomy_budget(),
                    false,
                )
                .unwrap();
            let ea = ChatEngine::new(store.clone(), FakeClient::new(100));
            let pa = ea
                .prepare_turn(&mut a, "外部事实：琥珀钥匙在杭州".into())
                .await
                .unwrap();
            ea.stream_turn(&mut a, &pa, CancellationToken::new(), |_| {})
                .await
                .unwrap();
            let mut b = store
                .create(
                    "model",
                    "http://localhost",
                    Some("b"),
                    roomy_budget(),
                    false,
                )
                .unwrap();
            let mut client = FakeClient::new(if probe_case { 800 } else { 100 });
            client.render_supported = probe_case;
            if probe_case {
                client.probe_error = Some(OllamaError::Protocol("probe failure".into()));
            } else {
                client.render_error = Some(OllamaError::Protocol("render failure".into()));
            }
            let calls = client.stream_calls.clone();
            let probes = client.probes.clone();
            let engine = ChatEngine::new(store.clone(), client);
            let prepared = engine
                .prepare_turn(&mut b, "琥珀钥匙在哪里".into())
                .await
                .unwrap();
            assert!(prepared.ready());
            assert_eq!(*probes.lock().unwrap(), 0);
            let reloaded = store.load(&b.id).unwrap();
            let turn = reloaded.turns.last().unwrap();
            assert_eq!(turn.context_trace.retrieval.status, "ok");
            assert!(!turn.context_trace.retrieval.candidates.is_empty());
            assert!(!turn.context_trace.retrieval.selected_evidence.is_empty());
            assert_eq!(turn.context_trace.decision, "ready");
            assert_eq!(turn.status, TurnStatus::Pending);
            assert!(turn.request_started_at.is_none());
            assert_eq!(*calls.lock().unwrap(), 0);
        }
    }

    #[tokio::test]
    async fn render_fallback_preserves_external_retrieval_trace() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut a = store
            .create("model", "http://localhost", Some("a"), budget(), false)
            .unwrap();
        let ea = ChatEngine::new(store.clone(), FakeClient::new(100));
        let pa = ea
            .prepare_turn(&mut a, "外部事实：翡翠罗盘".into())
            .await
            .unwrap();
        ea.stream_turn(&mut a, &pa, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        let mut b = store
            .create(
                "model",
                "http://localhost",
                Some("b"),
                BudgetConfig {
                    context_window: 4_000,
                    max_output_tokens: 500,
                    safety_margin_tokens: 0,
                    probe_ratio: 0.5,
                    warning_ratio: 0.9,
                    trim_target_ratio: 0.5,
                },
                false,
            )
            .unwrap();
        let mut client = FakeClient::new(100);
        client.render_supported = false;
        let probes = client.probes.clone();
        let engine = ChatEngine::new(store.clone(), client);
        let mut input = input_reaching_estimate(&b, b.budget.probe_threshold());
        input.push_str(" 翡翠罗盘是什么");
        let prepared = engine.prepare_turn(&mut b, input).await.unwrap();
        assert_eq!(*probes.lock().unwrap(), 1);
        assert!(!prepared.plan.retrieval_trace.selected_evidence.is_empty());
        assert_eq!(
            store
                .load(&b.id)
                .unwrap()
                .turns
                .last()
                .unwrap()
                .context_trace
                .retrieval
                .selected_evidence,
            prepared.plan.retrieval_trace.selected_evidence
        );
    }

    #[tokio::test]
    async fn thresholds_are_inclusive() {
        for (count, expected) in [
            (720, PreparationStatus::Ready),
            (810, PreparationStatus::LimitWarning),
        ] {
            let root = tempfile::tempdir().unwrap();
            let store = SessionStore::new(root.path()).unwrap();
            let mut session = store
                .create("model", "http://localhost", None, budget(), true)
                .unwrap();
            let client = FakeClient::new(count);
            let probes = client.probes.clone();
            let input = input_reaching_estimate(&session, session.budget.probe_threshold());
            let progress = Arc::new(Mutex::new(Vec::new()));
            let observed_progress = progress.clone();
            let prepared = ChatEngine::new(store, client)
                .prepare_turn_with_progress(&mut session, input, move |event| {
                    observed_progress.lock().unwrap().push(event);
                })
                .await
                .unwrap();
            assert_eq!(prepared.status, expected);
            assert_eq!(*probes.lock().unwrap(), 1);
            assert!(matches!(
                progress.lock().unwrap().as_slice(),
                [PreparationProgress::ExactContextCheckStarted { .. }]
            ));
        }
    }

    fn completed_turn(index: usize) -> Turn {
        let mut turn = Turn::pending(format!("user-{index}"));
        turn.status = TurnStatus::Complete;
        turn.assistant_content = format!("assistant-{index}");
        turn.usage = TokenUsage::new(Some(10), Some(5));
        turn
    }

    #[tokio::test]
    async fn continue_keeps_maximum_recent_suffix_at_trim_target() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut trim_budget = budget();
        trim_budget.context_window = 1_200;
        let mut session = store
            .create("model", "http://localhost", None, trim_budget, true)
            .unwrap();
        session.turns = (0..3).map(completed_turn).collect();
        store.save(&mut session).unwrap();
        let mut client = FakeClient::new(100);
        client.history_cost = Some((300, 250));
        let probes = client.probes.clone();
        let engine = ChatEngine::new(store, client);
        let prepared = engine
            .prepare_turn(&mut session, "current".into())
            .await
            .unwrap();
        let resumed = engine
            .resolve_limit(&mut session, prepared, LimitAction::ContinueWithTrim)
            .await
            .unwrap();
        assert_eq!(resumed.plan.exact_input_tokens, None);
        assert_eq!(*probes.lock().unwrap(), 1);
        assert!(plan_metric(&resumed.plan).unwrap() < session.budget.trim_target());
        assert_eq!(
            resumed.plan.included_turn_ids,
            vec![session.turns[2].id.clone()]
        );
        assert_eq!(session.active_context_start_index, 2);
        assert_eq!(
            session.turns.last().unwrap().context_trace.decision,
            "trimmed_and_continued"
        );
    }

    #[tokio::test]
    async fn limit_resolution_reuses_prepared_retrieval_without_rank_drift() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut a = store
            .create("model", "http://localhost", Some("a"), budget(), false)
            .unwrap();
        let ea = ChatEngine::new(store.clone(), FakeClient::new(100));
        let pa = ea
            .prepare_turn(&mut a, "外部唯一事实：朱砂钥匙".into())
            .await
            .unwrap();
        ea.stream_turn(&mut a, &pa, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        let mut b = store
            .create("model", "http://localhost", Some("b"), budget(), false)
            .unwrap();
        b.turns = (0..3).map(completed_turn).collect();
        store.save(&mut b).unwrap();
        let mut client = FakeClient::new(100);
        client.history_cost = Some((300, 250));
        let calls = client.stream_calls.clone();
        let eb = ChatEngine::new(store.clone(), client);
        let prepared = eb
            .prepare_turn(&mut b, "朱砂钥匙是什么".into())
            .await
            .unwrap();
        assert!(prepared.needs_limit_decision());
        let trace = prepared.plan.retrieval_trace.clone();
        let evidence = prepared.plan.evidence.clone();
        let mut c = store
            .create("model", "http://localhost", Some("c"), budget(), false)
            .unwrap();
        let ec = ChatEngine::new(store.clone(), FakeClient::new(100));
        let pc = ec
            .prepare_turn(&mut c, "朱砂钥匙朱砂钥匙朱砂钥匙".into())
            .await
            .unwrap();
        ec.stream_turn(&mut c, &pc, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        let resumed = eb
            .resolve_limit(&mut b, prepared, LimitAction::ContinueWithTrim)
            .await
            .unwrap();
        assert!(resumed.ready());
        assert_eq!(resumed.plan.retrieval_trace, trace);
        assert_eq!(resumed.plan.evidence, evidence);
        assert!(
            !resumed
                .plan
                .retrieval_trace
                .candidates
                .iter()
                .any(|candidate| candidate.session_id == c.id)
        );
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn end_session_preserves_prepared_retrieval_trace() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut a = store
            .create("model", "http://localhost", Some("a"), budget(), false)
            .unwrap();
        let ea = ChatEngine::new(store.clone(), FakeClient::new(100));
        let pa = ea
            .prepare_turn(&mut a, "外部事实：银杏信物".into())
            .await
            .unwrap();
        ea.stream_turn(&mut a, &pa, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        let mut b = store
            .create("model", "http://localhost", Some("b"), budget(), false)
            .unwrap();
        let client = FakeClient::new(850);
        let calls = client.stream_calls.clone();
        let eb = ChatEngine::new(store.clone(), client);
        let prepared = eb
            .prepare_turn(&mut b, "银杏信物是什么".into())
            .await
            .unwrap();
        assert!(prepared.needs_limit_decision());
        let trace = prepared.plan.retrieval_trace.clone();
        let evidence = prepared.plan.evidence.clone();
        let ended = eb
            .resolve_limit(&mut b, prepared, LimitAction::EndSession)
            .await
            .unwrap();
        assert_eq!(ended.status, PreparationStatus::Ended);
        let turn = store.load(&b.id).unwrap().turns.pop().unwrap();
        assert_eq!(turn.context_trace.retrieval, trace);
        assert_eq!(turn.context_trace.retrieval.selected_evidence, evidence);
        assert_eq!(turn.context_trace.decision, "paused_by_user");
        assert_eq!(turn.status, TurnStatus::Blocked);
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn mandatory_block_preserves_prepared_retrieval_trace() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut a = store
            .create("model", "http://localhost", Some("a"), budget(), false)
            .unwrap();
        let ea = ChatEngine::new(store.clone(), FakeClient::new(100));
        let pa = ea
            .prepare_turn(&mut a, "外部事实：松烟墨盒".into())
            .await
            .unwrap();
        ea.stream_turn(&mut a, &pa, CancellationToken::new(), |_| {})
            .await
            .unwrap();
        let mut b = store
            .create("model", "http://localhost", Some("b"), budget(), false)
            .unwrap();
        let client = FakeClient::new(850);
        let calls = client.stream_calls.clone();
        let eb = ChatEngine::new(store.clone(), client);
        let input = input_reaching_estimate(&b, b.budget.trim_target().saturating_add(1));
        let prepared = eb.prepare_turn(&mut b, input).await.unwrap();
        assert!(prepared.needs_limit_decision());
        let trace = prepared.plan.retrieval_trace.clone();
        let evidence = prepared.plan.evidence.clone();
        let blocked = eb
            .resolve_limit(&mut b, prepared, LimitAction::ContinueWithTrim)
            .await
            .unwrap();
        assert_eq!(blocked.status, PreparationStatus::Blocked);
        let turn = store.load(&b.id).unwrap().turns.pop().unwrap();
        assert_eq!(turn.context_trace.retrieval, trace);
        assert_eq!(turn.context_trace.retrieval.selected_evidence, evidence);
        assert_eq!(turn.context_trace.decision, "mandatory_above_trim_target");
        assert_eq!(turn.status, TurnStatus::Blocked);
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn interrupted_stream_never_promotes_probe_usage() {
        let root = tempfile::tempdir().unwrap();
        let store = SessionStore::new(root.path()).unwrap();
        let mut session = store
            .create("model", "http://localhost", None, budget(), true)
            .unwrap();
        let mut client = FakeClient::new(750);
        client.events = vec![ChatEvent::text(ChatEventKind::Content, "partial".into(), 2)];
        client.stream_error = Some(OllamaError::Stream {
            message: "lost".into(),
            live_output_tokens: 2,
        });
        let engine = ChatEngine::new(store, client);
        let prepared = engine
            .prepare_turn(&mut session, "question".into())
            .await
            .unwrap();
        assert!(
            engine
                .stream_turn(&mut session, &prepared, CancellationToken::new(), |_| {})
                .await
                .is_err()
        );
        let turn = session.turns.last().unwrap();
        assert_eq!(turn.status, TurnStatus::Interrupted);
        assert_eq!(turn.usage, TokenUsage::new(None, Some(2)));
        assert_eq!(turn.probe_usage, TokenUsage::zero());
        assert!(!turn.context_eligible());
    }
}
