pub mod config;
pub mod consolidation;
pub mod context;
pub mod engine;
pub mod episode;
pub mod knowledge;
pub mod model;
pub mod ollama;
pub mod retrieval;
pub mod store;
pub mod tui;
pub mod vector;
pub mod web;

pub use config::{
    AppConfig, KnowledgeConfig, KnowledgeSourceConfig, KnowledgeSourceKind, WebSearchConfig,
};
pub use consolidation::{
    BoundarySuggestionReason, CONSOLIDATION_MAX_CHARS, CONSOLIDATION_MAX_TURNS, ClaimCardinality,
    ClaimCertainty, ClaimDisposition, ClaimPolarity, ConsolidatedClaimObject,
    ConsolidatedClaimOutput, ConsolidatedEntityOutput, ConsolidationApplyError,
    ConsolidationApplyReport, ConsolidationApplyResult, ConsolidationAttemptRecord,
    ConsolidationAttemptStatus, ConsolidationBoundaryOutput, ConsolidationCandidateSnapshot,
    ConsolidationClaimEvidence, ConsolidationClaimObjectKind, ConsolidationEvent,
    ConsolidationEvidenceKind, ConsolidationInputBatch, ConsolidationQuote, ConsolidationRunReport,
    ConsolidationRunStatus, ConsolidationTrigger, ConsolidationWatermark, EntityAliasOutput,
    EntityDisambiguation, EntityResolution, EntityResolutionBasis, MemoryAliasCandidate,
    MemoryAliasKind, MemoryClaimCandidate, MemoryClaimEvidenceCandidate, MemoryClaimState,
    MemoryEntityCandidate, MemoryEntityKind, StructuredConsolidationOutput,
    structured_consolidation_schema,
};
pub use context::ContextAssembler;
pub use engine::{ChatEngine, LimitAction, PreparedTurn};
pub use episode::{
    EMBEDDING_COSINE_SIMILARITY_THRESHOLD, ENTITY_JACCARD_DISTANCE_THRESHOLD,
    EPISODE_ALGORITHM_VERSION, EpisodeBoundaryDecision, EpisodeDocument, EpisodeGapState,
    EpisodeMaterializationReport, EpisodeMember, EpisodeSignal, EpisodeSignalState,
    SOFT_BOUNDARY_VOTE_THRESHOLD,
};
pub use knowledge::{KnowledgeRecall, KnowledgeStore, KnowledgeSyncReport, KnowledgeTrace};
pub use model::{
    AgentChatRequest, AgentMessage, AgentRoundResult, BudgetConfig, EventRole, EvidenceKind,
    ProvenanceQuality, RetrievalConfig, RetrievalTrace, Session, SourceSpan, TokenUsage, ToolCall,
    ToolDefinition, ToolResultTrace, ToolRoundTrace, Turn, WebSourceTrace, WebTrace,
};
pub use ollama::{
    ModelInfo, OllamaClient, WebFetchResponse, WebSearchResponse, WebSearchResult,
    validate_public_http_url, validate_public_http_url_resolved,
};
pub use retrieval::{
    AggregateEmbeddingDocument, AggregateEmbeddingSnapshot, AnswerContext, AnswerContextItem,
    DirectMessageEmbedding, EmbeddingPublishReport, IndexedSession, LeafEmbeddingDocument,
    LeafEmbeddingSnapshot, RecallResult, RecalledEvidence, ResolvedSpan, RetrievalError,
    RetrievalStore, StoredEvent, SyncReport,
};
pub use store::{IndexSyncAfterSourceCommit, SessionStore};
pub use vector::{
    EMBEDDING_PREPROCESSING_VERSION, EmbeddingCoverage, EmbeddingWrite, StoredEmbedding,
    VectorError, VectorIndexSpec, decode_f32_le, encode_f32_le, equal_mean, l2_normalize,
    weighted_pool,
};
