pub mod config;
pub mod context;
pub mod engine;
pub mod knowledge;
pub mod model;
pub mod ollama;
pub mod retrieval;
pub mod store;
pub mod tui;
pub mod web;

pub use config::{
    AppConfig, KnowledgeConfig, KnowledgeSourceConfig, KnowledgeSourceKind, WebSearchConfig,
};
pub use context::ContextAssembler;
pub use engine::{ChatEngine, LimitAction, PreparedTurn};
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
    AnswerContext, AnswerContextItem, IndexedSession, RecallResult, RecalledEvidence, ResolvedSpan,
    RetrievalError, RetrievalStore, StoredEvent, SyncReport,
};
pub use store::{IndexSyncAfterSourceCommit, SessionStore};
