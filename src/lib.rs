pub mod context;
pub mod engine;
pub mod model;
pub mod ollama;
pub mod retrieval;
pub mod store;
pub mod tui;
pub mod web;

pub use context::ContextAssembler;
pub use engine::{ChatEngine, LimitAction, PreparedTurn};
pub use model::{
    BudgetConfig, EventRole, EvidenceKind, ProvenanceQuality, RetrievalConfig, RetrievalTrace,
    Session, SourceSpan, TokenUsage, Turn,
};
pub use ollama::{ModelInfo, OllamaClient};
pub use retrieval::{
    AnswerContext, AnswerContextItem, IndexedSession, RecallResult, RecalledEvidence, ResolvedSpan,
    RetrievalError, RetrievalStore, StoredEvent, SyncReport,
};
pub use store::{IndexSyncAfterSourceCommit, SessionStore};
