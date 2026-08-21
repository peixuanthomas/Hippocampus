use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::knowledge::KnowledgeTrace;

pub const SCHEMA_VERSION: u32 = 4;
pub const LEGACY_SCHEMA_VERSION: u32 = 1;
pub const PREVIOUS_SCHEMA_VERSION: u32 = 2;
pub const PREVIOUS_SCHEMA_VERSION_V3: u32 = 3;
pub const DEFAULT_SYSTEM_PROMPT: &str =
    "你是一个AI助手，你的任务是用简练且切中要害的解决用户的问题或者与用户对话。";

pub fn utc_now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::AutoSi, false)
}

pub fn default_ai_name() -> String {
    "LLM".to_owned()
}

pub fn identity_instruction(ai_name: &str) -> String {
    format!(
        "你的 AI 名称是 {:?}。当用户询问你的身份时，请使用这个名称。",
        ai_name
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BudgetConfig {
    pub context_window: u64,
    pub max_output_tokens: u64,
    pub safety_margin_tokens: u64,
    pub probe_ratio: f64,
    pub warning_ratio: f64,
    pub trim_target_ratio: f64,
}

/// Independent limits for lexical long-term evidence.  Kept in the source
/// session so a completed answer remains explainable after restart.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrievalConfig {
    pub candidate_limit: usize,
    pub max_selected: usize,
    pub evidence_char_budget: usize,
    pub expansion_char_budget: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            candidate_limit: 64,
            max_selected: 4,
            evidence_char_budget: 1600,
            expansion_char_budget: 800,
        }
    }
}

impl RetrievalConfig {
    pub fn validate(&self) -> Result<()> {
        let deferred_hard_limits = self.max_selected == usize::MAX
            && self.evidence_char_budget == usize::MAX
            && self.expansion_char_budget == usize::MAX;
        if self.candidate_limit == 0
            || self.candidate_limit > 512
            || !deferred_hard_limits
                && (self.max_selected == 0
                    || self.max_selected > self.candidate_limit
                    || self.evidence_char_budget == 0
                    || self.evidence_char_budget > 32_768
                    || self.expansion_char_budget > 16_384)
        {
            bail!("检索配置必须为合理的非零有界值");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalDocumentGranularity {
    Message,
    Fragment,
    Episode,
    Session,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QueryKind {
    ExactFact,
    #[default]
    GeneralSemantic,
    EventRecap,
    TemporalState,
    MultiHop,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalChannel {
    #[default]
    Bm25,
    Vector,
    Entity,
    State,
    Episode,
    Graph,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChannelTrace {
    #[serde(default)]
    pub channel: RetrievalChannel,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub candidate_count: usize,
    #[serde(default)]
    pub elapsed_ms: u64,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BudgetBucket {
    #[default]
    RecentHistory,
    ExactOrState,
    Episode,
    Graph,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetTokenBreakdown {
    #[serde(default)]
    pub recent_history: u64,
    #[serde(default)]
    pub exact_or_state: u64,
    #[serde(default)]
    pub episode: u64,
    #[serde(default)]
    pub graph: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetReflowTrace {
    #[serde(default)]
    pub bucket: BudgetBucket,
    #[serde(default)]
    pub offered_tokens: u64,
    #[serde(default)]
    pub consumed_tokens: u64,
    #[serde(default)]
    pub remaining_tokens: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetExclusionTrace {
    #[serde(default)]
    pub bucket: BudgetBucket,
    #[serde(default)]
    pub candidate_group_id: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub exact_increment_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetStageLatencyTrace {
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetProbeTrace {
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub request_sha256: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub usage: TokenUsage,
    #[serde(default)]
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetAllocationTrace {
    #[serde(default)]
    pub query_kind: QueryKind,
    #[serde(default = "default_recent_history_percent")]
    pub recent_history_percent: u8,
    #[serde(default = "default_exact_or_state_percent")]
    pub exact_or_state_percent: u8,
    #[serde(default = "default_episode_percent")]
    pub episode_percent: u8,
    #[serde(default = "default_graph_percent")]
    pub graph_percent: u8,
    #[serde(default)]
    pub mandatory_input_tokens: u64,
    #[serde(default)]
    pub available_input_tokens: u64,
    #[serde(default)]
    pub initial_tokens: BudgetTokenBreakdown,
    #[serde(default)]
    pub actual_tokens: BudgetTokenBreakdown,
    #[serde(default)]
    pub final_input_tokens: Option<u64>,
    #[serde(default)]
    pub reflow: Vec<BudgetReflowTrace>,
    #[serde(default)]
    pub exclusions: Vec<BudgetExclusionTrace>,
    #[serde(default)]
    pub stage_latencies: Vec<BudgetStageLatencyTrace>,
    #[serde(default)]
    pub probes: Vec<BudgetProbeTrace>,
}

const fn default_recent_history_percent() -> u8 {
    45
}

const fn default_exact_or_state_percent() -> u8 {
    30
}

const fn default_episode_percent() -> u8 {
    15
}

const fn default_graph_percent() -> u8 {
    10
}

impl Default for BudgetAllocationTrace {
    fn default() -> Self {
        Self::for_query_kind(QueryKind::GeneralSemantic)
    }
}

impl BudgetAllocationTrace {
    pub const fn for_query_kind(kind: QueryKind) -> Self {
        let (recent_history_percent, exact_or_state_percent, episode_percent, graph_percent) =
            match kind {
                QueryKind::ExactFact => (45, 40, 5, 10),
                QueryKind::GeneralSemantic => (45, 30, 15, 10),
                QueryKind::EventRecap => (35, 20, 35, 10),
                QueryKind::TemporalState => (35, 35, 15, 15),
                QueryKind::MultiHop => (30, 25, 15, 30),
            };
        Self {
            query_kind: kind,
            recent_history_percent,
            exact_or_state_percent,
            episode_percent,
            graph_percent,
            mandatory_input_tokens: 0,
            available_input_tokens: 0,
            initial_tokens: BudgetTokenBreakdown {
                recent_history: 0,
                exact_or_state: 0,
                episode: 0,
                graph: 0,
            },
            actual_tokens: BudgetTokenBreakdown {
                recent_history: 0,
                exact_or_state: 0,
                episode: 0,
                graph: 0,
            },
            final_input_tokens: None,
            reflow: Vec::new(),
            exclusions: Vec::new(),
            stage_latencies: Vec::new(),
            probes: Vec::new(),
        }
    }

    pub fn normalize_usage(&mut self) {
        for probe in &mut self.probes {
            probe.usage.refresh();
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphPathTrace {
    #[serde(default)]
    pub seed_document_id: String,
    #[serde(default)]
    pub target_document_id: String,
    #[serde(default)]
    pub edge_types: Vec<String>,
    #[serde(default)]
    pub node_ids: Vec<String>,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub path_quality: f64,
    #[serde(default)]
    pub seed_channel: RetrievalChannel,
    #[serde(default)]
    pub seed_node_id: String,
    #[serde(default)]
    pub seed_source_id: String,
    #[serde(default)]
    pub seed_rank: usize,
    #[serde(default)]
    pub seed_score: f64,
    #[serde(default)]
    pub seed_mass: f64,
    #[serde(default)]
    pub edge_ids: Vec<String>,
    #[serde(default)]
    pub target_rank: usize,
    #[serde(default)]
    pub target_granularity: Option<RetrievalDocumentGranularity>,
    #[serde(default)]
    pub target_session_id: String,
    #[serde(default)]
    pub span: Option<SourceSpan>,
    #[serde(default)]
    pub content_sha256: String,
    #[serde(default)]
    pub role: Option<EventRole>,
    #[serde(default)]
    pub selected: bool,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Core,
    Context,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RankedCandidate {
    pub raw_rank: usize,
    pub document_id: String,
    pub granularity: RetrievalDocumentGranularity,
    pub span: SourceSpan,
    pub role: EventRole,
    pub session_id: String,
    pub created_at: String,
    pub content_sha256: String,
    pub bm25_score: f64,
    pub selected: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FusionCandidateTrace {
    #[serde(default)]
    pub fused_rank: usize,
    #[serde(default)]
    pub document_id: String,
    #[serde(default = "default_source_span")]
    pub span: SourceSpan,
    #[serde(default)]
    pub session_id: String,
    #[serde(default = "default_retrieval_granularity")]
    pub granularity: RetrievalDocumentGranularity,
    #[serde(default)]
    pub source_document_ids: Vec<String>,
    #[serde(default)]
    pub episode_id: Option<String>,
    #[serde(default)]
    pub bm25_rank: Option<usize>,
    #[serde(default)]
    pub bm25_score: Option<f64>,
    #[serde(default)]
    pub vector_rank: Option<usize>,
    #[serde(default)]
    pub vector_score: Option<f64>,
    #[serde(default)]
    pub rrf_score: f64,
    #[serde(default)]
    pub protected_exact: bool,
    #[serde(default)]
    pub selected: bool,
    #[serde(default)]
    pub reason: String,
}

impl Default for FusionCandidateTrace {
    fn default() -> Self {
        Self {
            fused_rank: 0,
            document_id: String::new(),
            span: SourceSpan {
                event_id: String::new(),
                start_char: 0,
                end_char: 0,
            },
            session_id: String::new(),
            granularity: RetrievalDocumentGranularity::Message,
            source_document_ids: Vec::new(),
            episode_id: None,
            bm25_rank: None,
            bm25_score: None,
            vector_rank: None,
            vector_score: None,
            rrf_score: 0.0,
            protected_exact: false,
            selected: false,
            reason: String::new(),
        }
    }
}

fn default_source_span() -> SourceSpan {
    SourceSpan {
        event_id: String::new(),
        start_char: 0,
        end_char: 0,
    }
}

fn default_retrieval_granularity() -> RetrievalDocumentGranularity {
    RetrievalDocumentGranularity::Message
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectedEvidence {
    pub span: SourceSpan,
    pub content_sha256: String,
    pub role: EventRole,
    pub kind: EvidenceKind,
    pub originating_candidate_rank: Option<usize>,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EntityMatchTrace {
    #[serde(default)]
    pub matched_text: String,
    #[serde(default)]
    pub normalized_text: String,
    #[serde(default)]
    pub match_basis: String,
    #[serde(default)]
    pub candidate_entity_ids: Vec<String>,
    #[serde(default)]
    pub selected_entity_id: Option<String>,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StateSelectionTrace {
    #[serde(default)]
    pub rank: usize,
    #[serde(default)]
    pub claim_id: String,
    #[serde(default)]
    pub subject_entity_id: String,
    #[serde(default)]
    pub object_entity_id: Option<String>,
    #[serde(default)]
    pub predicate_key: String,
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub certainty: String,
    #[serde(default)]
    pub asserted_at: String,
    #[serde(default)]
    pub event_time: Option<String>,
    #[serde(default)]
    pub valid_from: String,
    #[serde(default)]
    pub valid_to: Option<String>,
    #[serde(default)]
    pub reference_time: String,
    #[serde(default)]
    pub related_claim_ids: Vec<String>,
    #[serde(default)]
    pub evidence_id: Option<String>,
    #[serde(default)]
    pub evidence_span: Option<SourceSpan>,
    #[serde(default)]
    pub evidence_role: Option<EventRole>,
    #[serde(default)]
    pub selected: bool,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalTrace {
    pub status: String,
    pub current_query_event_id: String,
    #[serde(default)]
    pub query_terms: Vec<String>,
    #[serde(default)]
    pub config: RetrievalConfig,
    #[serde(default)]
    pub candidates: Vec<RankedCandidate>,
    #[serde(default)]
    pub fusion_candidates: Vec<FusionCandidateTrace>,
    #[serde(default)]
    pub entity_matches: Vec<EntityMatchTrace>,
    #[serde(default)]
    pub state_selections: Vec<StateSelectionTrace>,
    #[serde(default)]
    pub selected_evidence: Vec<SelectedEvidence>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub query_kind: QueryKind,
    #[serde(default)]
    pub channels: Vec<ChannelTrace>,
    #[serde(default)]
    pub graph_paths: Vec<GraphPathTrace>,
    #[serde(default)]
    pub budget_allocation: BudgetAllocationTrace,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub elapsed_ms: u64,
    #[serde(default)]
    pub deadline_ms: u64,
    #[serde(default)]
    pub deadline_exceeded: bool,
    #[serde(default)]
    pub fast_fallback_used: bool,
}

impl Default for RetrievalTrace {
    fn default() -> Self {
        Self {
            status: "not_run".into(),
            current_query_event_id: String::new(),
            query_terms: Vec::new(),
            config: RetrievalConfig::default(),
            candidates: Vec::new(),
            fusion_candidates: Vec::new(),
            entity_matches: Vec::new(),
            state_selections: Vec::new(),
            selected_evidence: Vec::new(),
            error: None,
            query_kind: QueryKind::default(),
            channels: Vec::new(),
            graph_paths: Vec::new(),
            budget_allocation: BudgetAllocationTrace::default(),
            warnings: Vec::new(),
            elapsed_ms: 0,
            deadline_ms: 0,
            deadline_exceeded: false,
            fast_fallback_used: false,
        }
    }
}

impl RetrievalTrace {
    pub fn normalize_usage(&mut self) {
        self.budget_allocation.normalize_usage();
    }
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            context_window: 32_768,
            max_output_tokens: 4_096,
            safety_margin_tokens: 512,
            probe_ratio: 0.80,
            warning_ratio: 0.90,
            trim_target_ratio: 0.80,
        }
    }
}

impl BudgetConfig {
    pub fn validate(&self) -> Result<()> {
        if self.context_window <= self.max_output_tokens + self.safety_margin_tokens {
            bail!("context_window 必须大于输出预留与安全余量之和");
        }
        if !(0.0 < self.trim_target_ratio
            && self.trim_target_ratio <= self.probe_ratio
            && self.probe_ratio <= self.warning_ratio
            && self.warning_ratio < 1.0)
        {
            bail!("比例必须满足 0 < trim_target <= probe <= warning < 1");
        }
        Ok(())
    }

    pub fn input_budget(&self) -> u64 {
        self.context_window - self.max_output_tokens - self.safety_margin_tokens
    }

    pub fn probe_threshold(&self) -> u64 {
        (self.input_budget() as f64 * self.probe_ratio) as u64
    }

    pub fn warning_threshold(&self) -> u64 {
        (self.input_budget() as f64 * self.warning_ratio) as u64
    }

    pub fn trim_target(&self) -> u64 {
        (self.input_budget() as f64 * self.trim_target_ratio) as u64
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    #[serde(default, skip_deserializing)]
    pub total_tokens: Option<u64>,
}

impl TokenUsage {
    pub const fn new(input_tokens: Option<u64>, output_tokens: Option<u64>) -> Self {
        let total_tokens = match (input_tokens, output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        };
        Self {
            input_tokens,
            output_tokens,
            total_tokens,
        }
    }

    pub const fn zero() -> Self {
        Self::new(Some(0), Some(0))
    }

    pub fn refresh(&mut self) {
        self.total_tokens = match (self.input_tokens, self.output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        };
    }

    pub fn add(&mut self, other: Self) {
        if let Some(value) = other.input_tokens {
            self.input_tokens = Some(self.input_tokens.unwrap_or(0) + value);
        }
        if let Some(value) = other.output_tokens {
            self.output_tokens = Some(self.output_tokens.unwrap_or(0) + value);
        }
        self.refresh();
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Pending,
    Complete,
    Truncated,
    Blocked,
    Interrupted,
    Failed,
    NoAnswer,
}

impl TurnStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Complete => "complete",
            Self::Truncated => "truncated",
            Self::Blocked => "blocked",
            Self::Interrupted => "interrupted",
            Self::Failed => "failed",
            Self::NoAnswer => "no_answer",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceQuality {
    Exact,
    LegacyInferred,
}

fn legacy_inferred() -> ProvenanceQuality {
    ProvenanceQuality::LegacyInferred
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventRole {
    System,
    User,
    Assistant,
}

impl EventRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    pub event_id: String,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextItemTrace {
    pub role: EventRole,
    pub span: SourceSpan,
    pub content_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelRequestTrace {
    pub model: String,
    pub think: bool,
    pub context_window: u64,
    pub max_output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallFunction {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    #[serde(rename = "type", default = "function_tool_kind")]
    pub kind: String,
    pub function: ToolCallFunction,
}

fn function_tool_kind() -> String {
    "function".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thinking: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl From<&ChatMessage> for AgentMessage {
    fn from(value: &ChatMessage) -> Self {
        Self {
            role: value.role.clone(),
            content: value.content.clone(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            tool_name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentChatRequest {
    pub model: String,
    pub messages: Vec<AgentMessage>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    pub think: bool,
    pub num_ctx: u64,
    pub num_predict: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRoundResult {
    pub thinking: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
    pub done_reason: Option<String>,
    pub live_output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResultTrace {
    pub call_ordinal: usize,
    pub name: String,
    pub arguments: serde_json::Value,
    pub started_at: String,
    pub completed_at: String,
    pub status: String,
    pub full_response: String,
    pub full_response_sha256: String,
    pub injected_response: String,
    #[serde(default)]
    pub urls: Vec<String>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolRoundTrace {
    pub round: usize,
    pub started_at: String,
    pub completed_at: String,
    pub request_context_sha256: String,
    pub request_messages: Vec<AgentMessage>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    pub estimated_input_tokens: u64,
    pub exact_input_tokens: Option<u64>,
    pub assistant_thinking: String,
    pub assistant_content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub tool_results: Vec<ToolResultTrace>,
    pub usage: Option<TokenUsage>,
    pub done_reason: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WebSourceTrace {
    pub kind: String,
    pub title: String,
    pub url: String,
    pub round: usize,
    pub tool_call_ordinal: usize,
    pub observed_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WebTrace {
    pub status: String,
    pub enabled: bool,
    pub max_tool_rounds: usize,
    pub max_tool_calls: usize,
    #[serde(default)]
    pub steps: Vec<ToolRoundTrace>,
    #[serde(default)]
    pub sources: Vec<WebSourceTrace>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub unverified_realtime: bool,
    #[serde(default)]
    pub final_request_context_sha256: Option<String>,
}

impl Default for WebTrace {
    fn default() -> Self {
        Self {
            status: "disabled".into(),
            enabled: false,
            max_tool_rounds: 0,
            max_tool_calls: 0,
            steps: Vec::new(),
            sources: Vec::new(),
            warnings: Vec::new(),
            unverified_realtime: false,
            final_request_context_sha256: None,
        }
    }
}

impl WebTrace {
    pub fn normalize(&mut self) {
        for step in &mut self.steps {
            if let Some(usage) = &mut step.usage {
                usage.refresh();
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            if !self.steps.is_empty() || !self.sources.is_empty() {
                bail!("禁用的联网 trace 不能包含工具步骤或来源");
            }
            return Ok(());
        }
        if self.max_tool_rounds == 0 || self.max_tool_calls == 0 {
            bail!("启用的联网 trace 缺少有界预算");
        }
        let enabled_rounds = self
            .steps
            .iter()
            .filter(|step| !step.tools.is_empty())
            .count();
        if enabled_rounds > self.max_tool_rounds {
            bail!("联网 trace 超过工具轮次上限");
        }
        if self
            .steps
            .iter()
            .enumerate()
            .any(|(index, step)| step.tools.is_empty() && index + 1 != self.steps.len())
        {
            bail!("禁用工具的降级轮次只能出现在 trace 末尾");
        }
        let mut call_ordinals = HashSet::new();
        let mut results_by_origin = HashMap::new();
        let mut result_count = 0usize;
        for (index, step) in self.steps.iter().enumerate() {
            if step.round == 0 || (index > 0 && step.round <= self.steps[index - 1].round) {
                bail!("联网工具步骤 round 必须严格递增");
            }
            if agent_context_sha256(&step.request_messages, &step.tools)
                != step.request_context_sha256
            {
                bail!("联网工具步骤 {} 的请求哈希不匹配", step.round);
            }
            if let Some(usage) = step.usage {
                if step.completed_at.is_empty() {
                    bail!("已完成的联网工具步骤缺少完成时间");
                }
                if step.error.is_none()
                    && step.exact_input_tokens.is_some()
                    && step.exact_input_tokens != usage.input_tokens
                {
                    bail!("联网工具步骤 {} 的 probe/formal token 不匹配", step.round);
                }
            }
            if step.tool_results.len() > step.tool_calls.len() {
                bail!("联网工具步骤 {} 的结果多于 tool call", step.round);
            }
            for (result_index, result) in step.tool_results.iter().enumerate() {
                result_count += 1;
                if result_count > self.max_tool_calls {
                    bail!("联网 trace 超过工具调用上限");
                }
                if result.call_ordinal != result_count {
                    bail!("联网工具调用序号必须从 1 连续递增");
                }
                if !call_ordinals.insert(result.call_ordinal) {
                    bail!("联网 trace 包含重复工具调用序号 {}", result.call_ordinal);
                }
                if content_sha256(&result.full_response) != result.full_response_sha256 {
                    bail!("联网工具结果 {} 的完整响应哈希不匹配", result.call_ordinal);
                }
                if result.full_response.len() > 1024 * 1024 {
                    bail!("联网工具结果 {} 超过 1 MiB", result.call_ordinal);
                }
                let call = &step.tool_calls[result_index];
                if call.function.name != result.name || call.function.arguments != result.arguments
                {
                    bail!("联网工具结果 {} 找不到对应 tool call", result.call_ordinal);
                }
                validate_tool_result_payload(result)?;
                results_by_origin.insert((step.round, result.call_ordinal), result);
            }
        }
        for source in &self.sources {
            let result = results_by_origin
                .get(&(source.round, source.tool_call_ordinal))
                .ok_or_else(|| {
                    anyhow::anyhow!("联网来源 {:?} 找不到对应轮次与工具调用", source.url)
                })?;
            validate_web_source(source, result)?;
        }
        if let Some(hash) = &self.final_request_context_sha256
            && self.steps.last().map(|step| &step.request_context_sha256) != Some(hash)
        {
            bail!("联网 trace 的最终请求哈希不是最后一个工具步骤");
        }
        Ok(())
    }
}

fn validate_tool_result_payload(result: &ToolResultTrace) -> Result<()> {
    if result.status == "error" {
        if result.error.is_none() || !result.urls.is_empty() {
            bail!("失败的联网工具结果缺少错误或包含来源 URL");
        }
        let value: serde_json::Value = serde_json::from_str(&result.full_response)
            .map_err(|error| anyhow::anyhow!("联网错误响应不是 JSON：{error}"))?;
        if value.get("error").and_then(serde_json::Value::as_str) != result.error.as_deref()
            || !result.injected_response.contains(&result.full_response)
        {
            bail!("失败的联网工具结果与错误响应不一致");
        }
        return Ok(());
    }
    if result.status != "ok" || result.error.is_some() {
        bail!("联网工具结果状态无效");
    }
    let full: serde_json::Value = serde_json::from_str(&result.full_response)
        .map_err(|error| anyhow::anyhow!("联网完整响应不是 JSON：{error}"))?;
    let injected: serde_json::Value = serde_json::from_str(&result.injected_response)
        .map_err(|error| anyhow::anyhow!("联网注入响应不是 JSON：{error}"))?;
    match result.name.as_str() {
        "web_search" => {
            let full_results = full
                .get("results")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| anyhow::anyhow!("web_search 完整响应缺少 results"))?;
            let injected_results = injected
                .get("results")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| anyhow::anyhow!("web_search 注入响应缺少 results"))?;
            if !injected_results
                .iter()
                .all(|item| full_results.contains(item))
            {
                bail!("web_search 注入结果不是完整响应的子集");
            }
            let urls = injected_results
                .iter()
                .filter_map(|item| item.get("url").and_then(serde_json::Value::as_str))
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            if urls != result.urls {
                bail!("web_search 记录的 URL 与注入响应不一致");
            }
        }
        "web_fetch" => {
            let requested_url = result
                .arguments
                .get("url")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("web_fetch 参数缺少 url"))?;
            if injected.get("url").and_then(serde_json::Value::as_str) != Some(requested_url)
                || injected.get("title") != full.get("title")
            {
                bail!("web_fetch 注入元数据与完整响应不一致");
            }
            let full_content = full
                .get("content")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("web_fetch 完整响应缺少 content"))?;
            let injected_content = injected
                .get("content")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("web_fetch 注入响应缺少 content"))?;
            const TRUNCATION_NOTICE: &str = "\n[抓取正文已按配置截断；完整响应保存在会话 trace 中]";
            let content_matches =
                if let Some(prefix) = injected_content.strip_suffix(TRUNCATION_NOTICE) {
                    !prefix.is_empty() && full_content.starts_with(prefix) && prefix != full_content
                } else {
                    injected_content == full_content
                };
            if !content_matches {
                bail!("web_fetch 注入正文不是完整响应的前缀");
            }
            let full_links = full
                .get("links")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            let injected_links = injected
                .get("links")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            if !injected_links.iter().all(|link| full_links.contains(link)) {
                bail!("web_fetch 注入链接不是完整响应的子集");
            }
            let mut urls = vec![requested_url.to_owned()];
            urls.extend(
                injected_links
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(ToOwned::to_owned),
            );
            urls.sort();
            urls.dedup();
            if urls != result.urls {
                bail!("web_fetch 记录的 URL 与注入响应不一致");
            }
        }
        name => bail!("未知联网工具结果 {name:?}"),
    }
    Ok(())
}

fn validate_web_source(source: &WebSourceTrace, result: &ToolResultTrace) -> Result<()> {
    if result.status != "ok"
        || source.observed_at != result.completed_at
        || !result.urls.contains(&source.url)
    {
        bail!("联网来源 {:?} 与对应工具结果不一致", source.url);
    }
    let full: serde_json::Value = serde_json::from_str(&result.full_response)
        .map_err(|error| anyhow::anyhow!("联网完整响应不是 JSON：{error}"))?;
    let valid = match (result.name.as_str(), source.kind.as_str()) {
        ("web_search", "search") => full
            .get("results")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| {
                items.iter().any(|item| {
                    item.get("url").and_then(serde_json::Value::as_str) == Some(source.url.as_str())
                        && item.get("title").and_then(serde_json::Value::as_str)
                            == Some(source.title.as_str())
                })
            }),
        ("web_fetch", "fetch") => {
            result
                .arguments
                .get("url")
                .and_then(serde_json::Value::as_str)
                == Some(source.url.as_str())
                && full.get("title").and_then(serde_json::Value::as_str)
                    == Some(source.title.as_str())
        }
        ("web_fetch", "fetch_link") => {
            full.get("title").and_then(serde_json::Value::as_str) == Some(source.title.as_str())
                && full
                    .get("links")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|links| {
                        links
                            .iter()
                            .any(|link| link.as_str() == Some(source.url.as_str()))
                    })
        }
        _ => false,
    };
    if !valid {
        bail!("联网来源 {:?} 无法由完整工具响应重建", source.url);
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextTrace {
    #[serde(default)]
    pub included_turn_ids: Vec<String>,
    #[serde(default)]
    pub omitted_turn_ids: Vec<String>,
    pub estimated_upper_tokens: Option<u64>,
    pub exact_input_tokens: Option<u64>,
    #[serde(default)]
    pub input_budget: u64,
    #[serde(default = "default_decision")]
    pub decision: String,
    #[serde(default)]
    pub active_context_start_before: usize,
    #[serde(default)]
    pub active_context_start_after: usize,
    #[serde(default)]
    pub context_items: Vec<ContextItemTrace>,
    #[serde(default)]
    pub context_sha256: Option<String>,
    #[serde(default)]
    pub request: Option<ModelRequestTrace>,
    #[serde(default)]
    pub identity_instruction: Option<String>,
    #[serde(default)]
    pub untrusted_history_wrapped: bool,
    #[serde(default = "legacy_inferred")]
    pub provenance_quality: ProvenanceQuality,
    #[serde(default)]
    pub retrieval: RetrievalTrace,
    #[serde(default)]
    pub knowledge: KnowledgeTrace,
    #[serde(default)]
    pub web: WebTrace,
}

fn default_decision() -> String {
    "none".to_owned()
}

impl Default for ContextTrace {
    fn default() -> Self {
        Self {
            included_turn_ids: Vec::new(),
            omitted_turn_ids: Vec::new(),
            estimated_upper_tokens: None,
            exact_input_tokens: None,
            input_budget: 0,
            decision: default_decision(),
            active_context_start_before: 0,
            active_context_start_after: 0,
            context_items: Vec::new(),
            context_sha256: None,
            request: None,
            identity_instruction: None,
            untrusted_history_wrapped: false,
            provenance_quality: ProvenanceQuality::Exact,
            retrieval: RetrievalTrace::default(),
            knowledge: KnowledgeTrace::default(),
            web: WebTrace::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Turn {
    pub id: String,
    pub created_at: String,
    pub updated_at: String,
    pub status: TurnStatus,
    #[serde(default)]
    pub user_content: String,
    #[serde(default)]
    pub assistant_content: String,
    #[serde(default)]
    pub thinking: String,
    #[serde(default)]
    pub usage: TokenUsage,
    #[serde(default = "TokenUsage::zero")]
    pub probe_usage: TokenUsage,
    #[serde(default)]
    pub context_trace: ContextTrace,
    #[serde(default)]
    pub request_started_at: Option<String>,
    pub done_reason: Option<String>,
    pub error: Option<String>,
}

impl Turn {
    pub fn pending(user_content: String) -> Self {
        let now = utc_now();
        Self {
            id: Uuid::new_v4().simple().to_string(),
            created_at: now.clone(),
            updated_at: now,
            status: TurnStatus::Pending,
            user_content,
            assistant_content: String::new(),
            thinking: String::new(),
            usage: TokenUsage::default(),
            probe_usage: TokenUsage::zero(),
            context_trace: ContextTrace::default(),
            request_started_at: None,
            done_reason: None,
            error: None,
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = utc_now();
    }

    pub fn context_eligible(&self) -> bool {
        matches!(self.status, TurnStatus::Complete | TurnStatus::Truncated)
            && !self.assistant_content.is_empty()
    }

    pub fn normalize(&mut self) {
        self.usage.refresh();
        self.probe_usage.refresh();
        self.context_trace.retrieval.normalize_usage();
        self.context_trace.web.normalize();
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Paused,
}

impl SessionStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Session {
    pub schema_version: u32,
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub status: SessionStatus,
    pub model: String,
    pub ollama_host: String,
    #[serde(default = "default_ai_name")]
    pub ai_name: String,
    pub system_prompt: String,
    pub think: bool,
    pub budget: BudgetConfig,
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    pub active_context_start_index: usize,
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub cumulative_usage: TokenUsage,
    #[serde(default)]
    pub cumulative_probe_usage: TokenUsage,
}

impl Session {
    pub fn new(
        id: String,
        model: String,
        ollama_host: String,
        system_prompt: String,
        budget: BudgetConfig,
        think: bool,
    ) -> Result<Self> {
        Self::new_named(
            id,
            model,
            ollama_host,
            default_ai_name(),
            system_prompt,
            budget,
            think,
        )
    }

    pub fn new_named(
        id: String,
        model: String,
        ollama_host: String,
        ai_name: String,
        system_prompt: String,
        budget: BudgetConfig,
        think: bool,
    ) -> Result<Self> {
        budget.validate()?;
        if ai_name.trim().is_empty() {
            bail!("AI 名称不能为空");
        }
        let now = utc_now();
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            id,
            title: "新会话".to_owned(),
            created_at: now.clone(),
            updated_at: now,
            status: SessionStatus::Active,
            model,
            ollama_host,
            ai_name,
            system_prompt,
            think,
            budget,
            retrieval: RetrievalConfig::default(),
            active_context_start_index: 0,
            turns: Vec::new(),
            cumulative_usage: TokenUsage::zero(),
            cumulative_probe_usage: TokenUsage::zero(),
        })
    }

    pub fn validate(&self) -> Result<()> {
        if !matches!(
            self.schema_version,
            LEGACY_SCHEMA_VERSION
                | PREVIOUS_SCHEMA_VERSION
                | PREVIOUS_SCHEMA_VERSION_V3
                | SCHEMA_VERSION
        ) {
            bail!("不支持的会话 schema 版本：{}", self.schema_version);
        }
        if self.id.is_empty() {
            bail!("会话 ID 不能为空");
        }
        if self.model.is_empty() || self.ollama_host.is_empty() {
            bail!("模型与 Ollama 地址不能为空");
        }
        if self.ai_name.trim().is_empty() {
            bail!("AI 名称不能为空");
        }
        if self.active_context_start_index > self.turns.len() {
            bail!("active_context_start_index 超出轮次数量");
        }
        let mut turn_ids = HashSet::new();
        if self.turns.iter().any(|turn| !turn_ids.insert(&turn.id)) {
            bail!("会话包含重复的轮次 ID");
        }
        for turn in &self.turns {
            turn.context_trace.web.validate()?;
            if turn.request_started_at.is_some()
                && turn.context_trace.provenance_quality == ProvenanceQuality::Exact
                && (turn.context_trace.context_items.is_empty()
                    || turn.context_trace.context_sha256.is_none()
                    || turn.context_trace.request.is_none())
            {
                bail!("轮次 {} 缺少精确回答上下文溯源", turn.id);
            }
        }
        self.budget.validate()?;
        self.retrieval.validate()
    }

    pub fn touch(&mut self) {
        self.updated_at = utc_now();
    }

    pub fn normalize_legacy_provenance(&mut self) {
        if self.schema_version == LEGACY_SCHEMA_VERSION {
            for turn in &mut self.turns {
                turn.context_trace.provenance_quality = ProvenanceQuality::LegacyInferred;
            }
        }
    }

    pub fn eligible_turns(
        &self,
        before_index: Option<usize>,
        honor_active_start: bool,
    ) -> Vec<(usize, &Turn)> {
        let start = if honor_active_start {
            self.active_context_start_index
        } else {
            0
        };
        let stop = before_index
            .unwrap_or(self.turns.len())
            .min(self.turns.len());
        if start > stop {
            return Vec::new();
        }
        self.turns[start..stop]
            .iter()
            .enumerate()
            .filter_map(|(offset, turn)| turn.context_eligible().then_some((start + offset, turn)))
            .collect()
    }

    pub fn refresh_cumulative_usage(&mut self) {
        let mut answers = TokenUsage::zero();
        let mut probes = TokenUsage::zero();
        for turn in &mut self.turns {
            turn.normalize();
            if turn.usage.input_tokens.is_some() && turn.usage.output_tokens.is_some() {
                answers.add(turn.usage);
            }
            probes.add(turn.probe_usage);
        }
        self.cumulative_usage = answers;
        self.cumulative_probe_usage = probes;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextPlan {
    pub messages: Vec<ChatMessage>,
    pub context_items: Vec<ContextItemTrace>,
    pub context_sha256: String,
    pub included_turn_ids: Vec<String>,
    pub omitted_turn_ids: Vec<String>,
    pub selected_history_indices: Vec<usize>,
    pub estimated_upper_tokens: Option<u64>,
    pub exact_input_tokens: Option<u64>,
    pub input_budget: u64,
    pub identity_instruction: String,
    pub untrusted_history_wrapped: bool,
    pub retrieval_trace: RetrievalTrace,
    pub evidence: Vec<SelectedEvidence>,
    pub knowledge_trace: KnowledgeTrace,
}

pub fn content_sha256(content: &str) -> String {
    format!("{:x}", Sha256::digest(content.as_bytes()))
}

pub fn event_id(session_id: &str, turn_id: Option<&str>, role: EventRole) -> String {
    let mut hasher = Sha256::new();
    hash_part(&mut hasher, b"hippocampus:event:v1");
    hash_part(&mut hasher, session_id.as_bytes());
    hash_part(&mut hasher, turn_id.unwrap_or("__system__").as_bytes());
    hash_part(&mut hasher, role.as_str().as_bytes());
    format!("evt_{:x}", hasher.finalize())
}

pub fn context_sha256(messages: &[ChatMessage]) -> String {
    let mut hasher = Sha256::new();
    hash_part(&mut hasher, b"hippocampus:context:v1");
    for message in messages {
        hash_part(&mut hasher, message.role.as_bytes());
        hash_part(&mut hasher, message.content.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub fn agent_context_sha256(messages: &[AgentMessage], tools: &[ToolDefinition]) -> String {
    let mut hasher = Sha256::new();
    hash_part(&mut hasher, b"hippocampus:agent-context:v1");
    let messages = serde_json::to_vec(messages).expect("agent messages are serializable");
    let tools = serde_json::to_vec(tools).expect("tool definitions are serializable");
    hash_part(&mut hasher, &messages);
    hash_part(&mut hasher, &tools);
    format!("{:x}", hasher.finalize())
}

fn hash_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

impl ContextPlan {
    pub fn usage_ratio(&self) -> Option<f64> {
        let value = self.exact_input_tokens.or(self.estimated_upper_tokens)?;
        (self.input_budget > 0).then_some(value as f64 / self.input_budget as f64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatEventKind {
    Thinking,
    Content,
    Usage,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatEvent {
    pub kind: ChatEventKind,
    pub text: String,
    pub live_output_tokens: Option<u64>,
    pub usage: Option<TokenUsage>,
    pub done_reason: Option<String>,
}

impl ChatEvent {
    pub fn text(kind: ChatEventKind, text: String, live_output_tokens: u64) -> Self {
        Self {
            kind,
            text,
            live_output_tokens: Some(live_output_tokens),
            usage: None,
            done_reason: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budget_matches_legacy_behavior() {
        let budget = BudgetConfig::default();
        assert_eq!(budget.input_budget(), 28_160);
        assert_eq!(budget.probe_threshold(), 22_528);
        assert_eq!(budget.warning_threshold(), 25_344);
        assert_eq!(budget.trim_target(), 22_528);
    }

    #[test]
    fn unknown_usage_stays_unknown() {
        let usage = TokenUsage::new(None, Some(4));
        assert_eq!(usage.total_tokens, None);
    }

    #[test]
    fn event_ids_are_stable_and_content_independent() {
        let first = event_id("session", Some("turn"), EventRole::User);
        let second = event_id("session", Some("turn"), EventRole::User);
        assert_eq!(first, second);
        assert_ne!(
            first,
            event_id("session", Some("turn"), EventRole::Assistant)
        );
        assert_ne!(first, event_id("other", Some("turn"), EventRole::User));
        assert_eq!(first.len(), 68);
    }

    #[test]
    fn version_three_retrieval_trace_gets_schema_four_defaults() {
        let trace: RetrievalTrace = serde_json::from_value(serde_json::json!({
            "status": "complete",
            "current_query_event_id": "evt_previous",
            "query_terms": ["memory"]
        }))
        .unwrap();
        assert_eq!(trace.query_kind, QueryKind::GeneralSemantic);
        assert_eq!(trace.channels, Vec::<ChannelTrace>::new());
        assert_eq!(trace.graph_paths, Vec::<GraphPathTrace>::new());
        assert_eq!(
            trace.budget_allocation,
            BudgetAllocationTrace {
                query_kind: QueryKind::GeneralSemantic,
                recent_history_percent: 45,
                exact_or_state_percent: 30,
                episode_percent: 15,
                graph_percent: 10,
                ..Default::default()
            }
        );
        assert!(trace.warnings.is_empty());
        assert_eq!(trace.elapsed_ms, 0);

        let serialized = serde_json::to_value(trace).unwrap();
        assert_eq!(serialized["query_kind"], "general_semantic");
        assert_eq!(serialized["budget_allocation"]["graph_percent"], 10);
    }

    #[test]
    fn version_three_session_round_trips_and_upgrades_to_version_four() {
        let mut value = serde_json::to_value(
            Session::new(
                "session-v3".into(),
                "model".into(),
                "http://localhost:11434".into(),
                DEFAULT_SYSTEM_PROMPT.into(),
                BudgetConfig::default(),
                false,
            )
            .unwrap(),
        )
        .unwrap();
        value["schema_version"] = serde_json::json!(PREVIOUS_SCHEMA_VERSION_V3);

        let mut session: Session = serde_json::from_value(value).unwrap();
        session.validate().unwrap();
        assert_eq!(session.schema_version, PREVIOUS_SCHEMA_VERSION_V3);

        session.schema_version = SCHEMA_VERSION;
        let saved = serde_json::to_value(session).unwrap();
        assert_eq!(saved["schema_version"], SCHEMA_VERSION);
    }

    #[test]
    fn all_historical_session_schema_versions_remain_valid() {
        let mut session = Session::new(
            "compatible".into(),
            "model".into(),
            "http://localhost:11434".into(),
            DEFAULT_SYSTEM_PROMPT.into(),
            BudgetConfig::default(),
            false,
        )
        .unwrap();
        for version in 1..=SCHEMA_VERSION {
            session.schema_version = version;
            session.validate().unwrap();
        }
        session.schema_version = SCHEMA_VERSION + 1;
        assert!(session.validate().is_err());
    }
}
