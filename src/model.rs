use std::collections::HashSet;

use anyhow::{Result, bail};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::knowledge::KnowledgeTrace;

pub const SCHEMA_VERSION: u32 = 3;
pub const LEGACY_SCHEMA_VERSION: u32 = 1;
pub const PREVIOUS_SCHEMA_VERSION: u32 = 2;
pub const DEFAULT_SYSTEM_PROMPT: &str =
    "你是一个乐于助人的AI助手，你的任务是解决用户的问题或者与用户对话。";

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
        if self.candidate_limit == 0
            || self.candidate_limit > 512
            || self.max_selected == 0
            || self.max_selected > self.candidate_limit
            || self.evidence_char_budget == 0
            || self.evidence_char_budget > 32_768
            || self.expansion_char_budget > 16_384
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectedEvidence {
    pub span: SourceSpan,
    pub content_sha256: String,
    pub role: EventRole,
    pub kind: EvidenceKind,
    pub originating_candidate_rank: Option<usize>,
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
    pub selected_evidence: Vec<SelectedEvidence>,
    #[serde(default)]
    pub error: Option<String>,
}

impl Default for RetrievalTrace {
    fn default() -> Self {
        Self {
            status: "not_run".into(),
            current_query_event_id: String::new(),
            query_terms: Vec::new(),
            config: RetrievalConfig::default(),
            candidates: Vec::new(),
            selected_evidence: Vec::new(),
            error: None,
        }
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    #[serde(default = "legacy_inferred")]
    pub provenance_quality: ProvenanceQuality,
    #[serde(default)]
    pub retrieval: RetrievalTrace,
    #[serde(default)]
    pub knowledge: KnowledgeTrace,
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
            provenance_quality: ProvenanceQuality::Exact,
            retrieval: RetrievalTrace::default(),
            knowledge: KnowledgeTrace::default(),
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
            LEGACY_SCHEMA_VERSION | PREVIOUS_SCHEMA_VERSION | SCHEMA_VERSION
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
}
