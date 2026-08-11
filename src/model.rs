use anyhow::{Result, bail};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_SYSTEM_PROMPT: &str = "你是一个有帮助、诚实且简洁的 AI 助手。";

pub fn utc_now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::AutoSi, false)
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    pub system_prompt: String,
    pub think: bool,
    pub budget: BudgetConfig,
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
        budget.validate()?;
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
            system_prompt,
            think,
            budget,
            active_context_start_index: 0,
            turns: Vec::new(),
            cumulative_usage: TokenUsage::zero(),
            cumulative_probe_usage: TokenUsage::zero(),
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            bail!("不支持的会话 schema 版本：{}", self.schema_version);
        }
        if self.id.is_empty() {
            bail!("会话 ID 不能为空");
        }
        if self.model.is_empty() || self.ollama_host.is_empty() {
            bail!("模型与 Ollama 地址不能为空");
        }
        if self.active_context_start_index > self.turns.len() {
            bail!("active_context_start_index 超出轮次数量");
        }
        self.budget.validate()
    }

    pub fn touch(&mut self) {
        self.updated_at = utc_now();
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPlan {
    pub messages: Vec<ChatMessage>,
    pub included_turn_ids: Vec<String>,
    pub omitted_turn_ids: Vec<String>,
    pub selected_history_indices: Vec<usize>,
    pub estimated_upper_tokens: Option<u64>,
    pub exact_input_tokens: Option<u64>,
    pub input_budget: u64,
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
}
