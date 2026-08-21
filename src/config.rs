use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::model::DEFAULT_SYSTEM_PROMPT;

pub const DEFAULT_CONFIG_FILENAME: &str = "config.toml";
pub const DEFAULT_AI_NAME: &str = "LLM";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    pub ai_name: Option<String>,
    pub system_prompt: String,
    pub knowledge: KnowledgeConfig,
    pub memory: MemoryConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            ai_name: None,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_owned(),
            knowledge: KnowledgeConfig::default(),
            memory: MemoryConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn ai_name(&self) -> &str {
        self.ai_name.as_deref().unwrap_or(DEFAULT_AI_NAME)
    }

    pub fn load(explicit_path: Option<&Path>) -> Result<LoadedConfig> {
        let path = match explicit_path {
            Some(path) => {
                if !path.is_file() {
                    bail!("找不到配置文件 {}", path.display());
                }
                Some(absolute_path(path)?)
            }
            None => {
                let candidate = std::env::current_dir()?.join(DEFAULT_CONFIG_FILENAME);
                candidate.is_file().then_some(candidate)
            }
        };
        let Some(path) = path else {
            let config = Self::default();
            config.validate()?;
            return Ok(LoadedConfig { config, path: None });
        };
        let text = fs::read_to_string(&path)
            .with_context(|| format!("无法读取配置文件 {}", path.display()))?;
        let mut config: Self =
            toml::from_str(&text).with_context(|| format!("配置文件 {} 无效", path.display()))?;
        config.resolve_relative_sources(path.parent().unwrap_or_else(|| Path::new(".")));
        config.validate()?;
        Ok(LoadedConfig {
            config,
            path: Some(path),
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self
            .ai_name
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            bail!("ai_name 不能为空");
        }
        self.knowledge.validate()?;
        self.memory.validate()
    }

    fn resolve_relative_sources(&mut self, base: &Path) {
        for source in &mut self.knowledge.sources {
            if source.kind == KnowledgeSourceKind::Path {
                let path = Path::new(&source.location);
                if !path.is_absolute() {
                    source.location = base.join(path).to_string_lossy().into_owned();
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    pub config_version: u32,
    pub enabled: bool,
    pub embedding_model: String,
    pub embedding_dimensions: usize,
    pub embedding_batch_size: usize,
    pub candidate_limit: usize,
    pub vector_candidate_limit: usize,
    pub graph_candidate_limit: usize,
    pub max_graph_depth: usize,
    pub rrf_k: usize,
    pub consolidation_timeout_secs: u64,
    pub embedding_timeout_secs: u64,
    pub search_timeout_ms: u64,
    pub episode_gap_minutes: u64,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub budgets: AdaptiveBudgetConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            config_version: 1,
            enabled: false,
            embedding_model: "qwen3-embedding:8b".into(),
            embedding_dimensions: 1_024,
            embedding_batch_size: 16,
            candidate_limit: 64,
            vector_candidate_limit: 64,
            graph_candidate_limit: 32,
            max_graph_depth: 2,
            rrf_k: 60,
            consolidation_timeout_secs: 600,
            embedding_timeout_secs: 600,
            search_timeout_ms: 15_000,
            episode_gap_minutes: 30,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
            budgets: AdaptiveBudgetConfig::default(),
        }
    }
}

impl MemoryConfig {
    pub fn validate(&self) -> Result<()> {
        if self.config_version != 1 {
            bail!("memory.config_version 仅支持版本 1");
        }
        if self.embedding_model.trim().is_empty() {
            bail!("memory.embedding_model 不能为空");
        }
        if !(32..=4_096).contains(&self.embedding_dimensions) {
            bail!("memory.embedding_dimensions 必须在 32..=4096 之间");
        }
        if !(1..=256).contains(&self.embedding_batch_size) {
            bail!("memory.embedding_batch_size 必须在 1..=256 之间");
        }
        for (name, value) in [
            ("candidate_limit", self.candidate_limit),
            ("vector_candidate_limit", self.vector_candidate_limit),
            ("graph_candidate_limit", self.graph_candidate_limit),
        ] {
            if !(1..=512).contains(&value) {
                bail!("memory.{name} 必须在 1..=512 之间");
            }
        }
        if !(1..=60_000).contains(&self.search_timeout_ms) {
            bail!("memory.search_timeout_ms 必须在 1..=60000 之间");
        }
        if !(1..=2).contains(&self.max_graph_depth) {
            bail!("memory.max_graph_depth 必须在 1..=2 之间");
        }
        if !(1..=1_000).contains(&self.rrf_k) {
            bail!("memory.rrf_k 必须在 1..=1000 之间");
        }
        for (name, value) in [
            (
                "consolidation_timeout_secs",
                self.consolidation_timeout_secs,
            ),
            ("embedding_timeout_secs", self.embedding_timeout_secs),
        ] {
            if !(1..=3_600).contains(&value) {
                bail!("memory.{name} 必须在 1..=3600 之间");
            }
        }
        if !(1..=1_440).contains(&self.episode_gap_minutes) {
            bail!("memory.episode_gap_minutes 必须在 1..=1440 之间");
        }
        if !(2..=64).contains(&self.hnsw_m) {
            bail!("memory.hnsw_m 必须在 2..=64 之间");
        }
        if self.hnsw_ef_construction < self.hnsw_m || self.hnsw_ef_construction > 4_096 {
            bail!("memory.hnsw_ef_construction 必须在 hnsw_m..=4096 之间");
        }
        if !(1..=4_096).contains(&self.hnsw_ef_search) {
            bail!("memory.hnsw_ef_search 必须在 1..=4096 之间");
        }
        self.budgets.validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct AdaptiveBudgetConfig {
    pub exact_fact: MemoryBudgetConfig,
    pub general_semantic: MemoryBudgetConfig,
    pub event_recap: MemoryBudgetConfig,
    pub temporal_state: MemoryBudgetConfig,
    pub multi_hop: MemoryBudgetConfig,
}

impl Default for AdaptiveBudgetConfig {
    fn default() -> Self {
        Self {
            exact_fact: MemoryBudgetConfig::new(45, 40, 5, 10),
            general_semantic: MemoryBudgetConfig::new(45, 30, 15, 10),
            event_recap: MemoryBudgetConfig::new(35, 20, 35, 10),
            temporal_state: MemoryBudgetConfig::new(35, 35, 15, 15),
            multi_hop: MemoryBudgetConfig::new(30, 25, 15, 30),
        }
    }
}

impl AdaptiveBudgetConfig {
    fn validate(&self) -> Result<()> {
        for (name, budget) in [
            ("exact_fact", &self.exact_fact),
            ("general_semantic", &self.general_semantic),
            ("event_recap", &self.event_recap),
            ("temporal_state", &self.temporal_state),
            ("multi_hop", &self.multi_hop),
        ] {
            if budget.total() != 100 {
                bail!("memory.budgets.{name} 的比例之和必须恰好为 100");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MemoryBudgetConfig {
    pub recent_history: u8,
    pub exact_or_state: u8,
    pub episode: u8,
    pub graph: u8,
}

impl MemoryBudgetConfig {
    const fn new(recent_history: u8, exact_or_state: u8, episode: u8, graph: u8) -> Self {
        Self {
            recent_history,
            exact_or_state,
            episode,
            graph,
        }
    }

    const fn total(self) -> u16 {
        self.recent_history as u16
            + self.exact_or_state as u16
            + self.episode as u16
            + self.graph as u16
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedConfig {
    pub config: AppConfig,
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct KnowledgeConfig {
    pub auto_sync: bool,
    pub candidate_limit: usize,
    pub max_selected: usize,
    pub evidence_char_budget: usize,
    pub sources: Vec<KnowledgeSourceConfig>,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            auto_sync: false,
            candidate_limit: 64,
            max_selected: 4,
            evidence_char_budget: 3_200,
            sources: Vec::new(),
        }
    }
}

impl KnowledgeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.candidate_limit == 0 || self.candidate_limit > 512 {
            bail!("knowledge.candidate_limit 必须在 1..=512 之间");
        }
        if self.max_selected == 0 || self.max_selected > self.candidate_limit {
            bail!("knowledge.max_selected 必须在 1..=candidate_limit 之间");
        }
        if self.evidence_char_budget == 0 || self.evidence_char_budget > 32_768 {
            bail!("knowledge.evidence_char_budget 必须在 1..=32768 之间");
        }
        let mut ids = HashSet::new();
        for source in &self.sources {
            source.validate()?;
            if !ids.insert(source.id.as_str()) {
                bail!("knowledge.sources 包含重复 id {:?}", source.id);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum KnowledgeSourceKind {
    Path,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeSourceConfig {
    pub id: String,
    pub kind: KnowledgeSourceKind,
    pub location: String,
}

impl KnowledgeSourceConfig {
    fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty()
            || !self
                .id
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
        {
            bail!("knowledge source id 必须是非空 ASCII 字母数字、点、下划线或短横线");
        }
        if self.location.trim().is_empty() {
            bail!("knowledge source {:?} 的 location 不能为空", self.id);
        }
        Ok(())
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_implicit_config_uses_safe_defaults() {
        let config = AppConfig::default();
        assert_eq!(config.ai_name(), "LLM");
        assert!(config.knowledge.sources.is_empty());
        assert!(!config.memory.enabled);
    }

    #[test]
    fn missing_explicit_config_is_an_error() {
        let root = tempfile::tempdir().unwrap();
        let error = AppConfig::load(Some(&root.path().join("missing.toml"))).unwrap_err();
        assert!(error.to_string().contains("找不到配置文件"));
    }

    #[test]
    fn checked_in_config_enables_memory_with_defaulted_fields() {
        let config: AppConfig = toml::from_str(include_str!("../config.toml")).unwrap();
        config.validate().unwrap();
        assert!(config.memory.enabled);
        assert_eq!(config.memory.embedding_model, "qwen3-embedding:8b");
        assert_eq!(config.memory.embedding_dimensions, 1_024);
        assert_eq!(config.memory.budgets, AdaptiveBudgetConfig::default());
    }

    #[test]
    fn strict_config_resolves_relative_paths() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("custom.toml");
        fs::write(
            &path,
            r#"ai_name = "hippocampus"

[knowledge]
auto_sync = true

[memory]
enabled = true
embedding_model = "test-embedding"
embedding_dimensions = 768

[[knowledge.sources]]
id = "notes"
kind = "path"
location = "notes"
"#,
        )
        .unwrap();
        let loaded = AppConfig::load(Some(&path)).unwrap();
        assert_eq!(loaded.config.ai_name(), "hippocampus");
        assert!(loaded.config.memory.enabled);
        assert_eq!(loaded.config.memory.embedding_model, "test-embedding");
        assert_eq!(loaded.config.memory.embedding_dimensions, 768);
        assert_eq!(loaded.config.memory.embedding_batch_size, 16);
        assert_eq!(
            loaded.config.knowledge.sources[0].location,
            root.path().join("notes").to_string_lossy()
        );
    }

    #[test]
    fn rejects_unknown_fields_and_duplicate_sources() {
        let unknown: Result<AppConfig, _> = toml::from_str("mystery = true");
        assert!(unknown.is_err());
        let removed_web_config: Result<AppConfig, _> =
            toml::from_str("[web_search]\nenabled = true");
        assert!(removed_web_config.is_err());
        let remote_knowledge_source: Result<AppConfig, _> = toml::from_str(
            r#"[knowledge]

[[knowledge.sources]]
id = "remote"
kind = "url"
location = "https://example.com/docs"
"#,
        );
        assert!(remote_knowledge_source.is_err());

        let mut config = AppConfig::default();
        config.knowledge.sources = vec![
            KnowledgeSourceConfig {
                id: "same".into(),
                kind: KnowledgeSourceKind::Path,
                location: "/one".into(),
            },
            KnowledgeSourceConfig {
                id: "same".into(),
                kind: KnowledgeSourceKind::Path,
                location: "/two".into(),
            },
        ];
        assert!(config.validate().is_err());

        let empty_name: Result<AppConfig, _> = toml::from_str("ai_name = '   '");
        assert!(empty_name.unwrap().validate().is_err());
    }

    #[test]
    fn memory_defaults_match_the_version_one_table() {
        let memory = MemoryConfig::default();
        assert_eq!(memory.config_version, 1);
        assert_eq!(memory.embedding_model, "qwen3-embedding:8b");
        assert_eq!(memory.embedding_dimensions, 1_024);
        assert_eq!(memory.embedding_batch_size, 16);
        assert_eq!(memory.candidate_limit, 64);
        assert_eq!(memory.vector_candidate_limit, 64);
        assert_eq!(memory.graph_candidate_limit, 32);
        assert_eq!(memory.max_graph_depth, 2);
        assert_eq!(memory.rrf_k, 60);
        assert_eq!(memory.consolidation_timeout_secs, 600);
        assert_eq!(memory.embedding_timeout_secs, 600);
        assert_eq!(memory.search_timeout_ms, 15_000);
        assert_eq!(memory.episode_gap_minutes, 30);
        assert_eq!(memory.hnsw_m, 16);
        assert_eq!(memory.hnsw_ef_construction, 200);
        assert_eq!(memory.hnsw_ef_search, 64);
        assert_eq!(
            memory.budgets,
            AdaptiveBudgetConfig {
                exact_fact: MemoryBudgetConfig::new(45, 40, 5, 10),
                general_semantic: MemoryBudgetConfig::new(45, 30, 15, 10),
                event_recap: MemoryBudgetConfig::new(35, 20, 35, 10),
                temporal_state: MemoryBudgetConfig::new(35, 35, 15, 15),
                multi_hop: MemoryBudgetConfig::new(30, 25, 15, 30),
            }
        );
        memory.validate().unwrap();
    }

    #[test]
    fn memory_validation_rejects_representative_invalid_values() {
        let invalid = [
            |memory: &mut MemoryConfig| memory.config_version = 2,
            |memory: &mut MemoryConfig| memory.embedding_dimensions = 31,
            |memory: &mut MemoryConfig| memory.embedding_batch_size = 0,
            |memory: &mut MemoryConfig| memory.graph_candidate_limit = 513,
            |memory: &mut MemoryConfig| memory.max_graph_depth = 3,
            |memory: &mut MemoryConfig| memory.rrf_k = 0,
            |memory: &mut MemoryConfig| memory.embedding_timeout_secs = 3_601,
            |memory: &mut MemoryConfig| memory.search_timeout_ms = 0,
            |memory: &mut MemoryConfig| memory.episode_gap_minutes = 0,
            |memory: &mut MemoryConfig| memory.hnsw_m = 1,
            |memory: &mut MemoryConfig| memory.hnsw_ef_search = 4_097,
        ];
        for mutate in invalid {
            let mut memory = MemoryConfig::default();
            mutate(&mut memory);
            assert!(memory.validate().is_err());
        }

        let blank_model = MemoryConfig {
            embedding_model: "   ".into(),
            ..MemoryConfig::default()
        };
        assert!(blank_model.validate().is_err());

        let construction_too_small = MemoryConfig {
            hnsw_ef_construction: 15,
            ..MemoryConfig::default()
        };
        assert!(construction_too_small.validate().is_err());

        let mut invalid_budget = MemoryConfig::default();
        invalid_budget.budgets.multi_hop.graph = 29;
        assert!(invalid_budget.validate().is_err());
    }
}
