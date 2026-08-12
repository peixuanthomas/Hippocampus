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
    pub web_search: WebSearchConfig,
    pub knowledge: KnowledgeConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            ai_name: None,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_owned(),
            web_search: WebSearchConfig::default(),
            knowledge: KnowledgeConfig::default(),
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
        self.web_search.validate()?;
        self.knowledge.validate()
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedConfig {
    pub config: AppConfig,
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct WebSearchConfig {
    pub enabled: bool,
    pub max_results: usize,
    pub max_tool_rounds: usize,
    pub max_tool_calls: usize,
    pub max_injected_chars_per_fetch: usize,
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_results: 5,
            max_tool_rounds: 4,
            max_tool_calls: 8,
            max_injected_chars_per_fetch: 12_000,
        }
    }
}

impl WebSearchConfig {
    pub fn validate(&self) -> Result<()> {
        if !(1..=10).contains(&self.max_results) {
            bail!("web_search.max_results 必须在 1..=10 之间");
        }
        if !(1..=16).contains(&self.max_tool_rounds) {
            bail!("web_search.max_tool_rounds 必须在 1..=16 之间");
        }
        if !(1..=64).contains(&self.max_tool_calls) {
            bail!("web_search.max_tool_calls 必须在 1..=64 之间");
        }
        if !(1..=1_048_576).contains(&self.max_injected_chars_per_fetch) {
            bail!("web_search.max_injected_chars_per_fetch 必须在 1..=1048576 之间");
        }
        Ok(())
    }
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
    Url,
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
        if self.kind == KnowledgeSourceKind::Url {
            let url = reqwest::Url::parse(&self.location)
                .with_context(|| format!("knowledge source {:?} 的 URL 无效", self.id))?;
            if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                bail!("knowledge source {:?} 只支持 HTTP(S) URL", self.id);
            }
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
        assert!(!config.web_search.enabled);
        assert!(config.knowledge.sources.is_empty());
    }

    #[test]
    fn missing_explicit_config_is_an_error() {
        let root = tempfile::tempdir().unwrap();
        let error = AppConfig::load(Some(&root.path().join("missing.toml"))).unwrap_err();
        assert!(error.to_string().contains("找不到配置文件"));
    }

    #[test]
    fn strict_config_resolves_relative_paths() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("custom.toml");
        fs::write(
            &path,
            r#"ai_name = "hippocampus"

[web_search]
enabled = true

[knowledge]
auto_sync = true

[[knowledge.sources]]
id = "notes"
kind = "path"
location = "notes"
"#,
        )
        .unwrap();
        let loaded = AppConfig::load(Some(&path)).unwrap();
        assert_eq!(loaded.config.ai_name(), "hippocampus");
        assert_eq!(
            loaded.config.knowledge.sources[0].location,
            root.path().join("notes").to_string_lossy()
        );
    }

    #[test]
    fn rejects_unknown_fields_and_duplicate_sources() {
        let unknown: Result<AppConfig, _> = toml::from_str("mystery = true");
        assert!(unknown.is_err());

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

        let mut invalid_budget = AppConfig::default();
        invalid_budget.web_search.max_tool_calls = 0;
        assert!(invalid_budget.validate().is_err());

        let empty_name: Result<AppConfig, _> = toml::from_str("ai_name = '   '");
        assert!(empty_name.unwrap().validate().is_err());
    }
}
