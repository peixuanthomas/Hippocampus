use std::collections::{HashMap, HashSet};

use chrono::{DateTime, FixedOffset};
use rusqlite::{Connection, OptionalExtension, Transaction, params, types::ValueRef};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use crate::model::{EventRole, TurnStatus};
use crate::retrieval::{RetrievalError, RetrievalResult, RetrievalStore, StoredEvent};

pub const CONSOLIDATION_MAX_TURNS: usize = 16;
pub const CONSOLIDATION_MAX_CHARS: usize = 24_000;

const CONSOLIDATION_BATCH_KEY_VERSION: &str = "hippocampus-consolidation-batch-v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsolidationEvent {
    pub event_id: String,
    pub turn_id: String,
    pub sequence: usize,
    pub role: EventRole,
    pub created_at: String,
    pub content: String,
    pub content_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsolidationInputBatch {
    pub batch_key: String,
    pub session_id: String,
    pub watermark_before: usize,
    pub from_sequence: usize,
    pub through_sequence: usize,
    pub through_event_id: String,
    pub through_event_sha256: String,
    pub turn_count: usize,
    pub char_count: usize,
    pub events: Vec<ConsolidationEvent>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationAttemptStatus {
    Applied,
    Rejected,
    ModelError,
    Cancelled,
}

impl ConsolidationAttemptStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Applied => "applied",
            Self::Rejected => "rejected",
            Self::ModelError => "model_error",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsolidationAttemptRecord {
    pub attempt_id: String,
    pub batch_key: String,
    pub session_id: String,
    pub from_sequence: usize,
    pub through_sequence: usize,
    pub trigger: String,
    pub model: String,
    pub request_json: String,
    pub request_sha256: String,
    pub input_event_ids: Vec<String>,
    pub input_event_hashes: Vec<String>,
    pub response_json: Option<String>,
    pub response_sha256: Option<String>,
    pub status: ConsolidationAttemptStatus,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub latency_ms: u64,
    pub started_at: String,
    pub completed_at: String,
    pub validation_json: Option<String>,
    pub error_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsolidationWatermark {
    pub session_id: String,
    pub through_sequence: usize,
    pub through_event_id: Option<String>,
    pub through_event_sha256: Option<String>,
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEntityKind {
    Person,
    Organization,
    Location,
    Object,
    Concept,
    Unknown,
}

impl MemoryEntityKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Location => "location",
            Self::Object => "object",
            Self::Concept => "concept",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntityResolution {
    #[serde(rename = "self")]
    SelfEntity,
    New,
    Existing,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntityDisambiguation {
    Resolved,
    Pending,
}

impl EntityDisambiguation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Resolved => "resolved",
            Self::Pending => "pending",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntityResolutionBasis {
    SelfPronoun,
    FirstMention,
    ExplicitAlias,
    StableIdentifier,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MemoryAliasKind {
    ExplicitAlias,
    StableIdentifier,
}

impl MemoryAliasKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitAlias => "explicit_alias",
            Self::StableIdentifier => "stable_identifier",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct ConsolidationQuote {
    pub event_id: String,
    pub start_char: usize,
    pub end_char: usize,
    pub content_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EntityAliasOutput {
    pub text: String,
    pub kind: MemoryAliasKind,
    pub stable_identifier_kind: Option<String>,
    pub evidence: ConsolidationQuote,
    pub proof_evidence: ConsolidationQuote,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConsolidatedEntityOutput {
    pub local_id: String,
    pub name: String,
    pub kind: MemoryEntityKind,
    pub resolution: EntityResolution,
    pub disambiguation: EntityDisambiguation,
    pub basis: EntityResolutionBasis,
    pub existing_entity_id: Option<String>,
    pub name_evidence: ConsolidationQuote,
    pub resolution_evidence: Option<ConsolidationQuote>,
    pub aliases: Vec<EntityAliasOutput>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationClaimObjectKind {
    Text,
    Entity,
}

impl ConsolidationClaimObjectKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Entity => "entity",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConsolidatedClaimObject {
    pub kind: ConsolidationClaimObjectKind,
    pub text: Option<String>,
    pub entity_ref: Option<String>,
    pub span: Option<ConsolidationQuote>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClaimPolarity {
    Assert,
    Deny,
}

impl ClaimPolarity {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Assert => "assert",
            Self::Deny => "deny",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClaimCardinality {
    Single,
    Multi,
}

impl ClaimCardinality {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Multi => "multi",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClaimCertainty {
    Certain,
    Uncertain,
}

impl ClaimCertainty {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Certain => "certain",
            Self::Uncertain => "uncertain",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClaimDisposition {
    New,
    Confirm,
    Correct,
    Replace,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryClaimState {
    Active,
    Superseded,
    Conflicted,
    Uncertain,
}

impl MemoryClaimState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Conflicted => "conflicted",
            Self::Uncertain => "uncertain",
        }
    }

    const fn is_live(self) -> bool {
        matches!(self, Self::Active | Self::Uncertain | Self::Conflicted)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationEvidenceKind {
    Assertion,
    UserConfirmation,
    Correction,
    Temporal,
}

impl ConsolidationEvidenceKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Assertion => "assertion",
            Self::UserConfirmation => "user_confirmation",
            Self::Correction => "correction",
            Self::Temporal => "temporal",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConsolidationClaimEvidence {
    pub kind: ConsolidationEvidenceKind,
    pub quote: ConsolidationQuote,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConsolidatedClaimOutput {
    pub local_id: String,
    pub subject_ref: String,
    pub predicate_key: String,
    pub object: ConsolidatedClaimObject,
    pub polarity: ClaimPolarity,
    pub cardinality: ClaimCardinality,
    pub certainty: ClaimCertainty,
    pub disposition: ClaimDisposition,
    pub replaces_claim_ids: Vec<String>,
    pub conflicts_with_claim_ids: Vec<String>,
    pub event_time: Option<String>,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub evidence: Vec<ConsolidationClaimEvidence>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BoundarySuggestionReason {
    ExplicitTopicTransition,
    ModelTopicShift,
}

impl BoundarySuggestionReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitTopicTransition => "explicit_topic_transition",
            Self::ModelTopicShift => "model_topic_shift",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConsolidationBoundaryOutput {
    pub before_event_id: String,
    pub reason: BoundarySuggestionReason,
    pub evidence: Vec<ConsolidationQuote>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StructuredConsolidationOutput {
    pub entities: Vec<ConsolidatedEntityOutput>,
    pub claims: Vec<ConsolidatedClaimOutput>,
    pub boundaries: Vec<ConsolidationBoundaryOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryAliasCandidate {
    pub alias_id: String,
    pub text: String,
    pub normalized_text: String,
    pub kind: MemoryAliasKind,
    pub stable_identifier_kind: Option<String>,
    pub session_id: String,
    pub batch_key: String,
    pub event_id: String,
    pub start_char: usize,
    pub end_char: usize,
    pub content_sha256: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryClaimEvidenceCandidate {
    pub evidence_id: String,
    pub session_id: String,
    pub batch_key: String,
    pub event_id: String,
    pub sequence: usize,
    pub role: EventRole,
    pub kind: ConsolidationEvidenceKind,
    pub start_char: usize,
    pub end_char: usize,
    pub content_sha256: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryEntityCandidate {
    pub entity_id: String,
    pub kind: MemoryEntityKind,
    pub canonical_name: String,
    pub normalized_name: String,
    pub disambiguation: EntityDisambiguation,
    pub created_session_id: String,
    pub created_batch_key: String,
    pub created_event_id: String,
    pub created_start: usize,
    pub created_end: usize,
    pub created_hash: String,
    pub created_at: String,
    pub updated_at: String,
    pub aliases: Vec<MemoryAliasCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryClaimCandidate {
    pub claim_id: String,
    pub session_id: String,
    pub subject_entity_id: String,
    pub predicate_key: String,
    pub object_kind: ConsolidationClaimObjectKind,
    pub object_text: Option<String>,
    pub object_entity_id: Option<String>,
    pub normalized_object: String,
    pub polarity: ClaimPolarity,
    pub cardinality: ClaimCardinality,
    pub certainty: ClaimCertainty,
    pub state: MemoryClaimState,
    pub asserted_at: String,
    pub event_time: Option<String>,
    pub valid_from: String,
    pub valid_to: Option<String>,
    pub reference_time: String,
    pub created_batch_key: String,
    pub updated_batch_key: String,
    pub created_at: String,
    pub updated_at: String,
    pub evidence: Vec<MemoryClaimEvidenceCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsolidationCandidateSnapshot {
    pub entities: Vec<MemoryEntityCandidate>,
    pub claims: Vec<MemoryClaimCandidate>,
    pub snapshot_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsolidationApplyReport {
    pub session_id: String,
    pub batch_key: String,
    pub watermark_before: usize,
    pub watermark_after: usize,
    pub entities_created: usize,
    pub entities_reused: usize,
    pub aliases_created: usize,
    pub claims_created: usize,
    pub claims_confirmed: usize,
    pub claims_superseded: usize,
    pub claims_conflicted: usize,
    pub evidence_created: usize,
    pub boundaries_created: usize,
}

#[derive(Debug, Error)]
pub enum ConsolidationApplyError {
    #[error("巩固输出被拒绝：{message}")]
    Rejected {
        validation_json: String,
        message: String,
    },
    #[error("巩固输入已过期：{message}")]
    Stale { message: String },
    #[error(transparent)]
    Retrieval(#[from] RetrievalError),
}

pub type ConsolidationApplyResult<T> = std::result::Result<T, ConsolidationApplyError>;

/// Exact JSON Schema passed to Ollama's structured-output `format` field.
/// Deterministic Rust validation remains authoritative after decoding.
pub fn structured_consolidation_schema() -> Value {
    let nullable_string = || json!({"type": ["string", "null"], "maxLength": 512});
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": false,
        "required": ["entities", "claims", "boundaries"],
        "properties": {
            "entities": {"type": "array", "maxItems": 128, "items": {"$ref": "#/$defs/entity"}},
            "claims": {"type": "array", "maxItems": 256, "items": {"$ref": "#/$defs/claim"}},
            "boundaries": {"type": "array", "maxItems": 64, "items": {"$ref": "#/$defs/boundary"}}
        },
        "$defs": {
            "quote": {
                "type": "object", "additionalProperties": false,
                "required": ["event_id", "start_char", "end_char", "content_sha256"],
                "properties": {
                    "event_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "start_char": {"type": "integer", "minimum": 0},
                    "end_char": {"type": "integer", "minimum": 0},
                    "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}
                }
            },
            "alias": {
                "type": "object", "additionalProperties": false,
                "required": ["text", "kind", "stable_identifier_kind", "evidence", "proof_evidence"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 512},
                    "kind": {"enum": ["explicit_alias", "stable_identifier"]},
                    "stable_identifier_kind": {"type": ["string", "null"], "maxLength": 32},
                    "evidence": {"$ref": "#/$defs/quote"},
                    "proof_evidence": {"$ref": "#/$defs/quote"}
                }
            },
            "entity": {
                "type": "object", "additionalProperties": false,
                "required": ["local_id", "name", "kind", "resolution", "disambiguation", "basis", "existing_entity_id", "name_evidence", "resolution_evidence", "aliases"],
                "properties": {
                    "local_id": {"type": "string", "pattern": "^local_[A-Za-z0-9_-]{1,58}$", "maxLength": 64},
                    "name": {"type": "string", "minLength": 1, "maxLength": 512},
                    "kind": {"enum": ["person", "organization", "location", "object", "concept", "unknown"]},
                    "resolution": {"enum": ["self", "new", "existing"]},
                    "disambiguation": {"enum": ["resolved", "pending"]},
                    "basis": {"enum": ["self_pronoun", "first_mention", "explicit_alias", "stable_identifier", "ambiguous"]},
                    "existing_entity_id": {"type": ["string", "null"], "maxLength": 128},
                    "name_evidence": {"$ref": "#/$defs/quote"},
                    "resolution_evidence": {"anyOf": [{"$ref": "#/$defs/quote"}, {"type": "null"}]},
                    "aliases": {"type": "array", "maxItems": 16, "items": {"$ref": "#/$defs/alias"}}
                }
            },
            "claim_object": {
                "type": "object", "additionalProperties": false,
                "required": ["kind", "text", "entity_ref", "span"],
                "properties": {
                    "kind": {"enum": ["text", "entity"]},
                    "text": nullable_string(),
                    "entity_ref": {"type": ["string", "null"], "maxLength": 128},
                    "span": {"anyOf": [{"$ref": "#/$defs/quote"}, {"type": "null"}]}
                }
            },
            "claim_evidence": {
                "type": "object", "additionalProperties": false,
                "required": ["kind", "quote"],
                "properties": {
                    "kind": {"enum": ["assertion", "user_confirmation", "correction", "temporal"]},
                    "quote": {"$ref": "#/$defs/quote"}
                }
            },
            "claim": {
                "type": "object", "additionalProperties": false,
                "required": ["local_id", "subject_ref", "predicate_key", "object", "polarity", "cardinality", "certainty", "disposition", "replaces_claim_ids", "conflicts_with_claim_ids", "event_time", "valid_from", "valid_to", "evidence"],
                "properties": {
                    "local_id": {"type": "string", "pattern": "^local_[A-Za-z0-9_-]{1,58}$", "maxLength": 64},
                    "subject_ref": {"type": "string", "minLength": 1, "maxLength": 128},
                    "predicate_key": {"type": "string", "pattern": "^[a-z][a-z0-9_.-]{0,63}$"},
                    "object": {"$ref": "#/$defs/claim_object"},
                    "polarity": {"enum": ["assert", "deny"]},
                    "cardinality": {"enum": ["single", "multi"]},
                    "certainty": {"enum": ["certain", "uncertain"]},
                    "disposition": {"enum": ["new", "confirm", "correct", "replace"]},
                    "replaces_claim_ids": {"type": "array", "maxItems": 128, "uniqueItems": true, "items": {"type": "string", "minLength": 1, "maxLength": 128}},
                    "conflicts_with_claim_ids": {"type": "array", "maxItems": 128, "uniqueItems": true, "items": {"type": "string", "minLength": 1, "maxLength": 128}},
                    "event_time": {"type": ["string", "null"], "maxLength": 64},
                    "valid_from": {"type": ["string", "null"], "maxLength": 64},
                    "valid_to": {"type": ["string", "null"], "maxLength": 64},
                    "evidence": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"$ref": "#/$defs/claim_evidence"}}
                }
            },
            "boundary": {
                "type": "object", "additionalProperties": false,
                "required": ["before_event_id", "reason", "evidence"],
                "properties": {
                    "before_event_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "reason": {"enum": ["explicit_topic_transition", "model_topic_shift"]},
                    "evidence": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"$ref": "#/$defs/quote"}}
                }
            }
        }
    })
}

impl RetrievalStore {
    pub fn next_consolidation_batch(
        &self,
        session_id: &str,
    ) -> RetrievalResult<Option<ConsolidationInputBatch>> {
        let source_events = self.replay_session(session_id)?;
        let watermark = self.consolidation_watermark(session_id)?;
        let start_index = validated_resume_index(&source_events, &watermark)?;

        let mut events = Vec::new();
        let mut turn_count = 0_usize;
        let mut char_count = 0_usize;
        let mut cursor = start_index;

        while cursor < source_events.len() {
            let user = &source_events[cursor];
            if user.role == EventRole::System {
                cursor += 1;
                continue;
            }
            if user.role != EventRole::User {
                return Err(RetrievalError::CorruptIndex(format!(
                    "巩固批次中的轮次未从用户事件开始：{}",
                    user.id
                )));
            }
            let turn_id = user.turn_id.as_deref().ok_or_else(|| {
                RetrievalError::CorruptIndex(format!("用户事件 {} 缺少轮次 ID", user.id))
            })?;
            let status = user.turn_status.ok_or_else(|| {
                RetrievalError::CorruptIndex(format!("用户事件 {} 缺少轮次状态", user.id))
            })?;

            let mut turn_end = cursor + 1;
            while turn_end < source_events.len()
                && source_events[turn_end].turn_id.as_deref() == Some(turn_id)
            {
                if source_events[turn_end].role != EventRole::Assistant {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "轮次 {turn_id} 包含非法事件顺序"
                    )));
                }
                turn_end += 1;
            }
            if turn_end - cursor > 2 {
                return Err(RetrievalError::CorruptIndex(format!(
                    "轮次 {turn_id} 包含多个助手事件"
                )));
            }

            if status == TurnStatus::Pending {
                break;
            }

            let turn_char_count = source_events[cursor..turn_end]
                .iter()
                .try_fold(0_usize, |total, event| {
                    total.checked_add(event.content.chars().count())
                })
                .ok_or_else(|| RetrievalError::CorruptIndex("巩固批次字符数溢出".into()))?;
            let combined_char_count = char_count
                .checked_add(turn_char_count)
                .ok_or_else(|| RetrievalError::CorruptIndex("巩固批次字符数溢出".into()))?;

            if turn_count > 0
                && (turn_count == CONSOLIDATION_MAX_TURNS
                    || combined_char_count > CONSOLIDATION_MAX_CHARS)
            {
                break;
            }

            events.extend(
                source_events[cursor..turn_end]
                    .iter()
                    .map(consolidation_event),
            );
            turn_count += 1;
            char_count = combined_char_count;
            cursor = turn_end;

            if turn_count == CONSOLIDATION_MAX_TURNS
                || (turn_count == 1 && char_count > CONSOLIDATION_MAX_CHARS)
            {
                break;
            }
        }

        let Some(first) = events.first() else {
            return Ok(None);
        };
        let last = events
            .last()
            .expect("a non-empty consolidation batch has a final event");
        let batch_key = consolidation_batch_key(
            session_id,
            watermark.through_sequence,
            last.sequence,
            &events,
        );
        Ok(Some(ConsolidationInputBatch {
            batch_key,
            session_id: session_id.to_owned(),
            watermark_before: watermark.through_sequence,
            from_sequence: first.sequence,
            through_sequence: last.sequence,
            through_event_id: last.event_id.clone(),
            through_event_sha256: last.content_sha256.clone(),
            turn_count,
            char_count,
            events,
        }))
    }

    pub fn consolidation_watermark(
        &self,
        session_id: &str,
    ) -> RetrievalResult<ConsolidationWatermark> {
        let connection = self.open_connection()?;
        let stored = connection
            .query_row(
                "SELECT through_sequence, through_event_id, through_event_sha256, updated_at
                 FROM consolidation_watermarks WHERE session_id = ?1",
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
            .map_err(|error| self.database_error(error))?;
        let Some((through_sequence, through_event_id, through_event_sha256, updated_at)) = stored
        else {
            return Ok(ConsolidationWatermark {
                session_id: session_id.to_owned(),
                through_sequence: 0,
                through_event_id: None,
                through_event_sha256: None,
                updated_at: None,
            });
        };
        let through_sequence = nonnegative_usize(through_sequence, "watermark.through_sequence")?;
        if through_sequence == 0 {
            return Err(RetrievalError::CorruptIndex(
                "巩固水位零值必须由缺失记录表示".into(),
            ));
        }
        if through_event_id.is_none() != through_event_sha256.is_none() {
            return Err(RetrievalError::CorruptIndex(
                "巩固水位包含不完整的事件来源".into(),
            ));
        }
        Ok(ConsolidationWatermark {
            session_id: session_id.to_owned(),
            through_sequence,
            through_event_id,
            through_event_sha256,
            updated_at,
        })
    }

    pub fn consolidation_candidates(
        &self,
        entity_limit: usize,
        claim_limit: usize,
    ) -> RetrievalResult<ConsolidationCandidateSnapshot> {
        validate_candidate_limit(entity_limit, "entity_limit")?;
        validate_candidate_limit(claim_limit, "claim_limit")?;
        let connection = self.open_connection()?;
        validate_global_stable_identifier_integrity(&connection)?;
        let snapshot = load_candidate_snapshot(&connection, entity_limit, claim_limit)?;
        self.verify_snapshot_source_freshness(&snapshot)?;
        Ok(snapshot)
    }

    pub fn apply_consolidation_attempt(
        &self,
        batch: &ConsolidationInputBatch,
        candidates: &ConsolidationCandidateSnapshot,
        attempt: &ConsolidationAttemptRecord,
    ) -> ConsolidationApplyResult<ConsolidationApplyReport> {
        validate_applied_attempt(batch, attempt)?;
        let supplied_hash = candidate_snapshot_hash(&candidates.entities, &candidates.claims)
            .map_err(ConsolidationApplyError::Retrieval)?;
        if supplied_hash != candidates.snapshot_sha256 {
            return Err(stale("候选快照哈希与候选内容不一致"));
        }
        validate_candidate_snapshot(candidates).map_err(ConsolidationApplyError::Retrieval)?;

        let response = attempt.response_json.as_deref().ok_or_else(|| {
            rejected(
                "missing_response",
                "response_json",
                "applied 尝试必须包含响应 JSON",
            )
        })?;
        let output: StructuredConsolidationOutput =
            serde_json::from_str(response).map_err(|e| {
                rejected(
                    "invalid_response_json",
                    "response_json",
                    format!("结构化响应无法按严格契约解析：{e}"),
                )
            })?;
        let plan = validate_structured_output(batch, candidates, &output)?;

        let preflight_connection = self.open_connection()?;
        validate_global_stable_aliases(&preflight_connection, &plan)?;
        validate_plan_against_global_claims(&preflight_connection, &plan)?;
        let global_state_sha256 = global_memory_state_hash(&preflight_connection)?;
        drop(preflight_connection);
        self.verify_snapshot_source_freshness(candidates)
            .map_err(map_source_staleness)?;

        let pending = self
            .next_consolidation_batch(&batch.session_id)
            .map_err(map_source_staleness)?;
        if pending.as_ref() != Some(batch) {
            return Err(stale("当前待巩固批次已变化"));
        }

        let entity_limit = candidates.entities.len().max(1);
        let claim_limit = candidates.claims.len().max(1);
        let mut connection = self.open_connection()?;
        let transaction = connection
            .transaction()
            .map_err(|e| self.database_error(e))?;

        verify_batch_rows(&transaction, batch).map_err(ConsolidationApplyError::Retrieval)?;
        verify_watermark_before(&transaction, batch)?;
        let current = load_candidate_snapshot(&transaction, entity_limit, claim_limit)
            .map_err(ConsolidationApplyError::Retrieval)?;
        if current != *candidates {
            return Err(stale("候选实体或声明已在巩固期间变化"));
        }
        if global_memory_state_hash(&transaction).map_err(ConsolidationApplyError::Retrieval)?
            != global_state_sha256
        {
            return Err(stale("全局实体标识或声明状态已在巩固期间变化"));
        }

        let mut report = ConsolidationApplyReport {
            session_id: batch.session_id.clone(),
            batch_key: batch.batch_key.clone(),
            watermark_before: batch.watermark_before,
            watermark_after: batch.through_sequence,
            entities_created: 0,
            entities_reused: 0,
            aliases_created: 0,
            claims_created: 0,
            claims_confirmed: 0,
            claims_superseded: 0,
            claims_conflicted: 0,
            evidence_created: 0,
            boundaries_created: 0,
        };
        apply_validated_plan(&transaction, batch, attempt, &plan, &mut report)
            .map_err(|e| self.database_error(e))?;
        insert_attempt(&transaction, attempt).map_err(|e| self.database_error(e))?;
        compare_and_swap_watermark(&transaction, batch, &attempt.completed_at)?;
        transaction.commit().map_err(|e| self.database_error(e))?;
        Ok(report)
    }

    fn verify_snapshot_source_freshness(
        &self,
        snapshot: &ConsolidationCandidateSnapshot,
    ) -> RetrievalResult<()> {
        let mut expected_events = HashMap::<String, (String, usize, usize, String)>::new();
        for entity in &snapshot.entities {
            expected_events.insert(
                entity.created_event_id.clone(),
                (
                    entity.created_session_id.clone(),
                    entity.created_start,
                    entity.created_end,
                    entity.created_hash.clone(),
                ),
            );
            for alias in &entity.aliases {
                expected_events.insert(
                    alias.event_id.clone(),
                    (
                        alias.session_id.clone(),
                        alias.start_char,
                        alias.end_char,
                        alias.content_sha256.clone(),
                    ),
                );
            }
        }
        for claim in &snapshot.claims {
            for evidence in &claim.evidence {
                expected_events.insert(
                    evidence.event_id.clone(),
                    (
                        evidence.session_id.clone(),
                        evidence.start_char,
                        evidence.end_char,
                        evidence.content_sha256.clone(),
                    ),
                );
            }
        }
        for (event_id, (session_id, start, end, expected_hash)) in expected_events {
            let event = self.get_event(&event_id)?;
            if event.session_id != session_id {
                return Err(RetrievalError::CorruptIndex(format!(
                    "记忆来源事件 {event_id} 属于错误会话"
                )));
            }
            let text = slice_unicode(&event.content, start, end).ok_or_else(|| {
                RetrievalError::CorruptIndex(format!(
                    "记忆来源事件 {event_id} 的 Unicode 字符范围无效"
                ))
            })?;
            if sha256_bytes(text.as_bytes()) != expected_hash {
                return Err(RetrievalError::CorruptIndex(format!(
                    "记忆来源事件 {event_id} 的原始片段哈希不匹配"
                )));
            }
        }
        Ok(())
    }

    pub fn record_consolidation_failure(
        &self,
        record: &ConsolidationAttemptRecord,
    ) -> RetrievalResult<()> {
        if record.status == ConsolidationAttemptStatus::Applied {
            return Err(invalid_attempt(
                "record_consolidation_failure 不接受 applied 状态",
            ));
        }
        validate_attempt(record)?;
        let connection = self.open_connection()?;
        insert_attempt(&connection, record).map_err(|error| self.database_error(error))?;
        Ok(())
    }

    pub fn consolidation_attempts(
        &self,
        session_id: &str,
    ) -> RetrievalResult<Vec<ConsolidationAttemptRecord>> {
        let connection = self.open_connection()?;
        let mut statement = connection
            .prepare(
                "SELECT attempt_id, batch_key, session_id, from_sequence, through_sequence,
                        trigger, model, request_json, request_sha256, input_event_ids,
                        input_event_hashes, response_json, response_sha256, status, input_tokens,
                        output_tokens, latency_ms, started_at, completed_at, validation_json,
                        error_json
                 FROM consolidation_batches WHERE session_id = ?1
                 ORDER BY started_at, attempt_id",
            )
            .map_err(|error| self.database_error(error))?;
        let rows = statement
            .query_map([session_id], map_stored_attempt)
            .map_err(|error| self.database_error(error))?;
        let mut attempts = Vec::new();
        for row in rows {
            let stored = row.map_err(|error| self.database_error(error))?;
            attempts.push(decode_stored_attempt(stored)?);
        }
        Ok(attempts)
    }
}

fn consolidation_event(event: &StoredEvent) -> ConsolidationEvent {
    ConsolidationEvent {
        event_id: event.id.clone(),
        turn_id: event
            .turn_id
            .clone()
            .expect("system events are excluded before consolidation mapping"),
        sequence: event.sequence,
        role: event.role,
        created_at: event.created_at.clone(),
        content: event.content.clone(),
        content_sha256: event.content_sha256.clone(),
    }
}

fn validate_candidate_limit(value: usize, name: &str) -> RetrievalResult<()> {
    if !(1..=512).contains(&value) {
        return Err(RetrievalError::CorruptIndex(format!(
            "巩固候选参数 {name} 必须在 1..=512"
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct StoredEntityCandidate {
    entity_id: String,
    kind: String,
    canonical_name: String,
    normalized_name: String,
    disambiguation: String,
    created_session_id: String,
    created_batch_key: String,
    created_event_id: String,
    created_start: i64,
    created_end: i64,
    created_hash: String,
    created_at: String,
    updated_at: String,
}

#[derive(Debug)]
struct StoredAliasCandidate {
    alias_id: String,
    entity_id: String,
    text: String,
    normalized_text: String,
    kind: String,
    stable_identifier_kind: Option<String>,
    session_id: String,
    batch_key: String,
    event_id: String,
    start_char: i64,
    end_char: i64,
    content_sha256: String,
    created_at: String,
}

#[derive(Debug)]
struct StoredClaimCandidate {
    claim_id: String,
    session_id: String,
    subject_entity_id: String,
    predicate_key: String,
    object_kind: String,
    object_text: Option<String>,
    object_entity_id: Option<String>,
    normalized_object: String,
    polarity: String,
    cardinality: String,
    certainty: String,
    state: String,
    asserted_at: String,
    event_time: Option<String>,
    valid_from: String,
    valid_to: Option<String>,
    reference_time: String,
    created_batch_key: String,
    updated_batch_key: String,
    created_at: String,
    updated_at: String,
}

#[derive(Debug)]
struct StoredClaimEvidenceCandidate {
    evidence_id: String,
    session_id: String,
    batch_key: String,
    event_id: String,
    sequence: i64,
    role: String,
    kind: String,
    start_char: i64,
    end_char: i64,
    content_sha256: String,
    created_at: String,
}

fn load_candidate_snapshot(
    connection: &Connection,
    entity_limit: usize,
    claim_limit: usize,
) -> RetrievalResult<ConsolidationCandidateSnapshot> {
    validate_candidate_limit(entity_limit, "entity_limit")?;
    validate_candidate_limit(claim_limit, "claim_limit")?;
    let entity_limit = i64::try_from(entity_limit)
        .map_err(|_| RetrievalError::CorruptIndex("entity_limit 超出 SQLite INTEGER".into()))?;
    let claim_limit = i64::try_from(claim_limit)
        .map_err(|_| RetrievalError::CorruptIndex("claim_limit 超出 SQLite INTEGER".into()))?;

    let mut entity_statement = connection
        .prepare(
            "SELECT entity_id, kind, canonical_name, normalized_name, disambiguation,
                    created_session_id, created_batch_key, created_event_id, created_start,
                    created_end, created_hash, created_at, updated_at
             FROM memory_entities ORDER BY updated_at DESC, entity_id LIMIT ?1",
        )
        .map_err(candidate_database_error)?;
    let entity_rows = entity_statement
        .query_map([entity_limit], |row| {
            Ok(StoredEntityCandidate {
                entity_id: row.get(0)?,
                kind: row.get(1)?,
                canonical_name: row.get(2)?,
                normalized_name: row.get(3)?,
                disambiguation: row.get(4)?,
                created_session_id: row.get(5)?,
                created_batch_key: row.get(6)?,
                created_event_id: row.get(7)?,
                created_start: row.get(8)?,
                created_end: row.get(9)?,
                created_hash: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
            })
        })
        .map_err(candidate_database_error)?;
    let stored_entities = entity_rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(candidate_database_error)?;
    drop(entity_statement);

    let mut entities = Vec::with_capacity(stored_entities.len());
    for stored in stored_entities {
        let mut alias_statement = connection
            .prepare(
                "SELECT alias_id, entity_id, alias_text, normalized_alias, alias_kind,
                        stable_identifier_kind, session_id, batch_key, event_id, start_char,
                        end_char, content_sha256, created_at
                 FROM memory_entity_aliases WHERE entity_id = ?1 ORDER BY alias_id",
            )
            .map_err(candidate_database_error)?;
        let rows = alias_statement
            .query_map([&stored.entity_id], |row| {
                Ok(StoredAliasCandidate {
                    alias_id: row.get(0)?,
                    entity_id: row.get(1)?,
                    text: row.get(2)?,
                    normalized_text: row.get(3)?,
                    kind: row.get(4)?,
                    stable_identifier_kind: row.get(5)?,
                    session_id: row.get(6)?,
                    batch_key: row.get(7)?,
                    event_id: row.get(8)?,
                    start_char: row.get(9)?,
                    end_char: row.get(10)?,
                    content_sha256: row.get(11)?,
                    created_at: row.get(12)?,
                })
            })
            .map_err(candidate_database_error)?;
        let stored_aliases = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(candidate_database_error)?;
        drop(alias_statement);
        let aliases = stored_aliases
            .into_iter()
            .map(|alias| decode_alias_candidate(&stored.entity_id, alias))
            .collect::<RetrievalResult<Vec<_>>>()?;
        entities.push(MemoryEntityCandidate {
            entity_id: stored.entity_id,
            kind: parse_entity_kind(&stored.kind)?,
            canonical_name: stored.canonical_name,
            normalized_name: stored.normalized_name,
            disambiguation: parse_disambiguation(&stored.disambiguation)?,
            created_session_id: stored.created_session_id,
            created_batch_key: stored.created_batch_key,
            created_event_id: stored.created_event_id,
            created_start: nonnegative_usize(stored.created_start, "entity.created_start")?,
            created_end: nonnegative_usize(stored.created_end, "entity.created_end")?,
            created_hash: stored.created_hash,
            created_at: stored.created_at,
            updated_at: stored.updated_at,
            aliases,
        });
    }

    let mut claim_statement = connection
        .prepare(
            "SELECT claim_id, session_id, subject_entity_id, predicate_key, object_kind,
                    object_text, object_entity_id, normalized_object, polarity, cardinality,
                    certainty, state, asserted_at, event_time, valid_from, valid_to,
                    reference_time, created_batch_key, updated_batch_key, created_at, updated_at
             FROM memory_claims ORDER BY updated_at DESC, claim_id LIMIT ?1",
        )
        .map_err(candidate_database_error)?;
    let claim_rows = claim_statement
        .query_map([claim_limit], |row| {
            Ok(StoredClaimCandidate {
                claim_id: row.get(0)?,
                session_id: row.get(1)?,
                subject_entity_id: row.get(2)?,
                predicate_key: row.get(3)?,
                object_kind: row.get(4)?,
                object_text: row.get(5)?,
                object_entity_id: row.get(6)?,
                normalized_object: row.get(7)?,
                polarity: row.get(8)?,
                cardinality: row.get(9)?,
                certainty: row.get(10)?,
                state: row.get(11)?,
                asserted_at: row.get(12)?,
                event_time: row.get(13)?,
                valid_from: row.get(14)?,
                valid_to: row.get(15)?,
                reference_time: row.get(16)?,
                created_batch_key: row.get(17)?,
                updated_batch_key: row.get(18)?,
                created_at: row.get(19)?,
                updated_at: row.get(20)?,
            })
        })
        .map_err(candidate_database_error)?;
    let stored_claims = claim_rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(candidate_database_error)?;
    drop(claim_statement);
    let mut claims = Vec::with_capacity(stored_claims.len());
    for stored in stored_claims {
        let mut evidence_statement = connection
            .prepare(
                "SELECT evidence_id, session_id, batch_key, event_id, sequence, role, kind,
                        start_char, end_char, content_sha256, created_at
                 FROM memory_claim_evidence WHERE claim_id = ?1 ORDER BY evidence_id",
            )
            .map_err(candidate_database_error)?;
        let rows = evidence_statement
            .query_map([&stored.claim_id], |row| {
                Ok(StoredClaimEvidenceCandidate {
                    evidence_id: row.get(0)?,
                    session_id: row.get(1)?,
                    batch_key: row.get(2)?,
                    event_id: row.get(3)?,
                    sequence: row.get(4)?,
                    role: row.get(5)?,
                    kind: row.get(6)?,
                    start_char: row.get(7)?,
                    end_char: row.get(8)?,
                    content_sha256: row.get(9)?,
                    created_at: row.get(10)?,
                })
            })
            .map_err(candidate_database_error)?;
        let stored_evidence = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(candidate_database_error)?;
        drop(evidence_statement);
        let evidence = stored_evidence
            .into_iter()
            .map(decode_claim_evidence_candidate)
            .collect::<RetrievalResult<Vec<_>>>()?;
        claims.push(decode_claim_candidate(stored, evidence)?);
    }

    entities.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
    claims.sort_by(|left, right| left.claim_id.cmp(&right.claim_id));
    let snapshot_sha256 = candidate_snapshot_hash(&entities, &claims)?;
    let snapshot = ConsolidationCandidateSnapshot {
        entities,
        claims,
        snapshot_sha256,
    };
    validate_candidate_snapshot(&snapshot)?;
    validate_candidate_provenance(connection, &snapshot)?;
    Ok(snapshot)
}

fn candidate_database_error(source: rusqlite::Error) -> RetrievalError {
    RetrievalError::CorruptIndex(format!("无法读取巩固候选：{source}"))
}

fn global_stable_identifier_owners(
    connection: &Connection,
) -> RetrievalResult<HashMap<(String, String), HashSet<String>>> {
    let mut statement = connection
        .prepare(
            "SELECT stable_identifier_kind, normalized_alias, entity_id
             FROM memory_entity_aliases
             WHERE alias_kind = 'stable_identifier'
             ORDER BY stable_identifier_kind, normalized_alias, entity_id, alias_id",
        )
        .map_err(candidate_database_error)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(candidate_database_error)?;
    let mut owners = HashMap::<(String, String), HashSet<String>>::new();
    for row in rows {
        let (kind, value, entity_id) = row.map_err(candidate_database_error)?;
        if kind.trim().is_empty()
            || kind.chars().count() > 32
            || value.trim().is_empty()
            || entity_id.trim().is_empty()
        {
            return Err(RetrievalError::CorruptIndex(
                "全局稳定标识包含空白、超长或空实体字段".into(),
            ));
        }
        owners
            .entry((normalize_match(&kind), value))
            .or_default()
            .insert(entity_id);
    }
    Ok(owners)
}

fn validate_global_stable_identifier_integrity(connection: &Connection) -> RetrievalResult<()> {
    for ((kind, value), owners) in global_stable_identifier_owners(connection)? {
        if owners.len() > 1 {
            return Err(RetrievalError::CorruptIndex(format!(
                "稳定标识 ({kind}, {value}) 同时属于多个实体"
            )));
        }
    }
    Ok(())
}

fn validate_global_stable_aliases(
    connection: &Connection,
    plan: &ValidatedPlan,
) -> ConsolidationApplyResult<()> {
    validate_global_stable_identifier_integrity(connection)?;
    let owners = global_stable_identifier_owners(connection)?;
    let mut planned = HashMap::<(String, String), String>::new();
    for entity in &plan.entities {
        for alias in &entity.aliases {
            if alias.kind != MemoryAliasKind::StableIdentifier {
                continue;
            }
            let kind = alias
                .stable_identifier_kind
                .as_deref()
                .expect("validated stable identifier has a kind");
            let key = (normalize_match(kind), alias.normalized_text.clone());
            if let Some(existing_owners) = owners.get(&key)
                && (entity.create
                    || existing_owners.len() != 1
                    || !existing_owners.contains(&entity.entity_id))
            {
                return Err(rejected(
                    "stable_identifier_collision",
                    "entities.aliases",
                    "稳定标识已属于另一实体；不得创建或合并到不同实体",
                ));
            }
            if let Some(previous) = planned.insert(key, entity.entity_id.clone())
                && previous != entity.entity_id
            {
                return Err(rejected(
                    "ambiguous_stable_identifier",
                    "entities.aliases",
                    "同一响应将稳定标识分配给了多个实体",
                ));
            }
        }
    }
    Ok(())
}

fn global_memory_state_hash(connection: &Connection) -> RetrievalResult<String> {
    let mut hasher = Sha256::new();
    hash_length_delimited(&mut hasher, b"hippocampus-global-memory-state-v1");
    for (tag, query, columns) in [
        (
            "entities",
            "SELECT entity_id, kind, canonical_name, normalized_name, disambiguation,
                    created_session_id, created_batch_key, created_event_id, created_start,
                    created_end, created_hash, created_at, updated_at
             FROM memory_entities ORDER BY entity_id",
            13_usize,
        ),
        (
            "aliases",
            "SELECT alias_id, entity_id, alias_text, normalized_alias, alias_kind,
                    stable_identifier_kind, session_id, batch_key, event_id, start_char,
                    end_char, content_sha256, created_at
             FROM memory_entity_aliases ORDER BY alias_id",
            13_usize,
        ),
        (
            "claims",
            "SELECT claim_id, session_id, subject_entity_id, predicate_key, object_kind,
                    object_text, object_entity_id, normalized_object, polarity, cardinality,
                    certainty, state, asserted_at, event_time, valid_from, valid_to,
                    reference_time, created_batch_key, updated_batch_key, created_at, updated_at
             FROM memory_claims ORDER BY claim_id",
            21_usize,
        ),
        (
            "evidence",
            "SELECT evidence_id, claim_id, session_id, batch_key, event_id, sequence, role,
                    kind, start_char, end_char, content_sha256, created_at
             FROM memory_claim_evidence ORDER BY evidence_id",
            12_usize,
        ),
        (
            "transitions",
            "SELECT transition_id, claim_id, from_state, to_state, reason, related_claim_id,
                    session_id, batch_key, created_at
             FROM memory_claim_transitions ORDER BY transition_id",
            9_usize,
        ),
        (
            "boundaries",
            "SELECT boundary_id, session_id, batch_key, before_event_id, reason,
                    evidence_json, created_at
             FROM memory_boundary_suggestions ORDER BY boundary_id",
            7_usize,
        ),
    ] {
        hash_length_delimited(&mut hasher, tag.as_bytes());
        let mut statement = connection
            .prepare(query)
            .map_err(candidate_database_error)?;
        let mut rows = statement.query([]).map_err(candidate_database_error)?;
        while let Some(row) = rows.next().map_err(candidate_database_error)? {
            hash_length_delimited(&mut hasher, b"row");
            for index in 0..columns {
                let value = row.get_ref(index).map_err(candidate_database_error)?;
                match value {
                    ValueRef::Null => hash_length_delimited(&mut hasher, b"null"),
                    ValueRef::Integer(value) => {
                        hash_length_delimited(&mut hasher, b"integer");
                        hash_length_delimited(&mut hasher, &value.to_be_bytes());
                    }
                    ValueRef::Real(value) => {
                        hash_length_delimited(&mut hasher, b"real");
                        hash_length_delimited(&mut hasher, &value.to_bits().to_be_bytes());
                    }
                    ValueRef::Text(value) => {
                        hash_length_delimited(&mut hasher, b"text");
                        hash_length_delimited(&mut hasher, value);
                    }
                    ValueRef::Blob(value) => {
                        hash_length_delimited(&mut hasher, b"blob");
                        hash_length_delimited(&mut hasher, value);
                    }
                }
            }
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn decode_alias_candidate(
    expected_entity_id: &str,
    stored: StoredAliasCandidate,
) -> RetrievalResult<MemoryAliasCandidate> {
    if stored.entity_id != expected_entity_id {
        return Err(RetrievalError::CorruptIndex(
            "实体别名连接到了错误的实体".into(),
        ));
    }
    Ok(MemoryAliasCandidate {
        alias_id: stored.alias_id,
        text: stored.text,
        normalized_text: stored.normalized_text,
        kind: parse_alias_kind(&stored.kind)?,
        stable_identifier_kind: stored.stable_identifier_kind,
        session_id: stored.session_id,
        batch_key: stored.batch_key,
        event_id: stored.event_id,
        start_char: nonnegative_usize(stored.start_char, "alias.start_char")?,
        end_char: nonnegative_usize(stored.end_char, "alias.end_char")?,
        content_sha256: stored.content_sha256,
        created_at: stored.created_at,
    })
}

fn decode_claim_candidate(
    stored: StoredClaimCandidate,
    evidence: Vec<MemoryClaimEvidenceCandidate>,
) -> RetrievalResult<MemoryClaimCandidate> {
    Ok(MemoryClaimCandidate {
        claim_id: stored.claim_id,
        session_id: stored.session_id,
        subject_entity_id: stored.subject_entity_id,
        predicate_key: stored.predicate_key,
        object_kind: parse_object_kind(&stored.object_kind)?,
        object_text: stored.object_text,
        object_entity_id: stored.object_entity_id,
        normalized_object: stored.normalized_object,
        polarity: parse_polarity(&stored.polarity)?,
        cardinality: parse_cardinality(&stored.cardinality)?,
        certainty: parse_certainty(&stored.certainty)?,
        state: parse_claim_state(&stored.state)?,
        asserted_at: stored.asserted_at,
        event_time: stored.event_time,
        valid_from: stored.valid_from,
        valid_to: stored.valid_to,
        reference_time: stored.reference_time,
        created_batch_key: stored.created_batch_key,
        updated_batch_key: stored.updated_batch_key,
        created_at: stored.created_at,
        updated_at: stored.updated_at,
        evidence,
    })
}

fn decode_claim_evidence_candidate(
    stored: StoredClaimEvidenceCandidate,
) -> RetrievalResult<MemoryClaimEvidenceCandidate> {
    Ok(MemoryClaimEvidenceCandidate {
        evidence_id: stored.evidence_id,
        session_id: stored.session_id,
        batch_key: stored.batch_key,
        event_id: stored.event_id,
        sequence: nonnegative_usize(stored.sequence, "claim_evidence.sequence")?,
        role: match stored.role.as_str() {
            "user" => EventRole::User,
            "assistant" => EventRole::Assistant,
            value => {
                return Err(RetrievalError::CorruptIndex(format!(
                    "未知声明证据角色 {value}"
                )));
            }
        },
        kind: match stored.kind.as_str() {
            "assertion" => ConsolidationEvidenceKind::Assertion,
            "user_confirmation" => ConsolidationEvidenceKind::UserConfirmation,
            "correction" => ConsolidationEvidenceKind::Correction,
            "temporal" => ConsolidationEvidenceKind::Temporal,
            value => {
                return Err(RetrievalError::CorruptIndex(format!(
                    "未知声明证据类型 {value}"
                )));
            }
        },
        start_char: nonnegative_usize(stored.start_char, "claim_evidence.start_char")?,
        end_char: nonnegative_usize(stored.end_char, "claim_evidence.end_char")?,
        content_sha256: stored.content_sha256,
        created_at: stored.created_at,
    })
}

fn load_all_claim_candidates(
    connection: &Connection,
) -> RetrievalResult<Vec<MemoryClaimCandidate>> {
    let mut statement = connection
        .prepare(
            "SELECT claim_id, session_id, subject_entity_id, predicate_key, object_kind,
                    object_text, object_entity_id, normalized_object, polarity, cardinality,
                    certainty, state, asserted_at, event_time, valid_from, valid_to,
                    reference_time, created_batch_key, updated_batch_key, created_at, updated_at
             FROM memory_claims ORDER BY claim_id",
        )
        .map_err(candidate_database_error)?;
    let rows = statement
        .query_map([], |row| {
            Ok(StoredClaimCandidate {
                claim_id: row.get(0)?,
                session_id: row.get(1)?,
                subject_entity_id: row.get(2)?,
                predicate_key: row.get(3)?,
                object_kind: row.get(4)?,
                object_text: row.get(5)?,
                object_entity_id: row.get(6)?,
                normalized_object: row.get(7)?,
                polarity: row.get(8)?,
                cardinality: row.get(9)?,
                certainty: row.get(10)?,
                state: row.get(11)?,
                asserted_at: row.get(12)?,
                event_time: row.get(13)?,
                valid_from: row.get(14)?,
                valid_to: row.get(15)?,
                reference_time: row.get(16)?,
                created_batch_key: row.get(17)?,
                updated_batch_key: row.get(18)?,
                created_at: row.get(19)?,
                updated_at: row.get(20)?,
            })
        })
        .map_err(candidate_database_error)?;
    let stored = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(candidate_database_error)?;
    drop(statement);
    stored
        .into_iter()
        .map(|claim| decode_claim_candidate(claim, Vec::new()))
        .collect()
}

fn validate_plan_against_global_claims(
    connection: &Connection,
    plan: &ValidatedPlan,
) -> ConsolidationApplyResult<()> {
    let claims = load_all_claim_candidates(connection)?;
    for claim in &claims {
        if claim.claim_id.trim().is_empty()
            || claim.subject_entity_id.trim().is_empty()
            || !valid_predicate_key(&claim.predicate_key)
            || claim.normalized_object.trim().is_empty()
            || claim.created_batch_key.trim().is_empty()
            || claim.updated_batch_key.trim().is_empty()
        {
            return Err(ConsolidationApplyError::Retrieval(
                RetrievalError::CorruptIndex(format!("全局声明 {} 的结构字段损坏", claim.claim_id)),
            ));
        }
        let asserted = parse_stored_time(&claim.asserted_at, "global_claim.asserted_at")?;
        let reference = parse_stored_time(&claim.reference_time, "global_claim.reference_time")?;
        let valid_from = parse_stored_time(&claim.valid_from, "global_claim.valid_from")?;
        let valid_to = claim
            .valid_to
            .as_deref()
            .map(|value| parse_stored_time(value, "global_claim.valid_to"))
            .transpose()?;
        if asserted != reference || valid_to.is_some_and(|value| valid_from > value) {
            return Err(ConsolidationApplyError::Retrieval(
                RetrievalError::CorruptIndex(format!(
                    "全局声明 {} 的来源时间或有效区间损坏",
                    claim.claim_id
                )),
            ));
        }
        if let Some(event_time) = &claim.event_time {
            parse_stored_time(event_time, "global_claim.event_time")?;
        }
        match claim.object_kind {
            ConsolidationClaimObjectKind::Text => {
                let Some(text) = claim.object_text.as_deref() else {
                    return Err(ConsolidationApplyError::Retrieval(
                        RetrievalError::CorruptIndex(format!(
                            "全局文本声明 {} 缺少对象文本",
                            claim.claim_id
                        )),
                    ));
                };
                if claim.object_entity_id.is_some()
                    || normalize_match(text) != claim.normalized_object
                {
                    return Err(ConsolidationApplyError::Retrieval(
                        RetrievalError::CorruptIndex(format!(
                            "全局文本声明 {} 对象字段不一致",
                            claim.claim_id
                        )),
                    ));
                }
            }
            ConsolidationClaimObjectKind::Entity => {
                if claim.object_text.is_some()
                    || claim.object_entity_id.as_deref() != Some(&claim.normalized_object)
                {
                    return Err(ConsolidationApplyError::Retrieval(
                        RetrievalError::CorruptIndex(format!(
                            "全局实体声明 {} 对象字段不一致",
                            claim.claim_id
                        )),
                    ));
                }
            }
        }
    }

    for planned in &plan.claims {
        let mut exact = claims
            .iter()
            .filter(|claim| {
                claim.state.is_live()
                    && intervals_overlap(
                        &claim.valid_from,
                        claim.valid_to.as_deref(),
                        &planned.valid_from,
                        planned.valid_to.as_deref(),
                    )
                    && claim.subject_entity_id == planned.subject_entity_id
                    && claim.predicate_key == planned.predicate_key
                    && claim.object_kind == planned.object_kind
                    && claim.normalized_object == planned.normalized_object
                    && claim.polarity == planned.polarity
                    && claim.cardinality == planned.cardinality
            })
            .map(|claim| claim.claim_id.clone())
            .collect::<Vec<_>>();
        exact.sort();
        let mut contradictions = claims
            .iter()
            .filter(|claim| {
                claim.state.is_live()
                    && claim.subject_entity_id == planned.subject_entity_id
                    && claim.predicate_key == planned.predicate_key
                    && intervals_overlap(
                        &claim.valid_from,
                        claim.valid_to.as_deref(),
                        &planned.valid_from,
                        planned.valid_to.as_deref(),
                    )
                    && claim_contradicts(
                        claim,
                        planned.object_kind,
                        &planned.normalized_object,
                        planned.polarity,
                        planned.cardinality,
                    )
            })
            .map(|claim| claim.claim_id.clone())
            .collect::<Vec<_>>();
        contradictions.sort();
        contradictions.dedup();

        let consistent = match &planned.action {
            ValidatedClaimAction::Confirm { claim_id, .. } => {
                exact == [claim_id.clone()] && contradictions.is_empty()
            }
            ValidatedClaimAction::Create {
                conflicts,
                supersedes,
                ..
            } => {
                exact.is_empty()
                    && if supersedes.is_empty() {
                        contradictions == *conflicts
                    } else {
                        conflicts.is_empty() && contradictions == *supersedes
                    }
            }
        };
        if !consistent {
            return Err(rejected(
                "incomplete_candidate_state",
                "claims",
                "候选快照未覆盖当前数据库中的精确重复或全部确定性冲突",
            ));
        }
    }
    Ok(())
}

fn parse_entity_kind(value: &str) -> RetrievalResult<MemoryEntityKind> {
    match value {
        "person" => Ok(MemoryEntityKind::Person),
        "organization" => Ok(MemoryEntityKind::Organization),
        "location" => Ok(MemoryEntityKind::Location),
        "object" => Ok(MemoryEntityKind::Object),
        "concept" => Ok(MemoryEntityKind::Concept),
        "unknown" => Ok(MemoryEntityKind::Unknown),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知记忆实体类型 {value}"
        ))),
    }
}

fn parse_disambiguation(value: &str) -> RetrievalResult<EntityDisambiguation> {
    match value {
        "resolved" => Ok(EntityDisambiguation::Resolved),
        "pending" => Ok(EntityDisambiguation::Pending),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知实体消歧状态 {value}"
        ))),
    }
}

fn parse_alias_kind(value: &str) -> RetrievalResult<MemoryAliasKind> {
    match value {
        "explicit_alias" => Ok(MemoryAliasKind::ExplicitAlias),
        "stable_identifier" => Ok(MemoryAliasKind::StableIdentifier),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知实体别名类型 {value}"
        ))),
    }
}

fn parse_object_kind(value: &str) -> RetrievalResult<ConsolidationClaimObjectKind> {
    match value {
        "text" => Ok(ConsolidationClaimObjectKind::Text),
        "entity" => Ok(ConsolidationClaimObjectKind::Entity),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知声明对象类型 {value}"
        ))),
    }
}

fn parse_polarity(value: &str) -> RetrievalResult<ClaimPolarity> {
    match value {
        "assert" => Ok(ClaimPolarity::Assert),
        "deny" => Ok(ClaimPolarity::Deny),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知声明极性 {value}"
        ))),
    }
}

fn parse_cardinality(value: &str) -> RetrievalResult<ClaimCardinality> {
    match value {
        "single" => Ok(ClaimCardinality::Single),
        "multi" => Ok(ClaimCardinality::Multi),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知声明基数 {value}"
        ))),
    }
}

fn parse_certainty(value: &str) -> RetrievalResult<ClaimCertainty> {
    match value {
        "certain" => Ok(ClaimCertainty::Certain),
        "uncertain" => Ok(ClaimCertainty::Uncertain),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知声明确信度 {value}"
        ))),
    }
}

fn parse_claim_state(value: &str) -> RetrievalResult<MemoryClaimState> {
    match value {
        "active" => Ok(MemoryClaimState::Active),
        "superseded" => Ok(MemoryClaimState::Superseded),
        "conflicted" => Ok(MemoryClaimState::Conflicted),
        "uncertain" => Ok(MemoryClaimState::Uncertain),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "未知声明状态 {value}"
        ))),
    }
}

fn candidate_snapshot_hash(
    entities: &[MemoryEntityCandidate],
    claims: &[MemoryClaimCandidate],
) -> RetrievalResult<String> {
    #[derive(Serialize)]
    struct CandidateSnapshotHashInput<'a> {
        entities: &'a [MemoryEntityCandidate],
        claims: &'a [MemoryClaimCandidate],
    }
    let canonical_json = serde_json::to_vec(&CandidateSnapshotHashInput { entities, claims })
        .map_err(|e| RetrievalError::CorruptIndex(format!("无法编码巩固候选快照：{e}")))?;
    let mut hasher = Sha256::new();
    hash_length_delimited(&mut hasher, b"hippocampus-candidate-snapshot-v1");
    hash_length_delimited(&mut hasher, b"canonical-json");
    hash_length_delimited(&mut hasher, &canonical_json);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_candidate_snapshot(snapshot: &ConsolidationCandidateSnapshot) -> RetrievalResult<()> {
    if !is_lower_sha256(&snapshot.snapshot_sha256) {
        return Err(RetrievalError::CorruptIndex(
            "候选快照哈希不是小写 SHA-256".into(),
        ));
    }
    if snapshot
        .entities
        .windows(2)
        .any(|pair| pair[0].entity_id >= pair[1].entity_id)
        || snapshot
            .claims
            .windows(2)
            .any(|pair| pair[0].claim_id >= pair[1].claim_id)
    {
        return Err(RetrievalError::CorruptIndex(
            "巩固候选没有按稳定 ID 严格排序".into(),
        ));
    }
    let mut entity_ids = HashSet::new();
    for entity in &snapshot.entities {
        if entity.entity_id.trim().is_empty()
            || entity.canonical_name.trim().is_empty()
            || entity.created_session_id.trim().is_empty()
            || entity.created_batch_key.trim().is_empty()
            || entity.created_event_id.trim().is_empty()
            || entity.created_start >= entity.created_end
            || !is_lower_sha256(&entity.created_hash)
            || entity.normalized_name != normalize_match(&entity.canonical_name)
            || !entity_ids.insert(entity.entity_id.clone())
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "实体候选 {} 的结构或规范化字段损坏",
                entity.entity_id
            )));
        }
        parse_stored_time(&entity.created_at, "entity.created_at")?;
        parse_stored_time(&entity.updated_at, "entity.updated_at")?;
        if entity
            .aliases
            .windows(2)
            .any(|pair| pair[0].alias_id >= pair[1].alias_id)
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "实体 {} 的别名未严格排序",
                entity.entity_id
            )));
        }
        let mut alias_ids = HashSet::new();
        for alias in &entity.aliases {
            if alias.alias_id.trim().is_empty()
                || alias.text.trim().is_empty()
                || alias.normalized_text != normalize_match(&alias.text)
                || alias.session_id.trim().is_empty()
                || alias.batch_key.trim().is_empty()
                || alias.event_id.trim().is_empty()
                || alias.start_char >= alias.end_char
                || !is_lower_sha256(&alias.content_sha256)
                || !alias_ids.insert(alias.alias_id.clone())
            {
                return Err(RetrievalError::CorruptIndex(format!(
                    "别名候选 {} 的结构或规范化字段损坏",
                    alias.alias_id
                )));
            }
            match alias.kind {
                MemoryAliasKind::ExplicitAlias if alias.stable_identifier_kind.is_some() => {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "显式别名 {} 不应包含稳定标识类型",
                        alias.alias_id
                    )));
                }
                MemoryAliasKind::StableIdentifier => {
                    let _kind = alias
                        .stable_identifier_kind
                        .as_deref()
                        .filter(|value| !value.trim().is_empty() && value.chars().count() <= 32)
                        .ok_or_else(|| {
                            RetrievalError::CorruptIndex(format!(
                                "稳定标识 {} 缺少有效类型",
                                alias.alias_id
                            ))
                        })?;
                }
                MemoryAliasKind::ExplicitAlias => {}
            }
            parse_stored_time(&alias.created_at, "alias.created_at")?;
        }
    }
    for claim in &snapshot.claims {
        if claim.claim_id.trim().is_empty()
            || claim.session_id.trim().is_empty()
            || claim.subject_entity_id.trim().is_empty()
            || !valid_predicate_key(&claim.predicate_key)
            || claim.normalized_object.trim().is_empty()
            || claim.created_batch_key.trim().is_empty()
            || claim.updated_batch_key.trim().is_empty()
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "声明候选 {} 的必填字段损坏",
                claim.claim_id
            )));
        }
        match claim.object_kind {
            ConsolidationClaimObjectKind::Text => {
                let text = claim.object_text.as_deref().ok_or_else(|| {
                    RetrievalError::CorruptIndex(format!(
                        "文本声明 {} 缺少对象文本",
                        claim.claim_id
                    ))
                })?;
                if claim.object_entity_id.is_some()
                    || normalize_match(text) != claim.normalized_object
                {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "文本声明 {} 的对象字段不一致",
                        claim.claim_id
                    )));
                }
            }
            ConsolidationClaimObjectKind::Entity => {
                let entity_id = claim.object_entity_id.as_deref().ok_or_else(|| {
                    RetrievalError::CorruptIndex(format!(
                        "实体声明 {} 缺少对象实体",
                        claim.claim_id
                    ))
                })?;
                if claim.object_text.is_some() || claim.normalized_object != entity_id {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "实体声明 {} 的对象字段不一致",
                        claim.claim_id
                    )));
                }
            }
        }
        let asserted = parse_stored_time(&claim.asserted_at, "claim.asserted_at")?;
        let reference = parse_stored_time(&claim.reference_time, "claim.reference_time")?;
        if asserted != reference {
            return Err(RetrievalError::CorruptIndex(format!(
                "声明 {} 的 asserted_at 与 reference_time 不一致",
                claim.claim_id
            )));
        }
        let valid_from = parse_stored_time(&claim.valid_from, "claim.valid_from")?;
        let valid_to = claim
            .valid_to
            .as_deref()
            .map(|value| parse_stored_time(value, "claim.valid_to"))
            .transpose()?;
        if valid_to.is_some_and(|value| valid_from > value) {
            return Err(RetrievalError::CorruptIndex(format!(
                "声明 {} 的有效区间倒置",
                claim.claim_id
            )));
        }
        if let Some(value) = &claim.event_time {
            parse_stored_time(value, "claim.event_time")?;
        }
        if claim
            .evidence
            .windows(2)
            .any(|pair| pair[0].evidence_id >= pair[1].evidence_id)
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "声明 {} 的证据未按 ID 严格排序",
                claim.claim_id
            )));
        }
        let mut evidence_ids = HashSet::new();
        for evidence in &claim.evidence {
            if evidence.evidence_id.trim().is_empty()
                || evidence.session_id != claim.session_id
                || evidence.batch_key.trim().is_empty()
                || evidence.event_id.trim().is_empty()
                || evidence.start_char >= evidence.end_char
                || !is_lower_sha256(&evidence.content_sha256)
                || !evidence_ids.insert(evidence.evidence_id.clone())
                || matches!(
                    evidence.kind,
                    ConsolidationEvidenceKind::UserConfirmation
                        | ConsolidationEvidenceKind::Correction
                        | ConsolidationEvidenceKind::Temporal
                ) && evidence.role != EventRole::User
            {
                return Err(RetrievalError::CorruptIndex(format!(
                    "声明 {} 的证据 {} 结构损坏",
                    claim.claim_id, evidence.evidence_id
                )));
            }
            parse_stored_time(&evidence.created_at, "claim_evidence.created_at")?;
        }
    }
    if candidate_snapshot_hash(&snapshot.entities, &snapshot.claims)? != snapshot.snapshot_sha256 {
        return Err(RetrievalError::CorruptIndex(
            "候选快照内容哈希不匹配".into(),
        ));
    }
    Ok(())
}

fn validate_candidate_provenance(
    connection: &Connection,
    snapshot: &ConsolidationCandidateSnapshot,
) -> RetrievalResult<()> {
    for entity in &snapshot.entities {
        verify_stored_quote(
            connection,
            &entity.created_event_id,
            entity.created_start,
            entity.created_end,
            &entity.created_hash,
            Some(&entity.canonical_name),
            Some(&entity.created_session_id),
        )?;
        for alias in &entity.aliases {
            verify_stored_quote(
                connection,
                &alias.event_id,
                alias.start_char,
                alias.end_char,
                &alias.content_sha256,
                Some(&alias.text),
                Some(&alias.session_id),
            )?;
        }
    }
    for claim in &snapshot.claims {
        for evidence in &claim.evidence {
            verify_stored_quote(
                connection,
                &evidence.event_id,
                evidence.start_char,
                evidence.end_char,
                &evidence.content_sha256,
                None,
                Some(&evidence.session_id),
            )?;
            let stored = connection
                .query_row(
                    "SELECT sequence, role FROM events WHERE event_id = ?1",
                    [&evidence.event_id],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
                )
                .map_err(candidate_database_error)?;
            if nonnegative_usize(stored.0, "evidence.source_sequence")? != evidence.sequence
                || stored.1 != evidence.role.as_str()
            {
                return Err(RetrievalError::CorruptIndex(format!(
                    "声明证据 {} 的来源序号或角色不匹配",
                    evidence.evidence_id
                )));
            }
        }
    }
    Ok(())
}

fn verify_stored_quote(
    connection: &Connection,
    event_id: &str,
    start: usize,
    end: usize,
    expected_hash: &str,
    expected_text: Option<&str>,
    expected_session: Option<&str>,
) -> RetrievalResult<()> {
    let stored = connection
        .query_row(
            "SELECT session_id, content, content_sha256 FROM events WHERE event_id = ?1",
            [event_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(candidate_database_error)?
        .ok_or_else(|| RetrievalError::CorruptIndex(format!("记忆来源事件 {event_id} 不存在")))?;
    if expected_session.is_some_and(|expected| expected != stored.0)
        || sha256_bytes(stored.1.as_bytes()) != stored.2
    {
        return Err(RetrievalError::CorruptIndex(format!(
            "记忆来源事件 {event_id} 的会话或原文哈希不匹配"
        )));
    }
    let text = slice_unicode(&stored.1, start, end).ok_or_else(|| {
        RetrievalError::CorruptIndex(format!("记忆来源事件 {event_id} 的字符范围无效"))
    })?;
    if sha256_bytes(text.as_bytes()) != expected_hash
        || expected_text.is_some_and(|expected| expected != text)
    {
        return Err(RetrievalError::CorruptIndex(format!(
            "记忆来源事件 {event_id} 的精确片段不匹配"
        )));
    }
    Ok(())
}

fn parse_stored_time(value: &str, name: &str) -> RetrievalResult<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(value)
        .map_err(|_| RetrievalError::CorruptIndex(format!("{name} 不是 RFC3339 时间")))
}

#[derive(Debug, Clone)]
struct ValidatedQuote {
    quote: ConsolidationQuote,
    text: String,
    role: EventRole,
    sequence: usize,
    created_at: String,
}

#[derive(Debug)]
struct ValidatedAlias {
    alias_id: String,
    entity_id: String,
    text: String,
    normalized_text: String,
    kind: MemoryAliasKind,
    stable_identifier_kind: Option<String>,
    evidence: ValidatedQuote,
}

#[derive(Debug)]
struct ValidatedEntity {
    entity_id: String,
    kind: MemoryEntityKind,
    canonical_name: String,
    normalized_name: String,
    disambiguation: EntityDisambiguation,
    created_evidence: ValidatedQuote,
    create: bool,
    aliases: Vec<ValidatedAlias>,
}

#[derive(Debug, Clone)]
struct ResolvedEntityView {
    normalized_names: Vec<String>,
}

#[derive(Debug)]
struct ValidatedEvidence {
    evidence_id: String,
    kind: ConsolidationEvidenceKind,
    quote: ValidatedQuote,
}

#[derive(Debug)]
enum ValidatedClaimAction {
    Create {
        claim_id: String,
        state: MemoryClaimState,
        conflicts: Vec<String>,
        supersedes: Vec<String>,
        supersede_reason: Option<&'static str>,
    },
    Confirm {
        claim_id: String,
        previous_state: MemoryClaimState,
        final_state: MemoryClaimState,
        certainty_upgraded: bool,
    },
}

#[derive(Debug)]
struct ValidatedClaim {
    action: ValidatedClaimAction,
    subject_entity_id: String,
    predicate_key: String,
    object_kind: ConsolidationClaimObjectKind,
    object_text: Option<String>,
    object_entity_id: Option<String>,
    normalized_object: String,
    polarity: ClaimPolarity,
    cardinality: ClaimCardinality,
    certainty: ClaimCertainty,
    asserted_at: String,
    event_time: Option<String>,
    valid_from: String,
    valid_to: Option<String>,
    reference_time: String,
    evidence: Vec<ValidatedEvidence>,
}

#[derive(Debug)]
struct ValidatedBoundary {
    boundary_id: String,
    before_event_id: String,
    reason: BoundarySuggestionReason,
    evidence_json: String,
}

#[derive(Debug)]
struct ValidatedPlan {
    entities: Vec<ValidatedEntity>,
    claims: Vec<ValidatedClaim>,
    boundaries: Vec<ValidatedBoundary>,
}

type ValidatedEntitySet = (
    Vec<ValidatedEntity>,
    HashMap<String, String>,
    HashMap<String, ResolvedEntityView>,
);

type ValidatedClaimObjectFields = (Option<String>, Option<String>, String, Vec<String>);

fn rejected(
    code: impl Into<String>,
    path: impl Into<String>,
    message: impl Into<String>,
) -> ConsolidationApplyError {
    let code = code.into();
    let path = path.into();
    let message = message.into();
    let validation_json = serde_json::to_string(&json!({
        "code": code,
        "message": message,
        "path": path,
        "valid": false
    }))
    .expect("validation diagnostic contains only serializable strings");
    ConsolidationApplyError::Rejected {
        validation_json,
        message,
    }
}

fn stale(message: impl Into<String>) -> ConsolidationApplyError {
    ConsolidationApplyError::Stale {
        message: message.into(),
    }
}

fn map_source_staleness(error: RetrievalError) -> ConsolidationApplyError {
    match error {
        RetrievalError::StaleIndex { session_id } => stale(format!(
            "原始会话 {session_id} 已变化，候选或批次需要重新同步"
        )),
        error => ConsolidationApplyError::Retrieval(error),
    }
}

fn validate_applied_attempt(
    batch: &ConsolidationInputBatch,
    attempt: &ConsolidationAttemptRecord,
) -> ConsolidationApplyResult<()> {
    if attempt.status != ConsolidationAttemptStatus::Applied {
        return Err(rejected(
            "attempt_status",
            "status",
            "apply_consolidation_attempt 只接受 applied 状态",
        ));
    }
    validate_attempt(attempt).map_err(|error| {
        rejected(
            "attempt_contract",
            "attempt",
            format!("巩固尝试记录无效：{error}"),
        )
    })?;
    validate_batch_contract(batch)?;
    let expected_ids = batch
        .events
        .iter()
        .map(|event| event.event_id.clone())
        .collect::<Vec<_>>();
    let expected_hashes = batch
        .events
        .iter()
        .map(|event| event.content_sha256.clone())
        .collect::<Vec<_>>();
    if attempt.batch_key != batch.batch_key
        || attempt.session_id != batch.session_id
        || attempt.from_sequence != batch.from_sequence
        || attempt.through_sequence != batch.through_sequence
        || attempt.input_event_ids != expected_ids
        || attempt.input_event_hashes != expected_hashes
    {
        return Err(stale("尝试记录与精确输入批次不一致"));
    }
    if attempt.response_json.is_none() || attempt.response_sha256.is_none() {
        return Err(rejected(
            "missing_response",
            "response_json",
            "applied 尝试必须同时包含响应字节与哈希",
        ));
    }
    Ok(())
}

fn validate_batch_contract(batch: &ConsolidationInputBatch) -> ConsolidationApplyResult<()> {
    if batch.events.is_empty()
        || batch.session_id.trim().is_empty()
        || batch.batch_key.trim().is_empty()
        || batch.from_sequence > batch.through_sequence
    {
        return Err(stale("巩固输入批次结构无效"));
    }
    let first = &batch.events[0];
    let last = batch.events.last().expect("non-empty checked");
    if first.sequence != batch.from_sequence
        || last.sequence != batch.through_sequence
        || last.event_id != batch.through_event_id
        || last.content_sha256 != batch.through_event_sha256
        || batch.char_count
            != batch
                .events
                .iter()
                .map(|event| event.content.chars().count())
                .sum::<usize>()
        || batch.events.iter().any(|event| {
            event.role == EventRole::System
                || event.content_sha256 != sha256_bytes(event.content.as_bytes())
        })
        || batch
            .events
            .windows(2)
            .any(|pair| pair[0].sequence >= pair[1].sequence)
    {
        return Err(stale("巩固输入批次来源、序号或哈希不一致"));
    }
    let turn_count = batch
        .events
        .iter()
        .map(|event| event.turn_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    if turn_count != batch.turn_count {
        return Err(stale("巩固输入批次轮次数不一致"));
    }
    let recomputed = consolidation_batch_key(
        &batch.session_id,
        batch.watermark_before,
        batch.through_sequence,
        &batch.events,
    );
    if recomputed != batch.batch_key {
        return Err(stale("巩固输入批次键不匹配"));
    }
    Ok(())
}

fn validate_structured_output(
    batch: &ConsolidationInputBatch,
    candidates: &ConsolidationCandidateSnapshot,
    output: &StructuredConsolidationOutput,
) -> ConsolidationApplyResult<ValidatedPlan> {
    if output.entities.len() > 128 {
        return Err(rejected("max_items", "entities", "实体输出超过 128 项"));
    }
    if output.claims.len() > 256 {
        return Err(rejected("max_items", "claims", "声明输出超过 256 项"));
    }
    if output.boundaries.len() > 64 {
        return Err(rejected("max_items", "boundaries", "边界输出超过 64 项"));
    }

    let (entities, entity_refs, entity_views) =
        validate_entities(batch, candidates, &output.entities)?;
    let claims = validate_claims(
        batch,
        candidates,
        &entity_refs,
        &entity_views,
        &output.claims,
    )?;
    let boundaries = validate_boundaries(batch, &output.boundaries)?;
    Ok(ValidatedPlan {
        entities,
        claims,
        boundaries,
    })
}

fn validate_entities(
    batch: &ConsolidationInputBatch,
    candidates: &ConsolidationCandidateSnapshot,
    outputs: &[ConsolidatedEntityOutput],
) -> ConsolidationApplyResult<ValidatedEntitySet> {
    let candidate_by_id = candidates
        .entities
        .iter()
        .map(|entity| (entity.entity_id.as_str(), entity))
        .collect::<HashMap<_, _>>();
    let mut local_ids = HashSet::new();
    let mut local_name_counts = HashMap::<String, usize>::new();
    for (index, entity) in outputs.iter().enumerate() {
        validate_local_id(&entity.local_id, &format!("entities[{index}].local_id"))?;
        if !local_ids.insert(entity.local_id.clone()) {
            return Err(rejected(
                "duplicate_id",
                format!("entities[{index}].local_id"),
                "实体 local_id 重复",
            ));
        }
        validate_exact_text(&entity.name, 512, &format!("entities[{index}].name"))?;
        *local_name_counts
            .entry(normalize_match(&entity.name))
            .or_default() += 1;
    }

    let mut existing_names = HashSet::new();
    for entity in &candidates.entities {
        existing_names.insert(entity.normalized_name.clone());
        for alias in &entity.aliases {
            existing_names.insert(alias.normalized_text.clone());
        }
    }
    let stable_owners = stable_identifier_owners(&candidates.entities);
    let mut resolved_targets = HashSet::new();
    let mut plans = Vec::with_capacity(outputs.len());
    let mut refs = HashMap::new();
    let mut views = candidate_entity_views(&candidates.entities);

    for (index, output) in outputs.iter().enumerate() {
        let path = format!("entities[{index}]");
        if output.aliases.len() > 16 {
            return Err(rejected(
                "max_items",
                format!("{path}.aliases"),
                "单个实体别名超过 16 项",
            ));
        }
        let name_quote = validate_quote(
            batch,
            &output.name_evidence,
            &format!("{path}.name_evidence"),
        )?;
        if output.name != name_quote.text {
            return Err(rejected(
                "rewritten_text",
                format!("{path}.name"),
                "实体名称必须与精确引用片段逐字一致",
            ));
        }
        let normalized_name = normalize_match(&output.name);
        let name_collision = existing_names.contains(&normalized_name)
            || local_name_counts
                .get(&normalized_name)
                .copied()
                .unwrap_or(0)
                > 1;
        let pronoun_like = is_pronoun_like(&normalized_name);

        let (target_id, create, canonical_name, canonical_normalized, target_kind) =
            match output.resolution {
                EntityResolution::SelfEntity => {
                    validate_self_entity(output, &name_quote, &path)?;
                    if let Some(candidate) = candidate_by_id.get("ent_self") {
                        if !compatible_kind(output.kind, candidate.kind) {
                            return Err(rejected(
                                "entity_kind",
                                format!("{path}.kind"),
                                "self 实体类型与已有候选不兼容",
                            ));
                        }
                        (
                            "ent_self".to_owned(),
                            false,
                            candidate.canonical_name.clone(),
                            candidate.normalized_name.clone(),
                            candidate.kind,
                        )
                    } else {
                        (
                            "ent_self".to_owned(),
                            true,
                            output.name.clone(),
                            normalized_name.clone(),
                            output.kind,
                        )
                    }
                }
                EntityResolution::New => {
                    if output.existing_entity_id.is_some() || output.resolution_evidence.is_some() {
                        return Err(rejected(
                            "new_entity_contract",
                            path.clone(),
                            "new 实体不得携带 existing_entity_id 或 resolution_evidence",
                        ));
                    }
                    if name_quote.role != EventRole::User {
                        return Err(rejected(
                            "untrusted_entity_source",
                            format!("{path}.name_evidence"),
                            "非 self 实体名称必须来自用户原文",
                        ));
                    }
                    let must_pending = pronoun_like || name_collision;
                    if must_pending {
                        if output.basis != EntityResolutionBasis::Ambiguous
                            || output.disambiguation != EntityDisambiguation::Pending
                        {
                            return Err(rejected(
                                "ambiguous_entity",
                                path.clone(),
                                "代词或同名实体必须保持 ambiguous + pending",
                            ));
                        }
                    } else if output.basis != EntityResolutionBasis::FirstMention
                        || output.disambiguation != EntityDisambiguation::Resolved
                    {
                        return Err(rejected(
                            "new_entity_basis",
                            path.clone(),
                            "唯一首次出现的实体必须是 first_mention + resolved",
                        ));
                    }
                    let id = deterministic_id(
                        "ent",
                        &[
                            batch.batch_key.as_str(),
                            output.local_id.as_str(),
                            output.kind.as_str(),
                            output.name_evidence.event_id.as_str(),
                            &output.name_evidence.start_char.to_string(),
                            &output.name_evidence.end_char.to_string(),
                            output.name_evidence.content_sha256.as_str(),
                        ],
                    );
                    (
                        id,
                        true,
                        output.name.clone(),
                        normalized_name.clone(),
                        output.kind,
                    )
                }
                EntityResolution::Existing => {
                    if output.disambiguation != EntityDisambiguation::Resolved {
                        return Err(rejected(
                            "existing_entity_pending",
                            format!("{path}.disambiguation"),
                            "existing 实体必须已确定消歧",
                        ));
                    }
                    if name_quote.role != EventRole::User {
                        return Err(rejected(
                            "untrusted_entity_source",
                            format!("{path}.name_evidence"),
                            "非 self 实体名称必须来自用户原文",
                        ));
                    }
                    let existing_id = output.existing_entity_id.as_deref().ok_or_else(|| {
                        rejected(
                            "missing_existing_id",
                            format!("{path}.existing_entity_id"),
                            "existing 实体必须指向明确候选 ID",
                        )
                    })?;
                    let candidate = candidate_by_id.get(existing_id).ok_or_else(|| {
                        rejected(
                            "unknown_entity_reference",
                            format!("{path}.existing_entity_id"),
                            "existing_entity_id 不在候选快照中",
                        )
                    })?;
                    if !compatible_kind(output.kind, candidate.kind) {
                        return Err(rejected(
                            "entity_kind",
                            format!("{path}.kind"),
                            "待合并实体类型与候选不兼容",
                        ));
                    }
                    match output.basis {
                        EntityResolutionBasis::ExplicitAlias => {
                            validate_existing_explicit_alias(batch, output, candidate, &path)?
                        }
                        EntityResolutionBasis::StableIdentifier => validate_existing_stable_id(
                            batch,
                            output,
                            candidate,
                            &stable_owners,
                            &path,
                        )?,
                        _ => {
                            return Err(rejected(
                                "existing_entity_basis",
                                format!("{path}.basis"),
                                "existing 合并只允许 explicit_alias 或 stable_identifier",
                            ));
                        }
                    }
                    (
                        candidate.entity_id.clone(),
                        false,
                        candidate.canonical_name.clone(),
                        candidate.normalized_name.clone(),
                        candidate.kind,
                    )
                }
            };

        if !resolved_targets.insert(target_id.clone()) {
            return Err(rejected(
                "duplicate_reference",
                path.clone(),
                "同一输出中多个实体解析到了同一个全局实体",
            ));
        }

        let aliases = validate_entity_aliases(batch, output, &target_id, &canonical_name, &path)?;
        let mut normalized_names = vec![canonical_normalized.clone(), normalized_name.clone()];
        normalized_names.extend(aliases.iter().map(|alias| alias.normalized_text.clone()));
        normalized_names.sort();
        normalized_names.dedup();
        views
            .entry(target_id.clone())
            .and_modify(|view| {
                view.normalized_names.extend(normalized_names.clone());
                view.normalized_names.sort();
                view.normalized_names.dedup();
            })
            .or_insert(ResolvedEntityView { normalized_names });
        refs.insert(output.local_id.clone(), target_id.clone());
        plans.push(ValidatedEntity {
            entity_id: target_id,
            kind: target_kind,
            canonical_name,
            normalized_name: canonical_normalized,
            disambiguation: output.disambiguation,
            created_evidence: name_quote,
            create,
            aliases,
        });
    }
    Ok((plans, refs, views))
}

fn validate_self_entity(
    output: &ConsolidatedEntityOutput,
    name_quote: &ValidatedQuote,
    path: &str,
) -> ConsolidationApplyResult<()> {
    if output.basis != EntityResolutionBasis::SelfPronoun
        || output.disambiguation != EntityDisambiguation::Resolved
        || output.existing_entity_id.is_some()
        || output.resolution_evidence.is_some()
        || !output.aliases.is_empty()
    {
        return Err(rejected(
            "self_contract",
            path,
            "self 实体必须是 self_pronoun、resolved 且不带候选、解析证据或别名",
        ));
    }
    let normalized = normalize_match(&name_quote.text);
    let valid = match name_quote.role {
        EventRole::User => matches!(normalized.as_str(), "我" | "本人" | "i" | "me" | "my"),
        EventRole::Assistant => {
            matches!(normalized.as_str(), "你" | "您" | "you" | "your")
        }
        EventRole::System => false,
    };
    if !valid {
        return Err(rejected(
            "self_pronoun",
            format!("{path}.name_evidence"),
            "self 绑定的代词与事件角色不匹配",
        ));
    }
    Ok(())
}

fn validate_existing_explicit_alias(
    batch: &ConsolidationInputBatch,
    output: &ConsolidatedEntityOutput,
    candidate: &MemoryEntityCandidate,
    path: &str,
) -> ConsolidationApplyResult<()> {
    let resolution = output.resolution_evidence.as_ref().ok_or_else(|| {
        rejected(
            "missing_resolution_evidence",
            format!("{path}.resolution_evidence"),
            "显式别名合并必须提供完整证明片段",
        )
    })?;
    let resolution_quote =
        validate_quote(batch, resolution, &format!("{path}.resolution_evidence"))?;
    if resolution_quote.role != EventRole::User {
        return Err(rejected(
            "untrusted_resolution",
            format!("{path}.resolution_evidence"),
            "实体合并证明必须来自用户原文",
        ));
    }
    let output_name = normalize_match(&output.name);
    let candidate_names = candidate_normalized_names(candidate);
    let matching = output.aliases.iter().any(|alias| {
        alias.kind == MemoryAliasKind::ExplicitAlias
            && normalize_match(&alias.text) == output_name
            && alias.proof_evidence == *resolution
    });
    if !matching
        || !normalized_contains(&resolution_quote.text, &output_name)
        || !candidate_names
            .iter()
            .any(|name| normalized_contains(&resolution_quote.text, name))
        || !contains_alias_marker(&resolution_quote.text)
    {
        return Err(rejected(
            "explicit_alias_proof",
            format!("{path}.resolution_evidence"),
            "显式别名证明必须包含新名称、候选名称和确定性别名标记，并与别名 proof_evidence 相同",
        ));
    }
    Ok(())
}

fn validate_existing_stable_id(
    batch: &ConsolidationInputBatch,
    output: &ConsolidatedEntityOutput,
    candidate: &MemoryEntityCandidate,
    stable_owners: &HashMap<(String, String), HashSet<String>>,
    path: &str,
) -> ConsolidationApplyResult<()> {
    let resolution = output.resolution_evidence.as_ref().ok_or_else(|| {
        rejected(
            "missing_resolution_evidence",
            format!("{path}.resolution_evidence"),
            "稳定标识合并必须提供完整证明片段",
        )
    })?;
    let resolution_quote =
        validate_quote(batch, resolution, &format!("{path}.resolution_evidence"))?;
    if resolution_quote.role != EventRole::User {
        return Err(rejected(
            "untrusted_resolution",
            format!("{path}.resolution_evidence"),
            "稳定标识证明必须来自用户原文",
        ));
    }
    let candidate_names = candidate_normalized_names(candidate);
    let mut matched = false;
    for alias in &output.aliases {
        if alias.kind != MemoryAliasKind::StableIdentifier || alias.proof_evidence != *resolution {
            continue;
        }
        let Some(kind) = alias.stable_identifier_kind.as_deref() else {
            continue;
        };
        let key = (normalize_match(kind), normalize_match(&alias.text));
        let owners = stable_owners.get(&key).cloned().unwrap_or_default();
        if owners.len() > 1 {
            return Err(rejected(
                "ambiguous_stable_identifier",
                format!("{path}.aliases"),
                "稳定标识在多个当前实体间不唯一",
            ));
        }
        if owners.len() == 1
            && owners.contains(&candidate.entity_id)
            && normalized_contains(&resolution_quote.text, &key.1)
            && candidate_names
                .iter()
                .any(|name| normalized_contains(&resolution_quote.text, name))
        {
            matched = true;
            break;
        }
    }
    if !matched {
        return Err(rejected(
            "stable_identifier_proof",
            format!("{path}.aliases"),
            "稳定标识必须精确、唯一地匹配候选，并与 resolution_evidence 使用同一证明片段",
        ));
    }
    Ok(())
}

fn validate_entity_aliases(
    batch: &ConsolidationInputBatch,
    output: &ConsolidatedEntityOutput,
    entity_id: &str,
    canonical_name: &str,
    path: &str,
) -> ConsolidationApplyResult<Vec<ValidatedAlias>> {
    let mut aliases = Vec::with_capacity(output.aliases.len());
    let mut semantics = HashSet::new();
    let mut evidence_quotes = HashSet::new();
    for (index, alias) in output.aliases.iter().enumerate() {
        let alias_path = format!("{path}.aliases[{index}]");
        validate_exact_text(&alias.text, 512, &format!("{alias_path}.text"))?;
        let evidence = validate_quote(batch, &alias.evidence, &format!("{alias_path}.evidence"))?;
        let proof = validate_quote(
            batch,
            &alias.proof_evidence,
            &format!("{alias_path}.proof_evidence"),
        )?;
        if evidence.role != EventRole::User || proof.role != EventRole::User {
            return Err(rejected(
                "untrusted_alias",
                alias_path,
                "非 self 实体别名及证明必须来自用户原文",
            ));
        }
        if evidence.text != alias.text {
            return Err(rejected(
                "rewritten_text",
                format!("{alias_path}.text"),
                "别名文本必须与 alias-only 精确片段逐字一致",
            ));
        }
        if !evidence_quotes.insert(alias.evidence.clone()) {
            return Err(rejected(
                "duplicate_quote",
                format!("{alias_path}.evidence"),
                "同一实体别名列表包含重复精确片段",
            ));
        }
        let normalized = normalize_match(&alias.text);
        let entity_names = [
            normalize_match(&output.name),
            normalize_match(canonical_name),
        ];
        if !normalized_contains(&proof.text, &normalized)
            || !entity_names
                .iter()
                .any(|name| normalized_contains(&proof.text, name))
        {
            return Err(rejected(
                "alias_proof",
                format!("{alias_path}.proof_evidence"),
                "别名证明必须同时包含别名值和实体名称",
            ));
        }
        let stable_kind = match alias.kind {
            MemoryAliasKind::ExplicitAlias => {
                if alias.stable_identifier_kind.is_some() || !contains_alias_marker(&proof.text) {
                    return Err(rejected(
                        "explicit_alias_contract",
                        alias_path,
                        "显式别名不得含稳定标识类型，且证明必须含确定性别名标记",
                    ));
                }
                None
            }
            MemoryAliasKind::StableIdentifier => {
                let kind = alias
                    .stable_identifier_kind
                    .as_deref()
                    .filter(|kind| !kind.trim().is_empty() && kind.chars().count() <= 32)
                    .ok_or_else(|| {
                        rejected(
                            "stable_identifier_kind",
                            format!("{alias_path}.stable_identifier_kind"),
                            "稳定标识必须包含 1..=32 字符的标识类型",
                        )
                    })?;
                Some(kind.to_owned())
            }
        };
        let semantic = (
            alias.kind,
            normalize_match(stable_kind.as_deref().unwrap_or("")),
            normalized.clone(),
        );
        if !semantics.insert(semantic) {
            return Err(rejected(
                "duplicate_alias",
                alias_path,
                "同一实体包含重复别名或稳定标识",
            ));
        }
        let alias_id = deterministic_id(
            "alias",
            &[
                entity_id,
                alias.kind.as_str(),
                stable_kind.as_deref().unwrap_or(""),
                &normalized,
                &alias.evidence.event_id,
                &alias.evidence.start_char.to_string(),
                &alias.evidence.end_char.to_string(),
                &alias.evidence.content_sha256,
            ],
        );
        aliases.push(ValidatedAlias {
            alias_id,
            entity_id: entity_id.to_owned(),
            text: alias.text.clone(),
            normalized_text: normalized,
            kind: alias.kind,
            stable_identifier_kind: stable_kind,
            evidence,
        });
    }
    Ok(aliases)
}

fn stable_identifier_owners(
    candidates: &[MemoryEntityCandidate],
) -> HashMap<(String, String), HashSet<String>> {
    let mut owners = HashMap::<(String, String), HashSet<String>>::new();
    for entity in candidates {
        for alias in &entity.aliases {
            if alias.kind == MemoryAliasKind::StableIdentifier
                && let Some(kind) = &alias.stable_identifier_kind
            {
                owners
                    .entry((normalize_match(kind), alias.normalized_text.clone()))
                    .or_default()
                    .insert(entity.entity_id.clone());
            }
        }
    }
    owners
}

fn candidate_entity_views(
    candidates: &[MemoryEntityCandidate],
) -> HashMap<String, ResolvedEntityView> {
    candidates
        .iter()
        .map(|entity| {
            let mut names = vec![entity.normalized_name.clone()];
            names.extend(
                entity
                    .aliases
                    .iter()
                    .map(|alias| alias.normalized_text.clone()),
            );
            names.sort();
            names.dedup();
            (
                entity.entity_id.clone(),
                ResolvedEntityView {
                    normalized_names: names,
                },
            )
        })
        .collect()
}

fn candidate_normalized_names(candidate: &MemoryEntityCandidate) -> Vec<String> {
    let mut names = vec![candidate.normalized_name.clone()];
    names.extend(
        candidate
            .aliases
            .iter()
            .filter(|alias| alias.kind == MemoryAliasKind::ExplicitAlias)
            .map(|alias| alias.normalized_text.clone()),
    );
    names.sort();
    names.dedup();
    names
}

fn validate_claims(
    batch: &ConsolidationInputBatch,
    candidates: &ConsolidationCandidateSnapshot,
    entity_refs: &HashMap<String, String>,
    entity_views: &HashMap<String, ResolvedEntityView>,
    outputs: &[ConsolidatedClaimOutput],
) -> ConsolidationApplyResult<Vec<ValidatedClaim>> {
    let mut local_ids = HashSet::new();
    let mut semantic_keys = HashSet::new();
    let candidate_claims = candidates
        .claims
        .iter()
        .map(|claim| (claim.claim_id.as_str(), claim))
        .collect::<HashMap<_, _>>();
    let mut plans = Vec::with_capacity(outputs.len());

    for (index, output) in outputs.iter().enumerate() {
        let path = format!("claims[{index}]");
        validate_local_id(&output.local_id, &format!("{path}.local_id"))?;
        if !local_ids.insert(output.local_id.clone()) {
            return Err(rejected(
                "duplicate_id",
                format!("{path}.local_id"),
                "声明 local_id 重复",
            ));
        }
        if !valid_predicate_key(&output.predicate_key) {
            return Err(rejected(
                "predicate_key",
                format!("{path}.predicate_key"),
                "predicate_key 必须匹配 [a-z][a-z0-9_.-]{0,63}",
            ));
        }
        validate_reference_list(
            &output.replaces_claim_ids,
            128,
            &format!("{path}.replaces_claim_ids"),
        )?;
        validate_reference_list(
            &output.conflicts_with_claim_ids,
            128,
            &format!("{path}.conflicts_with_claim_ids"),
        )?;
        if output.evidence.is_empty() || output.evidence.len() > 16 {
            return Err(rejected(
                "evidence_bounds",
                format!("{path}.evidence"),
                "声明证据必须包含 1..=16 项",
            ));
        }

        let subject_entity_id = resolve_entity_reference(
            &output.subject_ref,
            entity_refs,
            entity_views,
            &format!("{path}.subject_ref"),
        )?;
        let (object_text, object_entity_id, normalized_object, confirmation_targets) =
            validate_claim_object(
                batch,
                &output.object,
                entity_refs,
                entity_views,
                &format!("{path}.object"),
            )?;
        let semantic_key = format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
            subject_entity_id,
            output.predicate_key,
            output.object.kind.as_str(),
            normalized_object,
            output.polarity.as_str(),
            output.cardinality.as_str(),
            output.certainty.as_str()
        );
        // Certainty is not part of semantic identity; strip the final tagged component.
        let semantic_identity = semantic_key
            .rsplit_once('\u{1f}')
            .map(|(identity, _)| identity.to_owned())
            .expect("tagged semantic key contains separators");
        if !semantic_keys.insert(semantic_identity) {
            return Err(rejected(
                "duplicate_claim",
                path.clone(),
                "同一响应包含重复声明语义；必须只输出一次",
            ));
        }

        let mut evidence = Vec::with_capacity(output.evidence.len());
        let mut evidence_keys = HashSet::new();
        for (evidence_index, item) in output.evidence.iter().enumerate() {
            let evidence_path = format!("{path}.evidence[{evidence_index}]");
            let quote = validate_quote(batch, &item.quote, &format!("{evidence_path}.quote"))?;
            if !evidence_keys.insert((item.kind, item.quote.clone())) {
                return Err(rejected(
                    "duplicate_quote",
                    evidence_path,
                    "声明包含重复的证据类型与精确片段",
                ));
            }
            if matches!(
                item.kind,
                ConsolidationEvidenceKind::UserConfirmation
                    | ConsolidationEvidenceKind::Correction
                    | ConsolidationEvidenceKind::Temporal
            ) && quote.role != EventRole::User
            {
                return Err(rejected(
                    "evidence_role",
                    evidence_path,
                    "确认、纠正和时态证据必须来自用户事件",
                ));
            }
            evidence.push((item.kind, quote));
        }

        let trusted = evidence
            .iter()
            .filter(|(kind, quote)| {
                quote.role == EventRole::User
                    && matches!(
                        kind,
                        ConsolidationEvidenceKind::Assertion
                            | ConsolidationEvidenceKind::UserConfirmation
                            | ConsolidationEvidenceKind::Correction
                    )
            })
            .collect::<Vec<_>>();
        if trusted.is_empty() {
            return Err(rejected(
                "assistant_only_claim",
                format!("{path}.evidence"),
                "声明至少需要一条用户断言、确认或纠正证据",
            ));
        }
        for (_, assertion) in evidence.iter().filter(|(kind, quote)| {
            *kind == ConsolidationEvidenceKind::Assertion && quote.role == EventRole::Assistant
        }) {
            let confirmed = evidence.iter().any(|(kind, confirmation)| {
                *kind == ConsolidationEvidenceKind::UserConfirmation
                    && confirmation.role == EventRole::User
                    && confirmation.sequence > assertion.sequence
                    && confirmation_targets
                        .iter()
                        .any(|target| normalized_contains(&confirmation.text, target))
            });
            if !confirmed {
                return Err(rejected(
                    "unconfirmed_assistant_assertion",
                    format!("{path}.evidence"),
                    "助手断言必须由更晚且明确包含对象的用户确认支持",
                ));
            }
        }

        let earliest = trusted
            .iter()
            .copied()
            .map(|(_, quote)| quote)
            .min_by(|left, right| {
                let left_time = DateTime::parse_from_rfc3339(&left.created_at);
                let right_time = DateTime::parse_from_rfc3339(&right.created_at);
                match (left_time, right_time) {
                    (Ok(left_time), Ok(right_time)) => left_time
                        .cmp(&right_time)
                        .then_with(|| left.sequence.cmp(&right.sequence))
                        .then_with(|| left.quote.event_id.cmp(&right.quote.event_id)),
                    _ => left
                        .sequence
                        .cmp(&right.sequence)
                        .then_with(|| left.quote.event_id.cmp(&right.quote.event_id)),
                }
            })
            .expect("trusted evidence is non-empty");
        DateTime::parse_from_rfc3339(&earliest.created_at).map_err(|_| {
            ConsolidationApplyError::Retrieval(RetrievalError::CorruptIndex(format!(
                "来源事件 {} 的 created_at 不是 RFC3339",
                earliest.quote.event_id
            )))
        })?;
        let asserted_at = earliest.created_at.clone();
        let reference_time = asserted_at.clone();

        let has_temporal_input =
            output.event_time.is_some() || output.valid_from.is_some() || output.valid_to.is_some();
        if has_temporal_input
            && !evidence.iter().any(|(kind, quote)| {
                *kind == ConsolidationEvidenceKind::Temporal && quote.role == EventRole::User
            })
        {
            return Err(rejected(
                "missing_temporal_evidence",
                path.clone(),
                "模型提供的时态字段必须有用户 Temporal 精确证据",
            ));
        }
        let event_time =
            validate_optional_rfc3339(output.event_time.as_deref(), &format!("{path}.event_time"))?;
        let valid_from = if let Some(value) = output.valid_from.as_deref() {
            validate_rfc3339(value, &format!("{path}.valid_from"))?;
            value.to_owned()
        } else if let Some(value) = event_time.as_deref() {
            value.to_owned()
        } else {
            asserted_at.clone()
        };
        let valid_to =
            validate_optional_rfc3339(output.valid_to.as_deref(), &format!("{path}.valid_to"))?;
        let valid_from_time = validate_rfc3339(&valid_from, &format!("{path}.valid_from"))?;
        let valid_to_time = valid_to
            .as_deref()
            .map(|value| validate_rfc3339(value, &format!("{path}.valid_to")))
            .transpose()?;
        if valid_to_time.is_some_and(|end| valid_from_time > end) {
            return Err(rejected(
                "invalid_interval",
                path.clone(),
                "valid_from 不得晚于 valid_to",
            ));
        }

        let exact = candidates
            .claims
            .iter()
            .filter(|claim| {
                claim.state.is_live()
                    && intervals_overlap(
                        &claim.valid_from,
                        claim.valid_to.as_deref(),
                        &valid_from,
                        valid_to.as_deref(),
                    )
                    && claim.subject_entity_id == subject_entity_id
                    && claim.predicate_key == output.predicate_key
                    && claim.object_kind == output.object.kind
                    && claim.normalized_object == normalized_object
                    && claim.polarity == output.polarity
                    && claim.cardinality == output.cardinality
            })
            .collect::<Vec<_>>();
        if exact.len() > 1 {
            return Err(ConsolidationApplyError::Retrieval(
                RetrievalError::CorruptIndex(format!(
                    "同一语义存在多个重叠的活跃声明：{}",
                    exact
                        .iter()
                        .map(|claim| claim.claim_id.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                )),
            ));
        }
        let mut contradictions = candidates
            .claims
            .iter()
            .filter(|claim| {
                claim.state.is_live()
                    && claim.subject_entity_id == subject_entity_id
                    && claim.predicate_key == output.predicate_key
                    && intervals_overlap(
                        &claim.valid_from,
                        claim.valid_to.as_deref(),
                        &valid_from,
                        valid_to.as_deref(),
                    )
                    && claim_contradicts(
                        claim,
                        output.object.kind,
                        &normalized_object,
                        output.polarity,
                        output.cardinality,
                    )
            })
            .map(|claim| claim.claim_id.clone())
            .collect::<Vec<_>>();
        contradictions.sort();
        contradictions.dedup();

        let first_trusted = trusted
            .iter()
            .copied()
            .map(|(_, quote)| quote)
            .min_by_key(|quote| (quote.sequence, quote.quote.start_char, quote.quote.end_char))
            .expect("trusted evidence is non-empty");
        let generated_claim_id = deterministic_id(
            "claim",
            &[
                batch.batch_key.as_str(),
                output.local_id.as_str(),
                subject_entity_id.as_str(),
                output.predicate_key.as_str(),
                output.object.kind.as_str(),
                normalized_object.as_str(),
                output.polarity.as_str(),
                output.cardinality.as_str(),
                first_trusted.quote.event_id.as_str(),
                &first_trusted.quote.start_char.to_string(),
                &first_trusted.quote.end_char.to_string(),
                first_trusted.quote.content_sha256.as_str(),
            ],
        );
        let action = if let Some(existing) = exact.first() {
            if output.disposition != ClaimDisposition::Confirm
                || !output.replaces_claim_ids.is_empty()
                || !output.conflicts_with_claim_ids.is_empty()
            {
                return Err(rejected(
                    "duplicate_requires_confirm",
                    path.clone(),
                    "精确重复声明必须使用 confirm，且不得携带冲突或替换列表",
                ));
            }
            let certainty_upgraded = existing.state == MemoryClaimState::Uncertain
                && output.certainty == ClaimCertainty::Certain;
            let final_state = if certainty_upgraded {
                MemoryClaimState::Active
            } else {
                existing.state
            };
            ValidatedClaimAction::Confirm {
                claim_id: existing.claim_id.clone(),
                previous_state: existing.state,
                final_state,
                certainty_upgraded,
            }
        } else {
            if output.disposition == ClaimDisposition::Confirm {
                return Err(rejected(
                    "confirm_without_duplicate",
                    format!("{path}.disposition"),
                    "不存在精确重复声明时不得使用 confirm",
                ));
            }
            match output.disposition {
                ClaimDisposition::New => {
                    if !output.replaces_claim_ids.is_empty()
                        || output.conflicts_with_claim_ids != contradictions
                    {
                        return Err(rejected(
                            "conflict_set",
                            path.clone(),
                            "new 声明必须精确列出全部确定性冲突且不得替换声明",
                        ));
                    }
                    let state = if contradictions.is_empty() {
                        match output.certainty {
                            ClaimCertainty::Certain => MemoryClaimState::Active,
                            ClaimCertainty::Uncertain => MemoryClaimState::Uncertain,
                        }
                    } else {
                        MemoryClaimState::Conflicted
                    };
                    ValidatedClaimAction::Create {
                        claim_id: generated_claim_id,
                        state,
                        conflicts: contradictions,
                        supersedes: Vec::new(),
                        supersede_reason: None,
                    }
                }
                ClaimDisposition::Correct | ClaimDisposition::Replace => {
                    if !evidence.iter().any(|(kind, quote)| {
                        *kind == ConsolidationEvidenceKind::Correction
                            && quote.role == EventRole::User
                    }) {
                        return Err(rejected(
                            "missing_correction_evidence",
                            format!("{path}.evidence"),
                            "correct/replace 必须有用户 Correction 精确证据",
                        ));
                    }
                    if !output.conflicts_with_claim_ids.is_empty()
                        || contradictions.is_empty()
                        || output.replaces_claim_ids != contradictions
                    {
                        return Err(rejected(
                            "replacement_set",
                            path.clone(),
                            "correct/replace 必须且只能替换全部当前重叠矛盾声明",
                        ));
                    }
                    for claim_id in &output.replaces_claim_ids {
                        let claim = candidate_claims.get(claim_id.as_str()).ok_or_else(|| {
                            rejected(
                                "unknown_claim_reference",
                                format!("{path}.replaces_claim_ids"),
                                "替换列表引用了候选快照外的声明",
                            )
                        })?;
                        if !claim.state.is_live()
                            || claim.subject_entity_id != subject_entity_id
                            || claim.predicate_key != output.predicate_key
                        {
                            return Err(rejected(
                                "invalid_replacement_reference",
                                format!("{path}.replaces_claim_ids"),
                                "替换列表只能引用同主语、同谓词的当前活跃矛盾声明",
                            ));
                        }
                    }
                    let state = match output.certainty {
                        ClaimCertainty::Certain => MemoryClaimState::Active,
                        ClaimCertainty::Uncertain => MemoryClaimState::Uncertain,
                    };
                    ValidatedClaimAction::Create {
                        claim_id: generated_claim_id,
                        state,
                        conflicts: Vec::new(),
                        supersedes: contradictions,
                        supersede_reason: Some(match output.disposition {
                            ClaimDisposition::Correct => "corrected",
                            ClaimDisposition::Replace => "replaced",
                            _ => unreachable!(),
                        }),
                    }
                }
                ClaimDisposition::Confirm => unreachable!(),
            }
        };

        let target_claim_id = match &action {
            ValidatedClaimAction::Create { claim_id, .. }
            | ValidatedClaimAction::Confirm { claim_id, .. } => claim_id,
        };
        let validated_evidence = evidence
            .into_iter()
            .map(|(kind, quote)| ValidatedEvidence {
                evidence_id: deterministic_id(
                    "evidence",
                    &[
                        target_claim_id,
                        kind.as_str(),
                        quote.quote.event_id.as_str(),
                        &quote.quote.start_char.to_string(),
                        &quote.quote.end_char.to_string(),
                        quote.quote.content_sha256.as_str(),
                    ],
                ),
                kind,
                quote,
            })
            .collect();
        plans.push(ValidatedClaim {
            action,
            subject_entity_id,
            predicate_key: output.predicate_key.clone(),
            object_kind: output.object.kind,
            object_text,
            object_entity_id,
            normalized_object,
            polarity: output.polarity,
            cardinality: output.cardinality,
            certainty: output.certainty,
            asserted_at,
            event_time,
            valid_from,
            valid_to,
            reference_time,
            evidence: validated_evidence,
        });
    }

    // A model response cannot safely reference deterministic IDs of another new local claim.
    // Reject contradictory new claims instead of silently inventing cross-output state.
    for left in 0..plans.len() {
        for right in (left + 1)..plans.len() {
            let left_claim = &plans[left];
            let right_claim = &plans[right];
            if left_claim.subject_entity_id == right_claim.subject_entity_id
                && left_claim.predicate_key == right_claim.predicate_key
                && intervals_overlap(
                    &left_claim.valid_from,
                    left_claim.valid_to.as_deref(),
                    &right_claim.valid_from,
                    right_claim.valid_to.as_deref(),
                )
                && planned_claims_contradict(left_claim, right_claim)
            {
                return Err(rejected(
                    "intra_response_conflict",
                    format!("claims[{left}],claims[{right}]"),
                    "同一响应中的新声明互相矛盾；请分批保留歧义而非自动交叉链接",
                ));
            }
        }
    }
    Ok(plans)
}

fn validate_claim_object(
    batch: &ConsolidationInputBatch,
    object: &ConsolidatedClaimObject,
    entity_refs: &HashMap<String, String>,
    entity_views: &HashMap<String, ResolvedEntityView>,
    path: &str,
) -> ConsolidationApplyResult<ValidatedClaimObjectFields> {
    match object.kind {
        ConsolidationClaimObjectKind::Text => {
            if object.entity_ref.is_some() {
                return Err(rejected(
                    "object_contract",
                    path,
                    "text 对象必须令 entity_ref 为 null",
                ));
            }
            let text = object
                .text
                .as_deref()
                .ok_or_else(|| rejected("object_contract", path, "text 对象必须包含 text"))?;
            validate_exact_text(text, 512, &format!("{path}.text"))?;
            let span = object
                .span
                .as_ref()
                .ok_or_else(|| rejected("object_contract", path, "text 对象必须包含精确 span"))?;
            let quote = validate_quote(batch, span, &format!("{path}.span"))?;
            if quote.text != text {
                return Err(rejected(
                    "rewritten_text",
                    format!("{path}.text"),
                    "文本对象必须与精确引用片段逐字一致",
                ));
            }
            let normalized = normalize_match(text);
            Ok((
                Some(text.to_owned()),
                None,
                normalized.clone(),
                vec![normalized],
            ))
        }
        ConsolidationClaimObjectKind::Entity => {
            if object.text.is_some() || object.span.is_some() {
                return Err(rejected(
                    "object_contract",
                    path,
                    "entity 对象必须令 text 和 span 为 null",
                ));
            }
            let reference = object.entity_ref.as_deref().ok_or_else(|| {
                rejected("object_contract", path, "entity 对象必须包含 entity_ref")
            })?;
            let entity_id = resolve_entity_reference(
                reference,
                entity_refs,
                entity_views,
                &format!("{path}.entity_ref"),
            )?;
            let view = entity_views.get(&entity_id).ok_or_else(|| {
                rejected(
                    "unknown_entity_reference",
                    format!("{path}.entity_ref"),
                    "对象实体不在声明可见候选中",
                )
            })?;
            Ok((
                None,
                Some(entity_id.clone()),
                entity_id,
                view.normalized_names.clone(),
            ))
        }
    }
}

fn resolve_entity_reference(
    reference: &str,
    local_refs: &HashMap<String, String>,
    views: &HashMap<String, ResolvedEntityView>,
    path: &str,
) -> ConsolidationApplyResult<String> {
    if let Some(entity_id) = local_refs.get(reference) {
        return Ok(entity_id.clone());
    }
    if views.contains_key(reference) {
        return Ok(reference.to_owned());
    }
    Err(rejected(
        "unknown_entity_reference",
        path,
        "实体引用既不是当前 local_id，也不在候选快照中",
    ))
}

fn claim_contradicts(
    existing: &MemoryClaimCandidate,
    new_object_kind: ConsolidationClaimObjectKind,
    new_object: &str,
    new_polarity: ClaimPolarity,
    new_cardinality: ClaimCardinality,
) -> bool {
    let same_object =
        existing.object_kind == new_object_kind && existing.normalized_object == new_object;
    if existing.cardinality == ClaimCardinality::Single
        || new_cardinality == ClaimCardinality::Single
    {
        !same_object || existing.polarity != new_polarity
    } else {
        same_object && existing.polarity != new_polarity
    }
}

fn planned_claims_contradict(left: &ValidatedClaim, right: &ValidatedClaim) -> bool {
    let same_object =
        left.object_kind == right.object_kind && left.normalized_object == right.normalized_object;
    if left.cardinality == ClaimCardinality::Single || right.cardinality == ClaimCardinality::Single
    {
        !same_object || left.polarity != right.polarity
    } else {
        same_object && left.polarity != right.polarity
    }
}

fn intervals_overlap(
    left_start: &str,
    left_end: Option<&str>,
    right_start: &str,
    right_end: Option<&str>,
) -> bool {
    let Ok(left_start) = DateTime::parse_from_rfc3339(left_start) else {
        return false;
    };
    let Ok(right_start) = DateTime::parse_from_rfc3339(right_start) else {
        return false;
    };
    let left_end = left_end.and_then(|value| DateTime::parse_from_rfc3339(value).ok());
    let right_end = right_end.and_then(|value| DateTime::parse_from_rfc3339(value).ok());
    left_end.is_none_or(|end| right_start <= end) && right_end.is_none_or(|end| left_start <= end)
}

fn validate_reference_list(
    values: &[String],
    max: usize,
    path: &str,
) -> ConsolidationApplyResult<()> {
    if values.len() > max
        || values
            .iter()
            .any(|value| value.trim().is_empty() || value.chars().count() > 128)
    {
        return Err(rejected(
            "reference_bounds",
            path,
            "引用列表包含空值、超长值或超过上限",
        ));
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(rejected(
            "reference_order",
            path,
            "声明引用列表必须去重并按 ID 严格升序排列",
        ));
    }
    Ok(())
}

fn validate_boundaries(
    batch: &ConsolidationInputBatch,
    outputs: &[ConsolidationBoundaryOutput],
) -> ConsolidationApplyResult<Vec<ValidatedBoundary>> {
    let events = batch
        .events
        .iter()
        .map(|event| (event.event_id.as_str(), event))
        .collect::<HashMap<_, _>>();
    let mut seen_ids = HashSet::new();
    let mut plans = Vec::new();
    for (index, output) in outputs.iter().enumerate() {
        let path = format!("boundaries[{index}]");
        let event = events.get(output.before_event_id.as_str()).ok_or_else(|| {
            rejected(
                "unknown_boundary_event",
                format!("{path}.before_event_id"),
                "边界只能引用当前批次事件",
            )
        })?;
        if event.role != EventRole::User {
            return Err(rejected(
                "boundary_role",
                format!("{path}.before_event_id"),
                "边界必须位于用户事件之前",
            ));
        }
        if output.evidence.is_empty() || output.evidence.len() > 8 {
            return Err(rejected(
                "boundary_evidence_bounds",
                format!("{path}.evidence"),
                "边界证据必须包含 1..=8 项",
            ));
        }
        let mut quotes = Vec::with_capacity(output.evidence.len());
        let mut quote_ids = HashSet::new();
        let mut has_user = false;
        for (quote_index, quote) in output.evidence.iter().enumerate() {
            let validated =
                validate_quote(batch, quote, &format!("{path}.evidence[{quote_index}]"))?;
            if !quote_ids.insert(quote.clone()) {
                return Err(rejected(
                    "duplicate_quote",
                    format!("{path}.evidence[{quote_index}]"),
                    "边界包含重复证据片段",
                ));
            }
            has_user |= validated.role == EventRole::User;
            quotes.push((validated.sequence, quote.clone()));
        }
        if !has_user {
            return Err(rejected(
                "boundary_trust",
                format!("{path}.evidence"),
                "边界至少需要一条用户精确证据",
            ));
        }
        quotes.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.start_char.cmp(&right.1.start_char))
                .then_with(|| left.1.end_char.cmp(&right.1.end_char))
                .then_with(|| left.1.event_id.cmp(&right.1.event_id))
        });
        let canonical_quotes = quotes
            .into_iter()
            .map(|(_, quote)| quote)
            .collect::<Vec<_>>();
        let evidence_json = serde_json::to_string(&canonical_quotes).map_err(|e| {
            ConsolidationApplyError::Retrieval(RetrievalError::CorruptIndex(format!(
                "无法编码边界证据：{e}"
            )))
        })?;
        let boundary_id = deterministic_id(
            "boundary",
            &[
                batch.session_id.as_str(),
                batch.batch_key.as_str(),
                output.before_event_id.as_str(),
                output.reason.as_str(),
                evidence_json.as_str(),
            ],
        );
        if seen_ids.insert(boundary_id.clone()) {
            plans.push(ValidatedBoundary {
                boundary_id,
                before_event_id: output.before_event_id.clone(),
                reason: output.reason,
                evidence_json,
            });
        }
    }
    Ok(plans)
}

fn validate_quote(
    batch: &ConsolidationInputBatch,
    quote: &ConsolidationQuote,
    path: &str,
) -> ConsolidationApplyResult<ValidatedQuote> {
    if quote.start_char >= quote.end_char || !is_lower_sha256(&quote.content_sha256) {
        return Err(rejected(
            "quote_bounds_or_hash",
            path,
            "引用必须具有非空 Unicode 字符范围及小写 SHA-256",
        ));
    }
    let event = batch
        .events
        .iter()
        .find(|event| event.event_id == quote.event_id)
        .ok_or_else(|| rejected("quote_event", path, "引用只能指向当前精确巩固批次中的事件"))?;
    let text =
        slice_unicode(&event.content, quote.start_char, quote.end_char).ok_or_else(|| {
            rejected(
                "quote_bounds",
                path,
                format!(
                    "Unicode 字符范围超出事件长度 {}",
                    event.content.chars().count()
                ),
            )
        })?;
    if text.trim().is_empty() {
        return Err(rejected("blank_quote", path, "引用片段不得仅包含空白"));
    }
    if sha256_bytes(text.as_bytes()) != quote.content_sha256 {
        return Err(rejected(
            "quote_hash",
            path,
            "引用哈希与 Unicode 精确片段的 UTF-8 字节不一致",
        ));
    }
    Ok(ValidatedQuote {
        quote: quote.clone(),
        text: text.to_owned(),
        role: event.role,
        sequence: event.sequence,
        created_at: event.created_at.clone(),
    })
}

fn validate_exact_text(value: &str, max: usize, path: &str) -> ConsolidationApplyResult<()> {
    let count = value.chars().count();
    if count == 0 || count > max || value.trim().is_empty() {
        return Err(rejected(
            "text_bounds",
            path,
            format!("文本必须为非空、非纯空白且不超过 {max} 个 Unicode 字符"),
        ));
    }
    Ok(())
}

fn validate_local_id(value: &str, path: &str) -> ConsolidationApplyResult<()> {
    let Some(suffix) = value.strip_prefix("local_") else {
        return Err(rejected("local_id", path, "local_id 必须以 local_ 开头"));
    };
    if value.len() > 64
        || suffix.is_empty()
        || !suffix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(rejected(
            "local_id",
            path,
            "local_id 后缀只能含 ASCII 字母、数字、下划线或连字符，且总长不超过 64",
        ));
    }
    Ok(())
}

fn valid_predicate_key(value: &str) -> bool {
    value.len() <= 64
        && value
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase())
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.' | b'-')
        })
}

fn validate_rfc3339(value: &str, path: &str) -> ConsolidationApplyResult<DateTime<FixedOffset>> {
    if value.chars().count() > 64 {
        return Err(rejected(
            "time_bounds",
            path,
            "时间字符串不得超过 64 个字符",
        ));
    }
    DateTime::parse_from_rfc3339(value)
        .map_err(|_| rejected("rfc3339", path, "时间必须是严格 RFC3339"))
}

fn validate_optional_rfc3339(
    value: Option<&str>,
    path: &str,
) -> ConsolidationApplyResult<Option<String>> {
    value
        .map(|value| {
            validate_rfc3339(value, path)?;
            Ok(value.to_owned())
        })
        .transpose()
}

fn normalize_match(value: &str) -> String {
    let lowered = value
        .nfkc()
        .flat_map(char::to_lowercase)
        .collect::<String>();
    lowered.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalized_contains(haystack: &str, needle: &str) -> bool {
    !needle.is_empty() && normalize_match(haystack).contains(needle)
}

fn contains_alias_marker(value: &str) -> bool {
    let normalized = normalize_match(value);
    [
        "也叫",
        "又叫",
        "昵称",
        "别名",
        "即",
        "简称",
        "aka",
        "also known as",
        "call me",
        "叫我",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
}

fn is_pronoun_like(value: &str) -> bool {
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
            | "此人"
            | "那个人"
            | "i"
            | "me"
            | "my"
            | "you"
            | "your"
            | "he"
            | "him"
            | "his"
            | "she"
            | "her"
            | "hers"
            | "they"
            | "them"
            | "their"
    )
}

fn compatible_kind(left: MemoryEntityKind, right: MemoryEntityKind) -> bool {
    left == right || left == MemoryEntityKind::Unknown || right == MemoryEntityKind::Unknown
}

fn deterministic_id(prefix: &str, parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hash_length_delimited(&mut hasher, b"hippocampus-derived-memory-id-v1");
    hash_length_delimited(&mut hasher, prefix.as_bytes());
    for part in parts {
        hash_length_delimited(&mut hasher, part.as_bytes());
    }
    format!("{prefix}_{:x}", hasher.finalize())
}

fn slice_unicode(value: &str, start: usize, end: usize) -> Option<&str> {
    if start >= end {
        return None;
    }
    let char_count = value.chars().count();
    if end > char_count {
        return None;
    }
    let start_byte = if start == char_count {
        value.len()
    } else {
        value.char_indices().nth(start)?.0
    };
    let end_byte = if end == char_count {
        value.len()
    } else {
        value.char_indices().nth(end)?.0
    };
    value.get(start_byte..end_byte)
}

fn verify_batch_rows(
    transaction: &Transaction<'_>,
    batch: &ConsolidationInputBatch,
) -> RetrievalResult<()> {
    for event in &batch.events {
        let stored = transaction
            .query_row(
                "SELECT session_id, sequence, role, created_at, content, content_sha256
                 FROM events WHERE event_id = ?1",
                [&event.event_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(candidate_database_error)?
            .ok_or_else(|| {
                RetrievalError::CorruptIndex(format!(
                    "巩固来源事件 {} 在事务中不存在",
                    event.event_id
                ))
            })?;
        let sequence = nonnegative_usize(stored.1, "event.sequence")?;
        if stored.0 != batch.session_id
            || sequence != event.sequence
            || stored.2 != event.role.as_str()
            || stored.3 != event.created_at
            || stored.4 != event.content
            || stored.5 != event.content_sha256
            || sha256_bytes(stored.4.as_bytes()) != stored.5
        {
            return Err(RetrievalError::CorruptIndex(format!(
                "巩固来源事件 {} 的原文、角色、序号或哈希已损坏",
                event.event_id
            )));
        }
    }
    Ok(())
}

fn verify_watermark_before(
    transaction: &Transaction<'_>,
    batch: &ConsolidationInputBatch,
) -> ConsolidationApplyResult<()> {
    let stored = transaction
        .query_row(
            "SELECT through_sequence, through_event_id, through_event_sha256
             FROM consolidation_watermarks WHERE session_id = ?1",
            [&batch.session_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .optional()
        .map_err(candidate_database_error)?;
    if batch.watermark_before == 0 {
        if stored.is_some() {
            return Err(stale("预期零水位但当前已有巩固水位"));
        }
        return Ok(());
    }
    let Some((sequence, event_id, event_hash)) = stored else {
        return Err(stale("预期已有巩固水位但当前记录缺失"));
    };
    let sequence = nonnegative_usize(sequence, "watermark.through_sequence")?;
    let event_id = event_id.ok_or_else(|| {
        ConsolidationApplyError::Retrieval(RetrievalError::CorruptIndex(
            "当前巩固水位缺少事件 ID".into(),
        ))
    })?;
    let event_hash = event_hash.ok_or_else(|| {
        ConsolidationApplyError::Retrieval(RetrievalError::CorruptIndex(
            "当前巩固水位缺少事件哈希".into(),
        ))
    })?;
    if sequence != batch.watermark_before {
        return Err(stale("当前巩固水位序号已变化"));
    }
    let source = transaction
        .query_row(
            "SELECT event_id, content_sha256 FROM events
             WHERE session_id = ?1 AND sequence = ?2",
            params![
                batch.session_id,
                i64::try_from(sequence).unwrap_or(i64::MAX)
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(candidate_database_error)?;
    if source.as_ref() != Some(&(event_id, event_hash)) {
        return Err(ConsolidationApplyError::Retrieval(
            RetrievalError::CorruptIndex("当前巩固水位的事件来源无法由原文索引验证".into()),
        ));
    }
    Ok(())
}

fn compare_and_swap_watermark(
    transaction: &Transaction<'_>,
    batch: &ConsolidationInputBatch,
    updated_at: &str,
) -> ConsolidationApplyResult<()> {
    if batch.watermark_before == 0 {
        let changed = transaction
            .execute(
                "INSERT INTO consolidation_watermarks
                 (session_id, through_sequence, through_event_id, through_event_sha256, updated_at)
                 SELECT ?1, ?2, ?3, ?4, ?5
                 WHERE NOT EXISTS (
                    SELECT 1 FROM consolidation_watermarks WHERE session_id = ?1
                 )",
                params![
                    batch.session_id,
                    i64::try_from(batch.through_sequence).map_err(|_| {
                        ConsolidationApplyError::Retrieval(RetrievalError::CorruptIndex(
                            "through_sequence 超出 SQLite INTEGER".into(),
                        ))
                    })?,
                    batch.through_event_id,
                    batch.through_event_sha256,
                    updated_at,
                ],
            )
            .map_err(candidate_database_error)?;
        if changed != 1 {
            return Err(stale("零水位 compare-and-swap 失败"));
        }
        return Ok(());
    }

    let old = transaction
        .query_row(
            "SELECT through_event_id, through_event_sha256
             FROM consolidation_watermarks
             WHERE session_id = ?1 AND through_sequence = ?2",
            params![
                batch.session_id,
                i64::try_from(batch.watermark_before).map_err(|_| {
                    ConsolidationApplyError::Retrieval(RetrievalError::CorruptIndex(
                        "watermark_before 超出 SQLite INTEGER".into(),
                    ))
                })?,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(candidate_database_error)?
        .ok_or_else(|| stale("旧水位 compare-and-swap 基线缺失"))?;
    let changed = transaction
        .execute(
            "UPDATE consolidation_watermarks
             SET through_sequence = ?1, through_event_id = ?2,
                 through_event_sha256 = ?3, updated_at = ?4
             WHERE session_id = ?5 AND through_sequence = ?6
               AND through_event_id = ?7 AND through_event_sha256 = ?8",
            params![
                i64::try_from(batch.through_sequence).map_err(|_| {
                    ConsolidationApplyError::Retrieval(RetrievalError::CorruptIndex(
                        "through_sequence 超出 SQLite INTEGER".into(),
                    ))
                })?,
                batch.through_event_id,
                batch.through_event_sha256,
                updated_at,
                batch.session_id,
                i64::try_from(batch.watermark_before).map_err(|_| {
                    ConsolidationApplyError::Retrieval(RetrievalError::CorruptIndex(
                        "watermark_before 超出 SQLite INTEGER".into(),
                    ))
                })?,
                old.0,
                old.1,
            ],
        )
        .map_err(candidate_database_error)?;
    if changed != 1 {
        return Err(stale("非零水位 compare-and-swap 失败"));
    }
    Ok(())
}

fn apply_validated_plan(
    transaction: &Transaction<'_>,
    batch: &ConsolidationInputBatch,
    attempt: &ConsolidationAttemptRecord,
    plan: &ValidatedPlan,
    report: &mut ConsolidationApplyReport,
) -> rusqlite::Result<()> {
    for entity in &plan.entities {
        if entity.create {
            transaction.execute(
                "INSERT INTO memory_entities
                 (entity_id, kind, canonical_name, normalized_name, disambiguation,
                  created_session_id, created_batch_key, created_event_id, created_start,
                  created_end, created_hash, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12)",
                params![
                    entity.entity_id,
                    entity.kind.as_str(),
                    entity.canonical_name,
                    entity.normalized_name,
                    entity.disambiguation.as_str(),
                    batch.session_id,
                    batch.batch_key,
                    entity.created_evidence.quote.event_id,
                    i64::try_from(entity.created_evidence.quote.start_char)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                    i64::try_from(entity.created_evidence.quote.end_char)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                    entity.created_evidence.quote.content_sha256,
                    attempt.completed_at,
                ],
            )?;
            report.entities_created += 1;
        } else {
            let changed = transaction.execute(
                "UPDATE memory_entities SET disambiguation = ?1, updated_at = ?2
                 WHERE entity_id = ?3",
                params![
                    entity.disambiguation.as_str(),
                    attempt.completed_at,
                    entity.entity_id
                ],
            )?;
            if changed != 1 {
                return Err(rusqlite::Error::QueryReturnedNoRows);
            }
            report.entities_reused += 1;
        }
        for alias in &entity.aliases {
            transaction.execute(
                "INSERT INTO memory_entity_aliases
                 (alias_id, entity_id, alias_text, normalized_alias, alias_kind,
                  stable_identifier_kind, session_id, batch_key, event_id, start_char,
                  end_char, content_sha256, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    alias.alias_id,
                    alias.entity_id,
                    alias.text,
                    alias.normalized_text,
                    alias.kind.as_str(),
                    alias.stable_identifier_kind,
                    batch.session_id,
                    batch.batch_key,
                    alias.evidence.quote.event_id,
                    i64::try_from(alias.evidence.quote.start_char)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                    i64::try_from(alias.evidence.quote.end_char)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                    alias.evidence.quote.content_sha256,
                    attempt.completed_at,
                ],
            )?;
            report.aliases_created += 1;
        }
    }

    for claim in &plan.claims {
        let claim_id = match &claim.action {
            ValidatedClaimAction::Create {
                claim_id,
                state,
                conflicts,
                supersedes,
                supersede_reason,
            } => {
                transaction.execute(
                    "INSERT INTO memory_claims
                     (claim_id, session_id, subject_entity_id, predicate_key, object_kind,
                      object_text, object_entity_id, normalized_object, polarity, cardinality,
                      certainty, state, asserted_at, event_time, valid_from, valid_to,
                      reference_time, created_batch_key, updated_batch_key, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                             ?14, ?15, ?16, ?17, ?18, ?18, ?19, ?19)",
                    params![
                        claim_id,
                        batch.session_id,
                        claim.subject_entity_id,
                        claim.predicate_key,
                        claim.object_kind.as_str(),
                        claim.object_text,
                        claim.object_entity_id,
                        claim.normalized_object,
                        claim.polarity.as_str(),
                        claim.cardinality.as_str(),
                        claim.certainty.as_str(),
                        state.as_str(),
                        claim.asserted_at,
                        claim.event_time,
                        claim.valid_from,
                        claim.valid_to,
                        claim.reference_time,
                        batch.batch_key,
                        attempt.completed_at,
                    ],
                )?;
                insert_transition(
                    transaction,
                    claim_id,
                    None,
                    *state,
                    "created",
                    None,
                    batch,
                    &attempt.completed_at,
                )?;
                report.claims_created += 1;
                if *state == MemoryClaimState::Conflicted {
                    report.claims_conflicted += 1;
                }

                for old_claim_id in conflicts {
                    let old_state = read_claim_state(transaction, old_claim_id)?;
                    if old_state != MemoryClaimState::Conflicted {
                        let changed = transaction.execute(
                            "UPDATE memory_claims
                             SET state = 'conflicted', updated_batch_key = ?1, updated_at = ?2
                             WHERE claim_id = ?3 AND state = ?4",
                            params![
                                batch.batch_key,
                                attempt.completed_at,
                                old_claim_id,
                                old_state.as_str(),
                            ],
                        )?;
                        if changed != 1 {
                            return Err(rusqlite::Error::QueryReturnedNoRows);
                        }
                        insert_transition(
                            transaction,
                            old_claim_id,
                            Some(old_state),
                            MemoryClaimState::Conflicted,
                            "conflicted",
                            Some(claim_id),
                            batch,
                            &attempt.completed_at,
                        )?;
                        report.claims_conflicted += 1;
                    }
                }
                if let Some(reason) = supersede_reason {
                    for old_claim_id in supersedes {
                        let (old_state, old_valid_to) =
                            read_claim_state_and_end(transaction, old_claim_id)?;
                        let changed = transaction.execute(
                            "UPDATE memory_claims
                             SET state = 'superseded',
                                 valid_to = CASE WHEN valid_to IS NULL THEN ?1 ELSE valid_to END,
                                 updated_batch_key = ?2, updated_at = ?3
                             WHERE claim_id = ?4 AND state = ?5",
                            params![
                                claim.valid_from,
                                batch.batch_key,
                                attempt.completed_at,
                                old_claim_id,
                                old_state.as_str(),
                            ],
                        )?;
                        if changed != 1 {
                            return Err(rusqlite::Error::QueryReturnedNoRows);
                        }
                        let transition_reason = if *reason == "corrected" {
                            "corrected"
                        } else {
                            "replaced"
                        };
                        insert_transition(
                            transaction,
                            old_claim_id,
                            Some(old_state),
                            MemoryClaimState::Superseded,
                            transition_reason,
                            Some(claim_id),
                            batch,
                            &attempt.completed_at,
                        )?;
                        let _ = old_valid_to;
                        report.claims_superseded += 1;
                    }
                }
                claim_id
            }
            ValidatedClaimAction::Confirm {
                claim_id,
                previous_state,
                final_state,
                certainty_upgraded,
            } => {
                let changed = if *certainty_upgraded {
                    transaction.execute(
                        "UPDATE memory_claims
                         SET state = ?1, certainty = 'certain', updated_batch_key = ?2,
                             updated_at = ?3
                         WHERE claim_id = ?4 AND state = ?5",
                        params![
                            final_state.as_str(),
                            batch.batch_key,
                            attempt.completed_at,
                            claim_id,
                            previous_state.as_str(),
                        ],
                    )?
                } else {
                    transaction.execute(
                        "UPDATE memory_claims SET updated_batch_key = ?1, updated_at = ?2
                         WHERE claim_id = ?3 AND state = ?4",
                        params![
                            batch.batch_key,
                            attempt.completed_at,
                            claim_id,
                            previous_state.as_str(),
                        ],
                    )?
                };
                if changed != 1 {
                    return Err(rusqlite::Error::QueryReturnedNoRows);
                }
                insert_transition(
                    transaction,
                    claim_id,
                    Some(*previous_state),
                    *final_state,
                    if *certainty_upgraded {
                        "certainty_upgraded"
                    } else {
                        "confirmed"
                    },
                    None,
                    batch,
                    &attempt.completed_at,
                )?;
                report.claims_confirmed += 1;
                claim_id
            }
        };

        for evidence in &claim.evidence {
            if insert_claim_evidence(
                transaction,
                claim_id,
                evidence,
                batch,
                &attempt.completed_at,
            )? {
                report.evidence_created += 1;
            }
        }
    }

    for boundary in &plan.boundaries {
        transaction.execute(
            "INSERT INTO memory_boundary_suggestions
             (boundary_id, session_id, batch_key, before_event_id, reason, evidence_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                boundary.boundary_id,
                batch.session_id,
                batch.batch_key,
                boundary.before_event_id,
                boundary.reason.as_str(),
                boundary.evidence_json,
                attempt.completed_at,
            ],
        )?;
        report.boundaries_created += 1;
    }
    Ok(())
}

fn read_claim_state(
    transaction: &Transaction<'_>,
    claim_id: &str,
) -> rusqlite::Result<MemoryClaimState> {
    let value = transaction.query_row(
        "SELECT state FROM memory_claims WHERE claim_id = ?1",
        [claim_id],
        |row| row.get::<_, String>(0),
    )?;
    parse_claim_state(&value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                error.to_string(),
            )),
        )
    })
}

fn read_claim_state_and_end(
    transaction: &Transaction<'_>,
    claim_id: &str,
) -> rusqlite::Result<(MemoryClaimState, Option<String>)> {
    let (state, valid_to) = transaction.query_row(
        "SELECT state, valid_to FROM memory_claims WHERE claim_id = ?1",
        [claim_id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
    )?;
    Ok((
        parse_claim_state(&state).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    error.to_string(),
                )),
            )
        })?,
        valid_to,
    ))
}

#[allow(clippy::too_many_arguments)]
fn insert_transition(
    transaction: &Transaction<'_>,
    claim_id: &str,
    from_state: Option<MemoryClaimState>,
    to_state: MemoryClaimState,
    reason: &str,
    related_claim_id: Option<&str>,
    batch: &ConsolidationInputBatch,
    created_at: &str,
) -> rusqlite::Result<()> {
    let transition_id = deterministic_id(
        "transition",
        &[
            claim_id,
            from_state.map(MemoryClaimState::as_str).unwrap_or(""),
            to_state.as_str(),
            reason,
            related_claim_id.unwrap_or(""),
            batch.batch_key.as_str(),
        ],
    );
    transaction.execute(
        "INSERT INTO memory_claim_transitions
         (transition_id, claim_id, from_state, to_state, reason, related_claim_id,
          session_id, batch_key, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            transition_id,
            claim_id,
            from_state.map(MemoryClaimState::as_str),
            to_state.as_str(),
            reason,
            related_claim_id,
            batch.session_id,
            batch.batch_key,
            created_at,
        ],
    )?;
    Ok(())
}

fn insert_claim_evidence(
    transaction: &Transaction<'_>,
    claim_id: &str,
    evidence: &ValidatedEvidence,
    batch: &ConsolidationInputBatch,
    created_at: &str,
) -> rusqlite::Result<bool> {
    let existing = transaction
        .query_row(
            "SELECT claim_id, session_id, batch_key, event_id, sequence, role, kind,
                    start_char, end_char, content_sha256
             FROM memory_claim_evidence WHERE evidence_id = ?1",
            [&evidence.evidence_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            },
        )
        .optional()?;
    let sequence = i64::try_from(evidence.quote.sequence)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let start = i64::try_from(evidence.quote.quote.start_char)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let end = i64::try_from(evidence.quote.quote.end_char)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    if let Some(existing) = existing {
        let expected = (
            claim_id.to_owned(),
            batch.session_id.clone(),
            batch.batch_key.clone(),
            evidence.quote.quote.event_id.clone(),
            sequence,
            evidence.quote.role.as_str().to_owned(),
            evidence.kind.as_str().to_owned(),
            start,
            end,
            evidence.quote.quote.content_sha256.clone(),
        );
        if existing == expected {
            return Ok(false);
        }
        return Err(rusqlite::Error::InvalidQuery);
    }
    transaction.execute(
        "INSERT INTO memory_claim_evidence
         (evidence_id, claim_id, session_id, batch_key, event_id, sequence, role, kind,
          start_char, end_char, content_sha256, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            evidence.evidence_id,
            claim_id,
            batch.session_id,
            batch.batch_key,
            evidence.quote.quote.event_id,
            sequence,
            evidence.quote.role.as_str(),
            evidence.kind.as_str(),
            start,
            end,
            evidence.quote.quote.content_sha256,
            created_at,
        ],
    )?;
    Ok(true)
}

fn validated_resume_index(
    events: &[StoredEvent],
    watermark: &ConsolidationWatermark,
) -> RetrievalResult<usize> {
    if watermark.through_sequence == 0 {
        return Ok(0);
    }
    let event_id = watermark
        .through_event_id
        .as_deref()
        .ok_or_else(|| RetrievalError::CorruptIndex("非零巩固水位缺少事件 ID".into()))?;
    let event_hash = watermark
        .through_event_sha256
        .as_deref()
        .ok_or_else(|| RetrievalError::CorruptIndex("非零巩固水位缺少事件哈希".into()))?;
    let position = events
        .iter()
        .position(|event| event.sequence == watermark.through_sequence)
        .ok_or_else(|| {
            RetrievalError::CorruptIndex(format!(
                "巩固水位序号 {} 找不到原始事件",
                watermark.through_sequence
            ))
        })?;
    let event = &events[position];
    if event.id != event_id || event.content_sha256 != event_hash {
        return Err(RetrievalError::CorruptIndex(format!(
            "巩固水位序号 {} 的事件来源不匹配",
            watermark.through_sequence
        )));
    }
    if events
        .get(position + 1)
        .is_some_and(|next| next.turn_id.is_some() && next.turn_id == event.turn_id)
    {
        return Err(RetrievalError::CorruptIndex(format!(
            "巩固水位落在轮次 {} 内部",
            event.turn_id.as_deref().unwrap_or("<system>")
        )));
    }
    Ok(position + 1)
}

fn consolidation_batch_key(
    session_id: &str,
    watermark_before: usize,
    through_sequence: usize,
    events: &[ConsolidationEvent],
) -> String {
    let mut hasher = Sha256::new();
    hash_length_delimited(&mut hasher, CONSOLIDATION_BATCH_KEY_VERSION.as_bytes());
    hash_length_delimited(&mut hasher, session_id.as_bytes());
    hash_length_delimited(&mut hasher, watermark_before.to_string().as_bytes());
    hash_length_delimited(&mut hasher, through_sequence.to_string().as_bytes());
    for event in events {
        hash_length_delimited(&mut hasher, event.event_id.as_bytes());
        hash_length_delimited(&mut hasher, event.content_sha256.as_bytes());
    }
    format!("cb_{:x}", hasher.finalize())
}

fn hash_length_delimited(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn validate_attempt(record: &ConsolidationAttemptRecord) -> RetrievalResult<()> {
    for (name, value) in [
        ("attempt_id", record.attempt_id.as_str()),
        ("batch_key", record.batch_key.as_str()),
        ("session_id", record.session_id.as_str()),
        ("trigger", record.trigger.as_str()),
        ("model", record.model.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(invalid_attempt(format!("{name} 不能为空")));
        }
    }
    if record
        .input_event_ids
        .iter()
        .any(|event_id| event_id.trim().is_empty())
    {
        return Err(invalid_attempt("input_event_ids 包含空 ID"));
    }
    if record.from_sequence > record.through_sequence {
        return Err(invalid_attempt("from_sequence 不能大于 through_sequence"));
    }
    validate_exact_hash(
        "request_sha256",
        &record.request_sha256,
        record.request_json.as_bytes(),
    )?;
    validate_json("request_json", &record.request_json)?;
    if record.input_event_ids.len() != record.input_event_hashes.len() {
        return Err(invalid_attempt("输入事件 ID 与哈希数量不一致"));
    }
    if record
        .input_event_hashes
        .iter()
        .any(|hash| !is_lower_sha256(hash))
    {
        return Err(invalid_attempt(
            "input_event_hashes 必须是小写 64 位十六进制 SHA-256",
        ));
    }
    match (&record.response_json, &record.response_sha256) {
        (None, None) => {}
        (Some(response), Some(hash)) => {
            validate_exact_hash("response_sha256", hash, response.as_bytes())?;
            if record.status == ConsolidationAttemptStatus::Applied {
                validate_json("response_json", response)?;
            }
        }
        _ => return Err(invalid_attempt("响应 JSON 与哈希必须同时存在或同时缺失")),
    }
    for (name, value) in [
        ("validation_json", record.validation_json.as_deref()),
        ("error_json", record.error_json.as_deref()),
    ] {
        if let Some(value) = value {
            validate_json(name, value)?;
        }
    }
    attempt_usize_to_sql(record.from_sequence, "from_sequence")?;
    attempt_usize_to_sql(record.through_sequence, "through_sequence")?;
    attempt_u64_to_sql(record.latency_ms, "latency_ms")?;
    if let Some(value) = record.input_tokens {
        attempt_u64_to_sql(value, "input_tokens")?;
    }
    if let Some(value) = record.output_tokens {
        attempt_u64_to_sql(value, "output_tokens")?;
    }
    Ok(())
}

fn validate_json(name: &str, value: &str) -> RetrievalResult<()> {
    serde_json::from_str::<serde_json::Value>(value)
        .map(|_| ())
        .map_err(|_| invalid_attempt(format!("{name} 不是有效 JSON")))
}

fn validate_exact_hash(name: &str, hash: &str, bytes: &[u8]) -> RetrievalResult<()> {
    if !is_lower_sha256(hash) {
        return Err(invalid_attempt(format!(
            "{name} 必须是小写 64 位十六进制 SHA-256"
        )));
    }
    if sha256_bytes(bytes) != hash {
        return Err(invalid_attempt(format!("{name} 与原始字节不匹配")));
    }
    Ok(())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn invalid_attempt(message: impl Into<String>) -> RetrievalError {
    RetrievalError::CorruptIndex(format!("巩固失败记录无效：{}", message.into()))
}

fn insert_attempt(
    connection: &Connection,
    record: &ConsolidationAttemptRecord,
) -> rusqlite::Result<()> {
    let from_sequence = i64::try_from(record.from_sequence)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let through_sequence = i64::try_from(record.through_sequence)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let input_tokens = record
        .input_tokens
        .map(i64::try_from)
        .transpose()
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let output_tokens = record
        .output_tokens
        .map(i64::try_from)
        .transpose()
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let latency_ms = i64::try_from(record.latency_ms)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let input_event_ids = serde_json::to_string(&record.input_event_ids)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let input_event_hashes = serde_json::to_string(&record.input_event_hashes)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    connection.execute(
        "INSERT INTO consolidation_batches
         (attempt_id, batch_key, session_id, from_sequence, through_sequence, trigger,
          model, request_json, request_sha256, input_event_ids, input_event_hashes,
          response_json, response_sha256, status, input_tokens, output_tokens, latency_ms,
          started_at, completed_at, validation_json, error_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                 ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
        params![
            record.attempt_id,
            record.batch_key,
            record.session_id,
            from_sequence,
            through_sequence,
            record.trigger,
            record.model,
            record.request_json,
            record.request_sha256,
            input_event_ids,
            input_event_hashes,
            record.response_json,
            record.response_sha256,
            record.status.as_str(),
            input_tokens,
            output_tokens,
            latency_ms,
            record.started_at,
            record.completed_at,
            record.validation_json,
            record.error_json,
        ],
    )?;
    Ok(())
}

fn attempt_usize_to_sql(value: usize, name: &str) -> RetrievalResult<i64> {
    i64::try_from(value).map_err(|_| invalid_attempt(format!("{name} 超出 SQLite INTEGER")))
}

fn attempt_u64_to_sql(value: u64, name: &str) -> RetrievalResult<i64> {
    i64::try_from(value).map_err(|_| invalid_attempt(format!("{name} 超出 SQLite INTEGER")))
}

fn nonnegative_usize(value: i64, name: &str) -> RetrievalResult<usize> {
    usize::try_from(value).map_err(|_| RetrievalError::CorruptIndex(format!("{name} 不是非负整数")))
}

#[derive(Debug)]
struct StoredAttempt {
    attempt_id: String,
    batch_key: String,
    session_id: String,
    from_sequence: i64,
    through_sequence: i64,
    trigger: String,
    model: String,
    request_json: String,
    request_sha256: String,
    input_event_ids: String,
    input_event_hashes: String,
    response_json: Option<String>,
    response_sha256: Option<String>,
    status: String,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    latency_ms: i64,
    started_at: String,
    completed_at: String,
    validation_json: Option<String>,
    error_json: Option<String>,
}

fn map_stored_attempt(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredAttempt> {
    Ok(StoredAttempt {
        attempt_id: row.get(0)?,
        batch_key: row.get(1)?,
        session_id: row.get(2)?,
        from_sequence: row.get(3)?,
        through_sequence: row.get(4)?,
        trigger: row.get(5)?,
        model: row.get(6)?,
        request_json: row.get(7)?,
        request_sha256: row.get(8)?,
        input_event_ids: row.get(9)?,
        input_event_hashes: row.get(10)?,
        response_json: row.get(11)?,
        response_sha256: row.get(12)?,
        status: row.get(13)?,
        input_tokens: row.get(14)?,
        output_tokens: row.get(15)?,
        latency_ms: row.get(16)?,
        started_at: row.get(17)?,
        completed_at: row.get(18)?,
        validation_json: row.get(19)?,
        error_json: row.get(20)?,
    })
}

fn decode_stored_attempt(stored: StoredAttempt) -> RetrievalResult<ConsolidationAttemptRecord> {
    let status = match stored.status.as_str() {
        "applied" => ConsolidationAttemptStatus::Applied,
        "rejected" => ConsolidationAttemptStatus::Rejected,
        "model_error" => ConsolidationAttemptStatus::ModelError,
        "cancelled" => ConsolidationAttemptStatus::Cancelled,
        value => {
            return Err(RetrievalError::CorruptIndex(format!(
                "巩固失败记录包含未知状态 {value}"
            )));
        }
    };
    let input_event_ids = serde_json::from_str::<Vec<String>>(&stored.input_event_ids)
        .map_err(|_| RetrievalError::CorruptIndex("巩固输入事件 ID 数组损坏".into()))?;
    let input_event_hashes = serde_json::from_str::<Vec<String>>(&stored.input_event_hashes)
        .map_err(|_| RetrievalError::CorruptIndex("巩固输入事件哈希数组损坏".into()))?;
    let record = ConsolidationAttemptRecord {
        attempt_id: stored.attempt_id,
        batch_key: stored.batch_key,
        session_id: stored.session_id,
        from_sequence: nonnegative_usize(stored.from_sequence, "attempt.from_sequence")?,
        through_sequence: nonnegative_usize(stored.through_sequence, "attempt.through_sequence")?,
        trigger: stored.trigger,
        model: stored.model,
        request_json: stored.request_json,
        request_sha256: stored.request_sha256,
        input_event_ids,
        input_event_hashes,
        response_json: stored.response_json,
        response_sha256: stored.response_sha256,
        status,
        input_tokens: stored
            .input_tokens
            .map(|value| nonnegative_u64(value, "attempt.input_tokens"))
            .transpose()?,
        output_tokens: stored
            .output_tokens
            .map(|value| nonnegative_u64(value, "attempt.output_tokens"))
            .transpose()?,
        latency_ms: nonnegative_u64(stored.latency_ms, "attempt.latency_ms")?,
        started_at: stored.started_at,
        completed_at: stored.completed_at,
        validation_json: stored.validation_json,
        error_json: stored.error_json,
    };
    validate_attempt(&record)?;
    Ok(record)
}

fn nonnegative_u64(value: i64, name: &str) -> RetrievalResult<u64> {
    u64::try_from(value).map_err(|_| RetrievalError::CorruptIndex(format!("{name} 不是非负整数")))
}

#[cfg(test)]
mod tests {
    use rusqlite::{Connection, params};

    use super::*;
    use crate::model::{EventRole, Session, Turn, TurnStatus, content_sha256, utc_now};
    use crate::retrieval::INDEX_FILENAME;
    use crate::store::SessionStore;

    fn new_session(root: &std::path::Path) -> (SessionStore, Session) {
        let store = SessionStore::new(root).unwrap();
        let session = store
            .create("model", "http://localhost", None, Default::default(), false)
            .unwrap();
        (store, session)
    }

    fn push_turn(
        session: &mut Session,
        user: impl Into<String>,
        status: TurnStatus,
        assistant: Option<&str>,
    ) -> String {
        let mut turn = Turn::pending(user.into());
        turn.status = status;
        if let Some(assistant) = assistant {
            turn.request_started_at = Some(utc_now());
            turn.context_trace.provenance_quality = crate::model::ProvenanceQuality::LegacyInferred;
            turn.assistant_content = assistant.to_owned();
        }
        let turn_id = turn.id.clone();
        session.turns.push(turn);
        turn_id
    }

    fn seed_watermark(
        store: &SessionStore,
        session_id: &str,
        sequence: usize,
        event_id: &str,
        event_hash: &str,
    ) {
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "INSERT INTO consolidation_watermarks
                 (session_id, through_sequence, through_event_id, through_event_sha256, updated_at)
                 VALUES (?1, ?2, ?3, ?4, '2026-01-01T00:00:00Z')
                 ON CONFLICT(session_id) DO UPDATE SET
                    through_sequence=excluded.through_sequence,
                    through_event_id=excluded.through_event_id,
                    through_event_sha256=excluded.through_event_sha256,
                    updated_at=excluded.updated_at",
                params![session_id, sequence as i64, event_id, event_hash],
            )
            .unwrap();
    }

    fn failed_attempt(
        attempt_id: &str,
        batch_key: &str,
        session_id: &str,
    ) -> ConsolidationAttemptRecord {
        let request_json = "{\"events\":[\"e1\",\"e2\"]}".to_owned();
        let response_json = "{\"entities\":[],\"claims\":[]}".to_owned();
        ConsolidationAttemptRecord {
            attempt_id: attempt_id.to_owned(),
            batch_key: batch_key.to_owned(),
            session_id: session_id.to_owned(),
            from_sequence: 1,
            through_sequence: 2,
            trigger: "tui_exit".into(),
            model: "qwen3.5:9b".into(),
            request_sha256: sha256_bytes(request_json.as_bytes()),
            request_json,
            input_event_ids: vec!["e1".into(), "e2".into()],
            input_event_hashes: vec![content_sha256("one"), content_sha256("two")],
            response_sha256: Some(sha256_bytes(response_json.as_bytes())),
            response_json: Some(response_json),
            status: ConsolidationAttemptStatus::Rejected,
            input_tokens: Some(41),
            output_tokens: Some(7),
            latency_ms: 1234,
            started_at: "2026-01-01T00:00:00Z".into(),
            completed_at: "2026-01-01T00:00:01Z".into(),
            validation_json: Some("{\"path\":\"$.claims[0]\"}".into()),
            error_json: Some("{\"message\":\"invalid\"}".into()),
        }
    }

    fn seed_attempt_direct(store: &RetrievalStore, record: &ConsolidationAttemptRecord) {
        store
            .consolidation_attempts(&record.session_id)
            .expect("initialize consolidation schema");
        let connection = Connection::open(store.index_path()).unwrap();
        connection
            .execute(
                "INSERT INTO consolidation_batches
                 (attempt_id, batch_key, session_id, from_sequence, through_sequence, trigger,
                  model, request_json, request_sha256, input_event_ids, input_event_hashes,
                  response_json, response_sha256, status, input_tokens, output_tokens, latency_ms,
                  started_at, completed_at, validation_json, error_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                         ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
                params![
                    record.attempt_id,
                    record.batch_key,
                    record.session_id,
                    record.from_sequence as i64,
                    record.through_sequence as i64,
                    record.trigger,
                    record.model,
                    record.request_json,
                    record.request_sha256,
                    serde_json::to_string(&record.input_event_ids).unwrap(),
                    serde_json::to_string(&record.input_event_hashes).unwrap(),
                    record.response_json,
                    record.response_sha256,
                    record.status.as_str(),
                    record.input_tokens.map(|value| value as i64),
                    record.output_tokens.map(|value| value as i64),
                    record.latency_ms as i64,
                    record.started_at,
                    record.completed_at,
                    record.validation_json,
                    record.error_json,
                ],
            )
            .unwrap();
    }

    fn push_complete_at(
        session: &mut Session,
        user: &str,
        assistant: Option<&str>,
        created_at: &str,
    ) {
        let mut turn = Turn::pending(user.to_owned());
        turn.created_at = created_at.to_owned();
        turn.updated_at = created_at.to_owned();
        turn.status = TurnStatus::Complete;
        if let Some(assistant) = assistant {
            turn.request_started_at = Some(created_at.to_owned());
            turn.assistant_content = assistant.to_owned();
            turn.context_trace.provenance_quality = crate::model::ProvenanceQuality::LegacyInferred;
        }
        session.turns.push(turn);
    }

    fn next_batch(store: &SessionStore, session: &mut Session) -> ConsolidationInputBatch {
        store.save(session).unwrap();
        store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap()
    }

    fn quote_nth(
        event: &ConsolidationEvent,
        needle: &str,
        occurrence: usize,
    ) -> ConsolidationQuote {
        let mut offset = 0_usize;
        let mut found = None;
        for _ in 0..=occurrence {
            let relative = event.content[offset..].find(needle).unwrap();
            let byte = offset + relative;
            found = Some(byte);
            offset = byte + needle.len();
        }
        let start_byte = found.unwrap();
        let start_char = event.content[..start_byte].chars().count();
        let end_char = start_char + needle.chars().count();
        ConsolidationQuote {
            event_id: event.event_id.clone(),
            start_char,
            end_char,
            content_sha256: sha256_bytes(needle.as_bytes()),
        }
    }

    fn full_quote(event: &ConsolidationEvent) -> ConsolidationQuote {
        ConsolidationQuote {
            event_id: event.event_id.clone(),
            start_char: 0,
            end_char: event.content.chars().count(),
            content_sha256: sha256_bytes(event.content.as_bytes()),
        }
    }

    fn empty_candidates() -> ConsolidationCandidateSnapshot {
        let entities = Vec::new();
        let claims = Vec::new();
        ConsolidationCandidateSnapshot {
            snapshot_sha256: candidate_snapshot_hash(&entities, &claims).unwrap(),
            entities,
            claims,
        }
    }

    fn new_entity_output(
        local_id: &str,
        name: &str,
        evidence: ConsolidationQuote,
    ) -> ConsolidatedEntityOutput {
        ConsolidatedEntityOutput {
            local_id: local_id.to_owned(),
            name: name.to_owned(),
            kind: MemoryEntityKind::Person,
            resolution: EntityResolution::New,
            disambiguation: EntityDisambiguation::Resolved,
            basis: EntityResolutionBasis::FirstMention,
            existing_entity_id: None,
            name_evidence: evidence,
            resolution_evidence: None,
            aliases: Vec::new(),
        }
    }

    fn text_claim_output(
        local_id: &str,
        subject_ref: &str,
        predicate_key: &str,
        text: &str,
        span: ConsolidationQuote,
        assertion: ConsolidationQuote,
    ) -> ConsolidatedClaimOutput {
        ConsolidatedClaimOutput {
            local_id: local_id.to_owned(),
            subject_ref: subject_ref.to_owned(),
            predicate_key: predicate_key.to_owned(),
            object: ConsolidatedClaimObject {
                kind: ConsolidationClaimObjectKind::Text,
                text: Some(text.to_owned()),
                entity_ref: None,
                span: Some(span),
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
                quote: assertion,
            }],
        }
    }

    fn applied_attempt(
        batch: &ConsolidationInputBatch,
        output: &StructuredConsolidationOutput,
    ) -> ConsolidationAttemptRecord {
        let request_json = serde_json::to_string(&json!({
            "batch_key": batch.batch_key,
            "candidate_snapshot": "test"
        }))
        .unwrap();
        let response_json = serde_json::to_string(output).unwrap();
        ConsolidationAttemptRecord {
            attempt_id: deterministic_id("attempt", &[&batch.batch_key, &response_json]),
            batch_key: batch.batch_key.clone(),
            session_id: batch.session_id.clone(),
            from_sequence: batch.from_sequence,
            through_sequence: batch.through_sequence,
            trigger: "test".into(),
            model: "qwen3.5:9b".into(),
            request_sha256: sha256_bytes(request_json.as_bytes()),
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
            response_sha256: Some(sha256_bytes(response_json.as_bytes())),
            response_json: Some(response_json),
            status: ConsolidationAttemptStatus::Applied,
            input_tokens: Some(10),
            output_tokens: Some(5),
            latency_ms: 1,
            started_at: "2026-01-01T00:00:00Z".into(),
            completed_at: "2026-01-01T00:00:01Z".into(),
            validation_json: Some("{\"valid\":true}".into()),
            error_json: None,
        }
    }

    fn apply_output(
        store: &SessionStore,
        batch: &ConsolidationInputBatch,
        candidates: &ConsolidationCandidateSnapshot,
        output: &StructuredConsolidationOutput,
    ) -> ConsolidationApplyResult<ConsolidationApplyReport> {
        store.retrieval().apply_consolidation_attempt(
            batch,
            candidates,
            &applied_attempt(batch, output),
        )
    }

    #[test]
    fn terminal_batching_stops_at_limits_and_pending_barrier() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        let first_turn = push_turn(
            &mut session,
            "first",
            TurnStatus::Complete,
            Some("first answer"),
        );
        for index in 1..17 {
            push_turn(
                &mut session,
                format!("turn {index}"),
                TurnStatus::Complete,
                None,
            );
        }
        store.save(&mut session).unwrap();

        let first = store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        assert_eq!(first.turn_count, CONSOLIDATION_MAX_TURNS);
        assert_eq!(first.events.len(), 17);
        assert_eq!(first.events[0].turn_id, first_turn);
        assert_eq!(first.events[0].role, EventRole::User);
        assert_eq!(first.events[1].turn_id, first_turn);
        assert_eq!(first.events[1].role, EventRole::Assistant);
        assert_eq!(first.events[1].content, "first answer");
        seed_watermark(
            &store,
            &session.id,
            first.through_sequence,
            &first.through_event_id,
            &first.through_event_sha256,
        );
        let second = store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        assert_eq!(second.turn_count, 1);
        assert_eq!(second.events[0].content, "turn 16");

        let barrier_root = tempfile::tempdir().unwrap();
        let (barrier_store, mut barrier_session) = new_session(barrier_root.path());
        push_turn(
            &mut barrier_session,
            "failed input",
            TurnStatus::Failed,
            None,
        );
        push_turn(
            &mut barrier_session,
            "no answer input",
            TurnStatus::NoAnswer,
            None,
        );
        push_turn(
            &mut barrier_session,
            "pending barrier",
            TurnStatus::Pending,
            None,
        );
        push_turn(
            &mut barrier_session,
            "must not pass",
            TurnStatus::Complete,
            None,
        );
        barrier_store.save(&mut barrier_session).unwrap();
        let batch = barrier_store
            .retrieval()
            .next_consolidation_batch(&barrier_session.id)
            .unwrap()
            .unwrap();
        assert_eq!(batch.turn_count, 2);
        assert_eq!(
            batch
                .events
                .iter()
                .map(|event| event.content.as_str())
                .collect::<Vec<_>>(),
            vec!["failed input", "no answer input"]
        );
        assert!(batch.events.iter().all(|event| !event.turn_id.is_empty()));
    }

    #[test]
    fn scalar_limit_keeps_turns_atomic_and_allows_oversized_first_turn() {
        let exact_root = tempfile::tempdir().unwrap();
        let (exact_store, mut exact_session) = new_session(exact_root.path());
        push_turn(
            &mut exact_session,
            "🙂".repeat(CONSOLIDATION_MAX_CHARS - 1),
            TurnStatus::Complete,
            Some("界"),
        );
        push_turn(&mut exact_session, "later", TurnStatus::Complete, None);
        exact_store.save(&mut exact_session).unwrap();
        let exact = exact_store
            .retrieval()
            .next_consolidation_batch(&exact_session.id)
            .unwrap()
            .unwrap();
        assert_eq!(exact.turn_count, 1);
        assert_eq!(exact.events.len(), 2);
        assert_eq!(exact.char_count, CONSOLIDATION_MAX_CHARS);
        assert_eq!(exact.events[0].content.chars().count(), 23_999);
        assert_eq!(exact.events[1].content, "界");

        let oversized_root = tempfile::tempdir().unwrap();
        let (oversized_store, mut oversized_session) = new_session(oversized_root.path());
        push_turn(
            &mut oversized_session,
            "好".repeat(CONSOLIDATION_MAX_CHARS + 1),
            TurnStatus::Complete,
            None,
        );
        push_turn(&mut oversized_session, "later", TurnStatus::Complete, None);
        oversized_store.save(&mut oversized_session).unwrap();
        let oversized = oversized_store
            .retrieval()
            .next_consolidation_batch(&oversized_session.id)
            .unwrap()
            .unwrap();
        assert_eq!(oversized.turn_count, 1);
        assert_eq!(oversized.events.len(), 1);
        assert_eq!(oversized.char_count, CONSOLIDATION_MAX_CHARS + 1);
        assert!(
            oversized
                .events
                .iter()
                .all(|event| event.content != "later")
        );
    }

    #[test]
    fn batch_key_is_deterministic_and_binds_ordered_provenance() {
        let event = |id: &str, content: &str, sequence: usize| ConsolidationEvent {
            event_id: id.into(),
            turn_id: format!("turn-{sequence}"),
            sequence,
            role: EventRole::User,
            created_at: "2026-01-01T00:00:00Z".into(),
            content: content.into(),
            content_sha256: content_sha256(content),
        };
        let events = vec![event("e1", "one", 1), event("e2", "two", 3)];
        let original = consolidation_batch_key("session", 0, 3, &events);
        assert_eq!(original, consolidation_batch_key("session", 0, 3, &events));
        assert!(original.starts_with("cb_"));

        let mut changed_hash = events.clone();
        changed_hash[0].content_sha256 = content_sha256("changed");
        assert_ne!(
            original,
            consolidation_batch_key("session", 0, 3, &changed_hash)
        );
        let mut changed_id = events.clone();
        changed_id[1].event_id = "e3".into();
        assert_ne!(
            original,
            consolidation_batch_key("session", 0, 3, &changed_id)
        );
        let mut reversed = events.clone();
        reversed.reverse();
        assert_ne!(
            original,
            consolidation_batch_key("session", 0, 3, &reversed)
        );
        assert_ne!(original, consolidation_batch_key("session", 2, 3, &events));
        assert_ne!(original, consolidation_batch_key("session", 0, 4, &events));
    }

    #[test]
    fn watermark_absence_resume_and_provenance_corruption_are_deterministic() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_turn(&mut session, "first", TurnStatus::Complete, Some("answer"));
        push_turn(&mut session, "second", TurnStatus::Blocked, None);
        store.save(&mut session).unwrap();
        assert_eq!(
            store
                .retrieval()
                .consolidation_watermark(&session.id)
                .unwrap(),
            ConsolidationWatermark {
                session_id: session.id.clone(),
                through_sequence: 0,
                through_event_id: None,
                through_event_sha256: None,
                updated_at: None,
            }
        );
        let all = store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        let first_answer = &all.events[1];

        seed_watermark(
            &store,
            &session.id,
            99,
            "missing",
            &content_sha256("missing"),
        );
        assert!(matches!(
            store.retrieval().next_consolidation_batch(&session.id),
            Err(RetrievalError::CorruptIndex(_))
        ));
        seed_watermark(
            &store,
            &session.id,
            first_answer.sequence,
            &first_answer.event_id,
            &content_sha256("wrong"),
        );
        assert!(matches!(
            store.retrieval().next_consolidation_batch(&session.id),
            Err(RetrievalError::CorruptIndex(_))
        ));
        seed_watermark(
            &store,
            &session.id,
            first_answer.sequence,
            &first_answer.event_id,
            &first_answer.content_sha256,
        );
        let watermark = store
            .retrieval()
            .consolidation_watermark(&session.id)
            .unwrap();
        assert_eq!(watermark.through_sequence, first_answer.sequence);
        assert_eq!(
            store
                .retrieval()
                .next_consolidation_batch(&session.id)
                .unwrap()
                .unwrap()
                .events
                .iter()
                .map(|event| event.content.as_str())
                .collect::<Vec<_>>(),
            vec!["second"]
        );
    }

    #[test]
    fn failure_attempts_round_trip_retry_and_never_create_watermark() {
        let root = tempfile::tempdir().unwrap();
        let store = RetrievalStore::new(root.path()).unwrap();
        let first = failed_attempt("attempt-b", "cb_same", "session");
        let mut second = first.clone();
        second.attempt_id = "attempt-a".into();
        second.status = ConsolidationAttemptStatus::ModelError;
        second.response_json = None;
        second.response_sha256 = None;
        second.input_tokens = None;
        second.output_tokens = None;
        second.validation_json = None;
        second.error_json = Some("{\"message\":\"offline\"}".into());
        let mut third = first.clone();
        third.attempt_id = "attempt-c".into();
        third.status = ConsolidationAttemptStatus::Cancelled;
        third.started_at = "2026-01-01T00:00:02Z".into();
        third.completed_at = "2026-01-01T00:00:02Z".into();
        third.response_json = None;
        third.response_sha256 = None;
        let expected = vec![second.clone(), first.clone(), third.clone()];

        store.record_consolidation_failure(&first).unwrap();
        store.record_consolidation_failure(&second).unwrap();
        store.record_consolidation_failure(&third).unwrap();
        assert_eq!(store.consolidation_attempts("session").unwrap(), expected);
        assert_eq!(
            store.consolidation_watermark("session").unwrap(),
            ConsolidationWatermark {
                session_id: "session".into(),
                through_sequence: 0,
                through_event_id: None,
                through_event_sha256: None,
                updated_at: None,
            }
        );

        let mut duplicate = first.clone();
        duplicate.model = "must-not-overwrite".into();
        assert!(matches!(
            store.record_consolidation_failure(&duplicate),
            Err(RetrievalError::Database { .. })
        ));
        assert_eq!(store.consolidation_attempts("session").unwrap(), expected);
    }

    #[test]
    fn applied_status_is_reserved_but_failure_api_cannot_write_it() {
        assert_eq!(
            serde_json::to_string(&ConsolidationAttemptStatus::Applied).unwrap(),
            "\"applied\""
        );
        assert_eq!(
            serde_json::from_str::<ConsolidationAttemptStatus>("\"applied\"").unwrap(),
            ConsolidationAttemptStatus::Applied
        );

        let root = tempfile::tempdir().unwrap();
        let store = RetrievalStore::new(root.path()).unwrap();
        assert!(store.consolidation_attempts("session").unwrap().is_empty());
        let mut applied = failed_attempt("applied-attempt", "cb_applied", "session");
        applied.status = ConsolidationAttemptStatus::Applied;

        assert!(matches!(
            store.record_consolidation_failure(&applied),
            Err(RetrievalError::CorruptIndex(message))
                if message == "巩固失败记录无效：record_consolidation_failure 不接受 applied 状态"
        ));
        assert!(store.consolidation_attempts("session").unwrap().is_empty());
        assert_eq!(
            store.consolidation_watermark("session").unwrap(),
            ConsolidationWatermark {
                session_id: "session".into(),
                through_sequence: 0,
                through_event_id: None,
                through_event_sha256: None,
                updated_at: None,
            }
        );

        seed_attempt_direct(&store, &applied);
        assert_eq!(
            store.consolidation_attempts("session").unwrap(),
            vec![applied]
        );
        assert_eq!(
            store
                .consolidation_watermark("session")
                .unwrap()
                .through_sequence,
            0
        );
    }

    #[test]
    fn invalid_attempts_are_rejected_before_insert() {
        let root = tempfile::tempdir().unwrap();
        let store = RetrievalStore::new(root.path()).unwrap();
        let base = failed_attempt("attempt", "cb_batch", "session");
        let mut invalid = Vec::new();

        let mut blank = base.clone();
        blank.trigger = "  ".into();
        invalid.push(blank);
        let mut reversed = base.clone();
        reversed.from_sequence = 3;
        reversed.through_sequence = 2;
        invalid.push(reversed);
        let mut request_hash = base.clone();
        request_hash.request_sha256 = content_sha256("wrong");
        invalid.push(request_hash);
        let mut malformed_request = base.clone();
        malformed_request.request_json = "{".into();
        malformed_request.request_sha256 = sha256_bytes(malformed_request.request_json.as_bytes());
        invalid.push(malformed_request);
        let mut arrays = base.clone();
        arrays.input_event_hashes.pop();
        invalid.push(arrays);
        let mut input_hash = base.clone();
        input_hash.input_event_hashes[0] = "ABC".into();
        invalid.push(input_hash);
        let mut partial_response = base.clone();
        partial_response.response_sha256 = None;
        invalid.push(partial_response);
        let mut response_hash = base.clone();
        response_hash.response_sha256 = Some(content_sha256("wrong"));
        invalid.push(response_hash);
        let mut validation_json = base.clone();
        validation_json.validation_json = Some("not json".into());
        invalid.push(validation_json);
        let mut error_json = base;
        error_json.error_json = Some("[".into());
        invalid.push(error_json);

        for record in invalid {
            assert!(matches!(
                store.record_consolidation_failure(&record),
                Err(RetrievalError::CorruptIndex(message))
                    if message.starts_with("巩固失败记录无效：")
            ));
        }
        assert!(store.consolidation_attempts("session").unwrap().is_empty());
    }

    #[test]
    fn malformed_request_is_rejected_but_rejected_raw_response_is_preserved() {
        let request_root = tempfile::tempdir().unwrap();
        let request_store = RetrievalStore::new(request_root.path()).unwrap();
        let mut malformed_request = failed_attempt("request", "cb_request", "session");
        malformed_request.request_json = "{".into();
        malformed_request.request_sha256 = sha256_bytes(malformed_request.request_json.as_bytes());
        seed_attempt_direct(&request_store, &malformed_request);
        assert!(matches!(
            request_store.consolidation_attempts("session"),
            Err(RetrievalError::CorruptIndex(message))
                if message == "巩固失败记录无效：request_json 不是有效 JSON"
        ));

        let response_root = tempfile::tempdir().unwrap();
        let response_store = RetrievalStore::new(response_root.path()).unwrap();
        let mut malformed_response = failed_attempt("response", "cb_response", "session");
        malformed_response.response_json = Some("[".into());
        malformed_response.response_sha256 = malformed_response
            .response_json
            .as_ref()
            .map(|response| sha256_bytes(response.as_bytes()));
        seed_attempt_direct(&response_store, &malformed_response);
        assert_eq!(
            response_store.consolidation_attempts("session").unwrap(),
            vec![malformed_response]
        );
    }

    #[test]
    fn v3_migration_is_additive_and_unknown_v5_precedes_ddl() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_turn(&mut session, "sentinel", TurnStatus::Complete, None);
        store.save(&mut session).unwrap();
        {
            let connection = Connection::open(store.retrieval().index_path()).unwrap();
            connection
                .execute_batch(
                    "DROP TABLE consolidation_batches;
                     DROP TABLE consolidation_watermarks;
                     PRAGMA user_version=3;",
                )
                .unwrap();
        }
        let migrated = RetrievalStore::new(root.path()).unwrap();
        assert_eq!(
            migrated.replay_session(&session.id).unwrap()[1].content,
            "sentinel"
        );
        let connection = Connection::open(migrated.index_path()).unwrap();
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            4
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM events", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM consolidation_batches", [], |row| row
                    .get::<_, i64>(
                    0
                ))
                .unwrap(),
            0
        );

        let unknown_root = tempfile::tempdir().unwrap();
        let index = unknown_root.path().join(INDEX_FILENAME);
        let unknown = Connection::open(&index).unwrap();
        unknown.pragma_update(None, "user_version", 5_i64).unwrap();
        drop(unknown);
        let unsupported = RetrievalStore::new(unknown_root.path()).unwrap();
        assert!(matches!(
            unsupported.consolidation_attempts("none"),
            Err(RetrievalError::UnsupportedIndexVersion(5))
        ));
        let unknown = Connection::open(index).unwrap();
        assert_eq!(
            unknown
                .query_row(
                    "SELECT count(*) FROM sqlite_master WHERE name='consolidation_batches'",
                    [],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn rebuild_preserves_ledger_and_watermark_then_revalidates_source() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_turn(&mut session, "first", TurnStatus::Complete, None);
        push_turn(&mut session, "second", TurnStatus::Complete, None);
        store.save(&mut session).unwrap();
        let initial = store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        let first = &initial.events[0];
        seed_watermark(
            &store,
            &session.id,
            first.sequence,
            &first.event_id,
            &first.content_sha256,
        );
        let attempt = failed_attempt("attempt", &initial.batch_key, &session.id);
        store
            .retrieval()
            .record_consolidation_failure(&attempt)
            .unwrap();

        store.retrieval().rebuild().unwrap();
        assert_eq!(
            store
                .retrieval()
                .consolidation_attempts(&session.id)
                .unwrap(),
            vec![attempt]
        );
        assert_eq!(
            store
                .retrieval()
                .consolidation_watermark(&session.id)
                .unwrap()
                .through_sequence,
            first.sequence
        );
        let resumed = store
            .retrieval()
            .next_consolidation_batch(&session.id)
            .unwrap()
            .unwrap();
        assert_eq!(resumed.events.len(), 1);
        assert_eq!(resumed.events[0].content, "second");
    }

    #[test]
    fn structured_schema_is_recursively_strict_and_serde_denies_unknown_fields() {
        fn assert_strict_objects(value: &Value, path: &str) {
            if value.get("type") == Some(&Value::String("object".into())) {
                assert_eq!(
                    value.get("additionalProperties"),
                    Some(&Value::Bool(false)),
                    "object is not strict at {path}"
                );
                assert!(
                    value.get("required").is_some(),
                    "missing required at {path}"
                );
            }
            match value {
                Value::Object(map) => {
                    for (key, child) in map {
                        assert_strict_objects(child, &format!("{path}/{key}"));
                    }
                }
                Value::Array(items) => {
                    for (index, child) in items.iter().enumerate() {
                        assert_strict_objects(child, &format!("{path}/{index}"));
                    }
                }
                _ => {}
            }
        }

        let schema = structured_consolidation_schema();
        assert_strict_objects(&schema, "$");
        assert_eq!(schema["properties"]["entities"]["maxItems"], 128);
        assert_eq!(schema["properties"]["claims"]["maxItems"], 256);
        assert_eq!(schema["properties"]["boundaries"]["maxItems"], 64);
        assert_eq!(
            schema["$defs"]["entity"]["properties"]["aliases"]["maxItems"],
            16
        );
        assert_eq!(
            schema["$defs"]["claim"]["properties"]["evidence"]["maxItems"],
            16
        );
        assert_eq!(
            schema["$defs"]["boundary"]["properties"]["evidence"]["maxItems"],
            8
        );

        assert!(
            serde_json::from_value::<StructuredConsolidationOutput>(json!({
                "entities": [], "claims": [], "boundaries": [], "extra": true
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<ConsolidationQuote>(json!({
                "event_id": "e", "start_char": 0, "end_char": 1,
                "content_sha256": "0".repeat(64), "extra": true
            }))
            .is_err()
        );
    }

    #[test]
    fn unicode_scalar_quotes_accept_exact_spans_and_reject_byte_offsets_rewrites_and_hashes() {
        let event = ConsolidationEvent {
            event_id: "evt_unicode".into(),
            turn_id: "turn".into(),
            sequence: 1,
            role: EventRole::User,
            created_at: "2026-01-01T00:00:00Z".into(),
            content: "甲🙂乙".into(),
            content_sha256: content_sha256("甲🙂乙"),
        };
        let batch = ConsolidationInputBatch {
            batch_key: "unused".into(),
            session_id: "session".into(),
            watermark_before: 0,
            from_sequence: 1,
            through_sequence: 1,
            through_event_id: event.event_id.clone(),
            through_event_sha256: event.content_sha256.clone(),
            turn_count: 1,
            char_count: 3,
            events: vec![event.clone()],
        };
        let emoji = ConsolidationQuote {
            event_id: event.event_id.clone(),
            start_char: 1,
            end_char: 2,
            content_sha256: content_sha256("🙂"),
        };
        assert_eq!(validate_quote(&batch, &emoji, "quote").unwrap().text, "🙂");

        let mut byte_offset = emoji.clone();
        byte_offset.start_char = 3;
        byte_offset.end_char = 4;
        assert!(matches!(
            validate_quote(&batch, &byte_offset, "quote"),
            Err(ConsolidationApplyError::Rejected { .. })
        ));
        let mut wrong_hash = emoji.clone();
        wrong_hash.content_sha256 = content_sha256("乙");
        assert!(matches!(
            validate_quote(&batch, &wrong_hash, "quote"),
            Err(ConsolidationApplyError::Rejected { .. })
        ));

        let mut entity = new_entity_output("local_emoji", "改写", emoji);
        entity.kind = MemoryEntityKind::Object;
        let output = StructuredConsolidationOutput {
            entities: vec![entity],
            claims: Vec::new(),
            boundaries: Vec::new(),
        };
        assert!(matches!(
            validate_structured_output(&batch, &empty_candidates(), &output),
            Err(ConsolidationApplyError::Rejected { .. })
        ));
    }

    #[test]
    fn zero_output_applies_atomically_and_invalid_multi_item_output_writes_nothing() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_complete_at(
            &mut session,
            "这一轮没有可提取的记忆",
            None,
            "2026-01-01T00:00:00Z",
        );
        let batch = next_batch(&store, &mut session);
        let output = StructuredConsolidationOutput {
            entities: Vec::new(),
            claims: Vec::new(),
            boundaries: Vec::new(),
        };
        let report = apply_output(&store, &batch, &empty_candidates(), &output).unwrap();
        assert_eq!(report.watermark_after, batch.through_sequence);
        assert_eq!(report.entities_created, 0);
        assert_eq!(
            store
                .retrieval()
                .consolidation_attempts(&session.id)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store
                .retrieval()
                .consolidation_watermark(&session.id)
                .unwrap()
                .through_sequence,
            batch.through_sequence
        );
        assert!(matches!(
            apply_output(&store, &batch, &empty_candidates(), &output),
            Err(ConsolidationApplyError::Stale { .. })
        ));

        let invalid_root = tempfile::tempdir().unwrap();
        let (invalid_store, mut invalid_session) = new_session(invalid_root.path());
        push_complete_at(
            &mut invalid_session,
            "Alice喜欢蓝色",
            None,
            "2026-01-01T00:00:00Z",
        );
        let invalid_batch = next_batch(&invalid_store, &mut invalid_session);
        let event = &invalid_batch.events[0];
        let entity = new_entity_output("local_alice", "Alice", quote_nth(event, "Alice", 0));
        let mut invalid_claim = text_claim_output(
            "local_claim",
            "local_alice",
            "likes_color",
            "蓝色",
            quote_nth(event, "蓝色", 0),
            full_quote(event),
        );
        invalid_claim.predicate_key = "INVALID KEY".into();
        let invalid_output = StructuredConsolidationOutput {
            entities: vec![entity],
            claims: vec![invalid_claim],
            boundaries: Vec::new(),
        };
        let error = apply_output(
            &invalid_store,
            &invalid_batch,
            &empty_candidates(),
            &invalid_output,
        )
        .unwrap_err();
        let ConsolidationApplyError::Rejected {
            validation_json, ..
        } = error
        else {
            panic!("expected rejected output");
        };
        assert!(serde_json::from_str::<Value>(&validation_json).is_ok());
        let connection = Connection::open(invalid_store.retrieval().index_path()).unwrap();
        for table in [
            "memory_entities",
            "memory_claims",
            "consolidation_batches",
            "consolidation_watermarks",
        ] {
            let count = connection
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap();
            assert_eq!(count, 0, "unexpected write in {table}");
        }
    }

    #[test]
    fn distinct_people_keep_their_own_hair_claims_without_cross_linking() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_complete_at(
            &mut session,
            "A是长发，B是短发。",
            None,
            "2026-01-01T00:00:00Z",
        );
        let batch = next_batch(&store, &mut session);
        let event = &batch.events[0];
        let output = StructuredConsolidationOutput {
            entities: vec![
                new_entity_output("local_a", "A", quote_nth(event, "A", 0)),
                new_entity_output("local_b", "B", quote_nth(event, "B", 0)),
            ],
            claims: vec![
                text_claim_output(
                    "local_a_hair",
                    "local_a",
                    "appearance.hair",
                    "长发",
                    quote_nth(event, "长发", 0),
                    quote_nth(event, "A是长发", 0),
                ),
                text_claim_output(
                    "local_b_hair",
                    "local_b",
                    "appearance.hair",
                    "短发",
                    quote_nth(event, "短发", 0),
                    quote_nth(event, "B是短发", 0),
                ),
            ],
            boundaries: Vec::new(),
        };
        let report = apply_output(&store, &batch, &empty_candidates(), &output).unwrap();
        assert_eq!(report.entities_created, 2);
        assert_eq!(report.claims_created, 2);
        let snapshot = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        let by_name = snapshot
            .entities
            .iter()
            .map(|entity| (entity.canonical_name.as_str(), entity.entity_id.as_str()))
            .collect::<HashMap<_, _>>();
        assert_ne!(by_name["A"], by_name["B"]);
        let claims = snapshot
            .claims
            .iter()
            .map(|claim| {
                (
                    claim.subject_entity_id.as_str(),
                    claim.object_text.as_deref().unwrap(),
                )
            })
            .collect::<HashSet<_>>();
        assert!(claims.contains(&(by_name["A"], "长发")));
        assert!(claims.contains(&(by_name["B"], "短发")));
        assert!(!claims.contains(&(by_name["A"], "短发")));
        assert!(!claims.contains(&(by_name["B"], "长发")));
    }

    #[test]
    fn same_names_and_pronouns_stay_separate_pending_and_name_only_merge_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_complete_at(
            &mut session,
            "甲叫小王，乙也叫小王，他尚未确定。",
            None,
            "2026-01-01T00:00:00Z",
        );
        let batch = next_batch(&store, &mut session);
        let event = &batch.events[0];
        let mut first = new_entity_output("local_first", "小王", quote_nth(event, "小王", 0));
        first.basis = EntityResolutionBasis::Ambiguous;
        first.disambiguation = EntityDisambiguation::Pending;
        let mut second = new_entity_output("local_second", "小王", quote_nth(event, "小王", 1));
        second.basis = EntityResolutionBasis::Ambiguous;
        second.disambiguation = EntityDisambiguation::Pending;
        let mut pronoun = new_entity_output("local_pronoun", "他", quote_nth(event, "他", 0));
        pronoun.basis = EntityResolutionBasis::Ambiguous;
        pronoun.disambiguation = EntityDisambiguation::Pending;
        let output = StructuredConsolidationOutput {
            entities: vec![first, second, pronoun],
            claims: Vec::new(),
            boundaries: Vec::new(),
        };
        apply_output(&store, &batch, &empty_candidates(), &output).unwrap();
        let snapshot = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(snapshot.entities.len(), 3);
        assert_eq!(
            snapshot
                .entities
                .iter()
                .map(|entity| entity.entity_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
            3
        );
        assert!(
            snapshot
                .entities
                .iter()
                .all(|entity| entity.disambiguation == EntityDisambiguation::Pending)
        );

        push_complete_at(&mut session, "小王今天来了。", None, "2026-01-02T00:00:00Z");
        let second_batch = next_batch(&store, &mut session);
        let second_event = &second_batch.events[0];
        let target = snapshot
            .entities
            .iter()
            .find(|entity| entity.canonical_name == "小王")
            .unwrap();
        let name_only = ConsolidatedEntityOutput {
            local_id: "local_name_only".into(),
            name: "小王".into(),
            kind: MemoryEntityKind::Person,
            resolution: EntityResolution::Existing,
            disambiguation: EntityDisambiguation::Resolved,
            basis: EntityResolutionBasis::FirstMention,
            existing_entity_id: Some(target.entity_id.clone()),
            name_evidence: quote_nth(second_event, "小王", 0),
            resolution_evidence: None,
            aliases: Vec::new(),
        };
        let output = StructuredConsolidationOutput {
            entities: vec![name_only],
            claims: Vec::new(),
            boundaries: Vec::new(),
        };
        assert!(matches!(
            apply_output(&store, &second_batch, &snapshot, &output),
            Err(ConsolidationApplyError::Rejected { .. })
        ));
        assert_eq!(
            store
                .retrieval()
                .consolidation_watermark(&session.id)
                .unwrap()
                .through_sequence,
            batch.through_sequence
        );
    }

    #[test]
    fn self_pronouns_are_role_bound_and_reuse_ent_self() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_complete_at(&mut session, "我喜欢茶。", None, "2026-01-01T00:00:00Z");
        let batch = next_batch(&store, &mut session);
        let event = &batch.events[0];
        let self_output = ConsolidatedEntityOutput {
            local_id: "local_self".into(),
            name: "我".into(),
            kind: MemoryEntityKind::Person,
            resolution: EntityResolution::SelfEntity,
            disambiguation: EntityDisambiguation::Resolved,
            basis: EntityResolutionBasis::SelfPronoun,
            existing_entity_id: None,
            name_evidence: quote_nth(event, "我", 0),
            resolution_evidence: None,
            aliases: Vec::new(),
        };
        apply_output(
            &store,
            &batch,
            &empty_candidates(),
            &StructuredConsolidationOutput {
                entities: vec![self_output],
                claims: Vec::new(),
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        let candidates = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(candidates.entities[0].entity_id, "ent_self");

        push_complete_at(
            &mut session,
            "你并不是用户第一人称。",
            None,
            "2026-01-02T00:00:00Z",
        );
        let wrong_batch = next_batch(&store, &mut session);
        let wrong_event = &wrong_batch.events[0];
        let wrong = ConsolidatedEntityOutput {
            local_id: "local_wrong_self".into(),
            name: "你".into(),
            kind: MemoryEntityKind::Person,
            resolution: EntityResolution::SelfEntity,
            disambiguation: EntityDisambiguation::Resolved,
            basis: EntityResolutionBasis::SelfPronoun,
            existing_entity_id: None,
            name_evidence: quote_nth(wrong_event, "你", 0),
            resolution_evidence: None,
            aliases: Vec::new(),
        };
        assert!(matches!(
            apply_output(
                &store,
                &wrong_batch,
                &candidates,
                &StructuredConsolidationOutput {
                    entities: vec![wrong],
                    claims: Vec::new(),
                    boundaries: Vec::new(),
                },
            ),
            Err(ConsolidationApplyError::Rejected { .. })
        ));

        let assistant_event = ConsolidationEvent {
            event_id: "assistant".into(),
            turn_id: "turn".into(),
            sequence: 2,
            role: EventRole::Assistant,
            created_at: "2026-01-01T00:00:00Z".into(),
            content: "你".into(),
            content_sha256: content_sha256("你"),
        };
        let manual_batch = ConsolidationInputBatch {
            batch_key: "manual".into(),
            session_id: "manual".into(),
            watermark_before: 0,
            from_sequence: 2,
            through_sequence: 2,
            through_event_id: "assistant".into(),
            through_event_sha256: content_sha256("你"),
            turn_count: 1,
            char_count: 1,
            events: vec![assistant_event.clone()],
        };
        let assistant_self = ConsolidatedEntityOutput {
            local_id: "local_assistant_self".into(),
            name: "你".into(),
            kind: MemoryEntityKind::Person,
            resolution: EntityResolution::SelfEntity,
            disambiguation: EntityDisambiguation::Resolved,
            basis: EntityResolutionBasis::SelfPronoun,
            existing_entity_id: None,
            name_evidence: full_quote(&assistant_event),
            resolution_evidence: None,
            aliases: Vec::new(),
        };
        assert!(
            validate_structured_output(
                &manual_batch,
                &empty_candidates(),
                &StructuredConsolidationOutput {
                    entities: vec![assistant_self],
                    claims: Vec::new(),
                    boundaries: Vec::new(),
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn explicit_alias_and_unique_stable_identifier_merge_only_with_exact_user_proof() {
        let alias_root = tempfile::tempdir().unwrap();
        let (alias_store, mut alias_session) = new_session(alias_root.path());
        push_complete_at(
            &mut alias_session,
            "王明的别名是小明。",
            None,
            "2026-01-01T00:00:00Z",
        );
        let first_batch = next_batch(&alias_store, &mut alias_session);
        let first_event = &first_batch.events[0];
        let mut entity = new_entity_output("local_wang", "王明", quote_nth(first_event, "王明", 0));
        entity.aliases = vec![EntityAliasOutput {
            text: "小明".into(),
            kind: MemoryAliasKind::ExplicitAlias,
            stable_identifier_kind: None,
            evidence: quote_nth(first_event, "小明", 0),
            proof_evidence: full_quote(first_event),
        }];
        apply_output(
            &alias_store,
            &first_batch,
            &empty_candidates(),
            &StructuredConsolidationOutput {
                entities: vec![entity],
                claims: Vec::new(),
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        let first_candidates = alias_store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        let target_id = first_candidates.entities[0].entity_id.clone();

        push_complete_at(
            &mut alias_session,
            "小明即王明。",
            None,
            "2026-01-02T00:00:00Z",
        );
        let alias_batch = next_batch(&alias_store, &mut alias_session);
        let alias_event = &alias_batch.events[0];
        let proof = full_quote(alias_event);
        let existing = ConsolidatedEntityOutput {
            local_id: "local_alias_existing".into(),
            name: "小明".into(),
            kind: MemoryEntityKind::Person,
            resolution: EntityResolution::Existing,
            disambiguation: EntityDisambiguation::Resolved,
            basis: EntityResolutionBasis::ExplicitAlias,
            existing_entity_id: Some(target_id.clone()),
            name_evidence: quote_nth(alias_event, "小明", 0),
            resolution_evidence: Some(proof.clone()),
            aliases: vec![EntityAliasOutput {
                text: "小明".into(),
                kind: MemoryAliasKind::ExplicitAlias,
                stable_identifier_kind: None,
                evidence: quote_nth(alias_event, "小明", 0),
                proof_evidence: proof,
            }],
        };
        let report = apply_output(
            &alias_store,
            &alias_batch,
            &first_candidates,
            &StructuredConsolidationOutput {
                entities: vec![existing],
                claims: Vec::new(),
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(report.entities_reused, 1);
        assert_eq!(
            alias_store
                .retrieval()
                .consolidation_candidates(512, 512)
                .unwrap()
                .entities[0]
                .entity_id,
            target_id
        );

        let stable_root = tempfile::tempdir().unwrap();
        let (stable_store, mut stable_session) = new_session(stable_root.path());
        push_complete_at(
            &mut stable_session,
            "张三的工号是E123。",
            None,
            "2026-01-01T00:00:00Z",
        );
        let stable_batch = next_batch(&stable_store, &mut stable_session);
        let stable_event = &stable_batch.events[0];
        let mut stable_new =
            new_entity_output("local_zhang", "张三", quote_nth(stable_event, "张三", 0));
        stable_new.aliases = vec![EntityAliasOutput {
            text: "E123".into(),
            kind: MemoryAliasKind::StableIdentifier,
            stable_identifier_kind: Some("employee_id".into()),
            evidence: quote_nth(stable_event, "E123", 0),
            proof_evidence: full_quote(stable_event),
        }];
        apply_output(
            &stable_store,
            &stable_batch,
            &empty_candidates(),
            &StructuredConsolidationOutput {
                entities: vec![stable_new],
                claims: Vec::new(),
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        let stable_candidates = stable_store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        let stable_target = stable_candidates.entities[0].entity_id.clone();
        push_complete_at(
            &mut stable_session,
            "李雷就是工号E123的张三。",
            None,
            "2026-01-02T00:00:00Z",
        );
        let merge_batch = next_batch(&stable_store, &mut stable_session);
        let merge_event = &merge_batch.events[0];
        let stable_proof = full_quote(merge_event);
        let stable_existing = ConsolidatedEntityOutput {
            local_id: "local_stable_existing".into(),
            name: "李雷".into(),
            kind: MemoryEntityKind::Person,
            resolution: EntityResolution::Existing,
            disambiguation: EntityDisambiguation::Resolved,
            basis: EntityResolutionBasis::StableIdentifier,
            existing_entity_id: Some(stable_target.clone()),
            name_evidence: quote_nth(merge_event, "李雷", 0),
            resolution_evidence: Some(stable_proof.clone()),
            aliases: vec![EntityAliasOutput {
                text: "E123".into(),
                kind: MemoryAliasKind::StableIdentifier,
                stable_identifier_kind: Some("employee_id".into()),
                evidence: quote_nth(merge_event, "E123", 0),
                proof_evidence: stable_proof,
            }],
        };
        assert!(
            apply_output(
                &stable_store,
                &merge_batch,
                &stable_candidates,
                &StructuredConsolidationOutput {
                    entities: vec![stable_existing.clone()],
                    claims: Vec::new(),
                    boundaries: Vec::new(),
                },
            )
            .is_ok()
        );

        let mut ambiguous = stable_candidates.clone();
        let mut duplicate = ambiguous.entities[0].clone();
        duplicate.entity_id = "ent_duplicate".into();
        duplicate.aliases[0].alias_id = "alias_duplicate".into();
        ambiguous.entities.push(duplicate);
        ambiguous
            .entities
            .sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
        ambiguous.snapshot_sha256 =
            candidate_snapshot_hash(&ambiguous.entities, &ambiguous.claims).unwrap();
        assert!(matches!(
            validate_structured_output(
                &merge_batch,
                &ambiguous,
                &StructuredConsolidationOutput {
                    entities: vec![stable_existing],
                    claims: Vec::new(),
                    boundaries: Vec::new(),
                },
            ),
            Err(ConsolidationApplyError::Rejected { .. })
        ));
    }

    #[test]
    fn assistant_assertions_require_later_explicit_user_confirmation() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_complete_at(
            &mut session,
            "请判断Alice的偏好。",
            Some("Alice喜欢蓝色。"),
            "2026-01-01T00:00:00Z",
        );
        let batch = next_batch(&store, &mut session);
        let user = batch
            .events
            .iter()
            .find(|event| event.role == EventRole::User)
            .unwrap();
        let assistant = batch
            .events
            .iter()
            .find(|event| event.role == EventRole::Assistant)
            .unwrap();
        let entity = new_entity_output("local_alice", "Alice", quote_nth(user, "Alice", 0));
        let claim = text_claim_output(
            "local_preference",
            "local_alice",
            "preference.color",
            "蓝色",
            quote_nth(assistant, "蓝色", 0),
            full_quote(assistant),
        );
        let output = StructuredConsolidationOutput {
            entities: vec![entity],
            claims: vec![claim],
            boundaries: Vec::new(),
        };
        assert!(matches!(
            apply_output(&store, &batch, &empty_candidates(), &output),
            Err(ConsolidationApplyError::Rejected { .. })
        ));

        let confirmed_root = tempfile::tempdir().unwrap();
        let (confirmed_store, mut confirmed_session) = new_session(confirmed_root.path());
        push_complete_at(
            &mut confirmed_session,
            "请判断Alice的偏好。",
            Some("Alice喜欢蓝色。"),
            "2026-01-01T00:00:00Z",
        );
        push_complete_at(
            &mut confirmed_session,
            "我确认Alice喜欢蓝色，时间是2026-02-01。",
            None,
            "2026-02-01T12:00:00Z",
        );
        let confirmed_batch = next_batch(&confirmed_store, &mut confirmed_session);
        let first_user = &confirmed_batch.events[0];
        let assistant = &confirmed_batch.events[1];
        let confirmation = &confirmed_batch.events[2];
        let entity = new_entity_output("local_alice", "Alice", quote_nth(first_user, "Alice", 0));
        let mut claim = text_claim_output(
            "local_preference",
            "local_alice",
            "preference.color",
            "蓝色",
            quote_nth(assistant, "蓝色", 0),
            full_quote(assistant),
        );
        claim.evidence.push(ConsolidationClaimEvidence {
            kind: ConsolidationEvidenceKind::UserConfirmation,
            quote: quote_nth(confirmation, "我确认Alice喜欢蓝色", 0),
        });
        claim.evidence.push(ConsolidationClaimEvidence {
            kind: ConsolidationEvidenceKind::Temporal,
            quote: quote_nth(confirmation, "2026-02-01", 0),
        });
        claim.event_time = Some("2026-02-01T00:00:00Z".into());
        let output = StructuredConsolidationOutput {
            entities: vec![entity],
            claims: vec![claim],
            boundaries: Vec::new(),
        };
        apply_output(
            &confirmed_store,
            &confirmed_batch,
            &empty_candidates(),
            &output,
        )
        .unwrap();
        let saved = confirmed_store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(
            saved.claims[0].event_time.as_deref(),
            Some("2026-02-01T00:00:00Z")
        );
        assert_eq!(saved.claims[0].valid_from, "2026-02-01T00:00:00Z");
        assert_eq!(saved.claims[0].asserted_at, "2026-02-01T12:00:00Z");
        assert_eq!(saved.claims[0].reference_time, saved.claims[0].asserted_at);
    }

    #[test]
    fn claim_state_machine_confirms_conflicts_corrects_and_keeps_multi_values() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_complete_at(
            &mut session,
            "Alice的状态是旧值。",
            None,
            "2026-01-01T00:00:00Z",
        );
        let first_batch = next_batch(&store, &mut session);
        let first_event = &first_batch.events[0];
        let entity = new_entity_output("local_alice", "Alice", quote_nth(first_event, "Alice", 0));
        let mut uncertain = text_claim_output(
            "local_old",
            "local_alice",
            "profile.state",
            "旧值",
            quote_nth(first_event, "旧值", 0),
            full_quote(first_event),
        );
        uncertain.certainty = ClaimCertainty::Uncertain;
        apply_output(
            &store,
            &first_batch,
            &empty_candidates(),
            &StructuredConsolidationOutput {
                entities: vec![entity],
                claims: vec![uncertain],
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        let first = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(first.claims[0].state, MemoryClaimState::Uncertain);
        let entity_id = first.entities[0].entity_id.clone();
        let old_id = first.claims[0].claim_id.clone();
        let old_valid_from = first.claims[0].valid_from.clone();

        push_complete_at(
            &mut session,
            "我确认Alice的状态是旧值。",
            None,
            "2026-01-02T00:00:00Z",
        );
        let confirm_batch = next_batch(&store, &mut session);
        let confirm_event = &confirm_batch.events[0];
        let mut confirm = text_claim_output(
            "local_confirm",
            &entity_id,
            "profile.state",
            "旧值",
            quote_nth(confirm_event, "旧值", 0),
            full_quote(confirm_event),
        );
        confirm.disposition = ClaimDisposition::Confirm;
        let report = apply_output(
            &store,
            &confirm_batch,
            &first,
            &StructuredConsolidationOutput {
                entities: Vec::new(),
                claims: vec![confirm],
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(report.claims_confirmed, 1);
        let confirmed = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(confirmed.claims.len(), 1);
        assert_eq!(confirmed.claims[0].claim_id, old_id);
        assert_eq!(confirmed.claims[0].state, MemoryClaimState::Active);
        assert_eq!(confirmed.claims[0].certainty, ClaimCertainty::Certain);
        assert_eq!(confirmed.claims[0].valid_from, old_valid_from);
        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM memory_claim_evidence", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            2
        );
        drop(connection);

        push_complete_at(
            &mut session,
            "Alice的状态是新值。",
            None,
            "2026-01-03T00:00:00Z",
        );
        let conflict_batch = next_batch(&store, &mut session);
        let conflict_event = &conflict_batch.events[0];
        let mut conflict = text_claim_output(
            "local_new",
            &entity_id,
            "profile.state",
            "新值",
            quote_nth(conflict_event, "新值", 0),
            full_quote(conflict_event),
        );
        conflict.conflicts_with_claim_ids = vec![old_id.clone()];
        let report = apply_output(
            &store,
            &conflict_batch,
            &confirmed,
            &StructuredConsolidationOutput {
                entities: Vec::new(),
                claims: vec![conflict],
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(report.claims_conflicted, 2);
        let conflicted = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(
            conflicted
                .claims
                .iter()
                .filter(|claim| claim.state == MemoryClaimState::Conflicted)
                .count(),
            2
        );
        let mut replaced_ids = conflicted
            .claims
            .iter()
            .map(|claim| claim.claim_id.clone())
            .collect::<Vec<_>>();
        replaced_ids.sort();

        push_complete_at(
            &mut session,
            "更正：Alice的状态应为最终值。",
            None,
            "2026-01-04T00:00:00Z",
        );
        let correction_batch = next_batch(&store, &mut session);
        let correction_event = &correction_batch.events[0];
        let mut correction = text_claim_output(
            "local_final",
            &entity_id,
            "profile.state",
            "最终值",
            quote_nth(correction_event, "最终值", 0),
            full_quote(correction_event),
        );
        correction.disposition = ClaimDisposition::Correct;
        correction.replaces_claim_ids = replaced_ids.clone();
        correction.evidence[0].kind = ConsolidationEvidenceKind::Correction;
        let report = apply_output(
            &store,
            &correction_batch,
            &conflicted,
            &StructuredConsolidationOutput {
                entities: Vec::new(),
                claims: vec![correction],
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(report.claims_superseded, 2);
        let corrected = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(
            corrected
                .claims
                .iter()
                .filter(|claim| claim.state == MemoryClaimState::Superseded)
                .count(),
            2
        );
        let final_claim = corrected
            .claims
            .iter()
            .find(|claim| claim.object_text.as_deref() == Some("最终值"))
            .unwrap();
        assert_eq!(final_claim.state, MemoryClaimState::Active);
        for old in corrected
            .claims
            .iter()
            .filter(|claim| claim.state == MemoryClaimState::Superseded)
        {
            assert_eq!(
                old.valid_to.as_deref(),
                Some(final_claim.valid_from.as_str())
            );
        }

        push_complete_at(
            &mut session,
            "Alice喜欢茶，也喜欢咖啡。",
            None,
            "2026-01-05T00:00:00Z",
        );
        let multi_batch = next_batch(&store, &mut session);
        let multi_event = &multi_batch.events[0];
        let mut tea = text_claim_output(
            "local_tea",
            &entity_id,
            "preference.drink",
            "茶",
            quote_nth(multi_event, "茶", 0),
            quote_nth(multi_event, "Alice喜欢茶", 0),
        );
        tea.cardinality = ClaimCardinality::Multi;
        let mut coffee = text_claim_output(
            "local_coffee",
            &entity_id,
            "preference.drink",
            "咖啡",
            quote_nth(multi_event, "咖啡", 0),
            quote_nth(multi_event, "也喜欢咖啡", 0),
        );
        coffee.cardinality = ClaimCardinality::Multi;
        apply_output(
            &store,
            &multi_batch,
            &corrected,
            &StructuredConsolidationOutput {
                entities: Vec::new(),
                claims: vec![tea, coffee],
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        let multi = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        let tea_id = multi
            .claims
            .iter()
            .find(|claim| {
                claim.predicate_key == "preference.drink"
                    && claim.object_text.as_deref() == Some("茶")
                    && claim.polarity == ClaimPolarity::Assert
            })
            .unwrap()
            .claim_id
            .clone();
        assert!(
            multi
                .claims
                .iter()
                .filter(|claim| claim.predicate_key == "preference.drink")
                .all(|claim| claim.state == MemoryClaimState::Active)
        );

        push_complete_at(
            &mut session,
            "Alice不喜欢茶。",
            None,
            "2026-01-06T00:00:00Z",
        );
        let denial_batch = next_batch(&store, &mut session);
        let denial_event = &denial_batch.events[0];
        let mut denial = text_claim_output(
            "local_denial",
            &entity_id,
            "preference.drink",
            "茶",
            quote_nth(denial_event, "茶", 0),
            full_quote(denial_event),
        );
        denial.cardinality = ClaimCardinality::Multi;
        denial.polarity = ClaimPolarity::Deny;
        denial.conflicts_with_claim_ids = vec![tea_id];
        apply_output(
            &store,
            &denial_batch,
            &multi,
            &StructuredConsolidationOutput {
                entities: Vec::new(),
                claims: vec![denial],
                boundaries: Vec::new(),
            },
        )
        .unwrap();
        let denied = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(
            denied
                .claims
                .iter()
                .filter(|claim| {
                    claim.predicate_key == "preference.drink"
                        && claim.object_text.as_deref() == Some("茶")
                        && claim.state == MemoryClaimState::Conflicted
                })
                .count(),
            2
        );
        assert_eq!(
            denied
                .claims
                .iter()
                .find(|claim| claim.object_text.as_deref() == Some("咖啡"))
                .unwrap()
                .state,
            MemoryClaimState::Active
        );
    }

    #[test]
    fn temporal_fields_require_user_evidence_rfc3339_and_ordered_intervals() {
        let event = ConsolidationEvent {
            event_id: "event".into(),
            turn_id: "turn".into(),
            sequence: 1,
            role: EventRole::User,
            created_at: "2026-02-01T12:00:00Z".into(),
            content: "Alice状态为蓝色，日期2026-02-01。".into(),
            content_sha256: content_sha256("Alice状态为蓝色，日期2026-02-01。"),
        };
        let batch = ConsolidationInputBatch {
            batch_key: "manual".into(),
            session_id: "session".into(),
            watermark_before: 0,
            from_sequence: 1,
            through_sequence: 1,
            through_event_id: event.event_id.clone(),
            through_event_sha256: event.content_sha256.clone(),
            turn_count: 1,
            char_count: event.content.chars().count(),
            events: vec![event.clone()],
        };
        let entity = new_entity_output("local_alice", "Alice", quote_nth(&event, "Alice", 0));
        let mut base = text_claim_output(
            "local_time",
            "local_alice",
            "profile.color",
            "蓝色",
            quote_nth(&event, "蓝色", 0),
            quote_nth(&event, "Alice状态为蓝色", 0),
        );
        base.event_time = Some("2026-02-01T00:00:00Z".into());
        let no_temporal = StructuredConsolidationOutput {
            entities: vec![entity.clone()],
            claims: vec![base.clone()],
            boundaries: Vec::new(),
        };
        assert!(matches!(
            validate_structured_output(&batch, &empty_candidates(), &no_temporal),
            Err(ConsolidationApplyError::Rejected { .. })
        ));

        base.evidence.push(ConsolidationClaimEvidence {
            kind: ConsolidationEvidenceKind::Temporal,
            quote: quote_nth(&event, "2026-02-01", 0),
        });
        let valid = StructuredConsolidationOutput {
            entities: vec![entity.clone()],
            claims: vec![base.clone()],
            boundaries: Vec::new(),
        };
        let plan = validate_structured_output(&batch, &empty_candidates(), &valid).unwrap();
        assert_eq!(plan.claims[0].valid_from, "2026-02-01T00:00:00Z");
        assert_eq!(plan.claims[0].asserted_at, "2026-02-01T12:00:00Z");

        let mut invalid_time = base.clone();
        invalid_time.event_time = Some("2026/02/01".into());
        assert!(matches!(
            validate_structured_output(
                &batch,
                &empty_candidates(),
                &StructuredConsolidationOutput {
                    entities: vec![entity.clone()],
                    claims: vec![invalid_time],
                    boundaries: Vec::new(),
                },
            ),
            Err(ConsolidationApplyError::Rejected { .. })
        ));
        let mut reversed = base;
        reversed.valid_from = Some("2026-03-01T00:00:00Z".into());
        reversed.valid_to = Some("2026-02-01T00:00:00Z".into());
        assert!(matches!(
            validate_structured_output(
                &batch,
                &empty_candidates(),
                &StructuredConsolidationOutput {
                    entities: vec![entity],
                    claims: vec![reversed],
                    boundaries: Vec::new(),
                },
            ),
            Err(ConsolidationApplyError::Rejected { .. })
        ));
    }

    #[test]
    fn boundary_suggestions_deduplicate_and_candidate_hashes_detect_tampering_after_rebuild() {
        let root = tempfile::tempdir().unwrap();
        let (store, mut session) = new_session(root.path());
        push_complete_at(
            &mut session,
            "现在换个话题，Alice来了。",
            None,
            "2026-01-01T00:00:00Z",
        );
        let batch = next_batch(&store, &mut session);
        let event = &batch.events[0];
        let entity = new_entity_output("local_alice", "Alice", quote_nth(event, "Alice", 0));
        let boundary = ConsolidationBoundaryOutput {
            before_event_id: event.event_id.clone(),
            reason: BoundarySuggestionReason::ExplicitTopicTransition,
            evidence: vec![quote_nth(event, "换个话题", 0)],
        };
        let report = apply_output(
            &store,
            &batch,
            &empty_candidates(),
            &StructuredConsolidationOutput {
                entities: vec![entity],
                claims: Vec::new(),
                boundaries: vec![boundary.clone(), boundary],
            },
        )
        .unwrap();
        assert_eq!(report.boundaries_created, 1);
        let before = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(
            before,
            store
                .retrieval()
                .consolidation_candidates(512, 512)
                .unwrap()
        );
        store.retrieval().rebuild().unwrap();
        let after = store
            .retrieval()
            .consolidation_candidates(512, 512)
            .unwrap();
        assert_eq!(before, after);

        let connection = Connection::open(store.retrieval().index_path()).unwrap();
        connection
            .execute(
                "UPDATE memory_entities SET normalized_name = 'tampered' WHERE entity_id = ?1",
                [&before.entities[0].entity_id],
            )
            .unwrap();
        drop(connection);
        assert!(matches!(
            store.retrieval().consolidation_candidates(512, 512),
            Err(RetrievalError::CorruptIndex(_))
        ));
    }
}
