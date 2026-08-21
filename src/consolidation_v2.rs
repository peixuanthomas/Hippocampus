use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::consolidation::{
    ConsolidatedClaimOutput, ConsolidatedEntityOutput, ConsolidationCandidateSnapshot,
    ConsolidationEvidenceKind, ConsolidationInputBatch, StructuredConsolidationOutput,
    structured_consolidation_schema,
};
use crate::model::{ChatMessage, EventRole, content_sha256};
use crate::ollama::StructuredChatRequest;
use crate::retrieval::{RetrievalError, RetrievalResult};

pub const FACTS_SYSTEM_PROMPT: &str = "Extract only user-adopted facts. Source clauses are the only evidence. Assistant context is interpretation-only and must never be quoted as evidence. Return exactly schema-conforming JSON. Every source reference must copy clause_id, event_id, Unicode scalar-value offsets, and exact text from the source catalog. Never return hashes or span IDs. Use a context_id only for a user's explicit confirmation, denial, or correction. Do not invent or summarize.";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceClause {
    pub clause_id: String,
    pub span_id: String,
    pub event_id: String,
    pub start_char: usize,
    pub end_char: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AssistantContext {
    pub context_id: String,
    pub text: String,
    pub extractable: bool,
    #[serde(skip)]
    pub assistant_event_id: String,
}

#[derive(Debug, Clone)]
pub struct PreparedFactRequest {
    pub request: StructuredChatRequest,
    pub clauses: Vec<SourceClause>,
    pub assistant_contexts: Vec<AssistantContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoundaryInput {
    pub session_id: String,
    pub watermark_before: usize,
    pub event_id: String,
    pub sequence: usize,
    pub created_at: String,
    pub content: String,
    pub content_sha256: String,
    pub previous_user_event_id: Option<String>,
    pub previous_user_created_at: Option<String>,
    pub previous_user_content: Option<String>,
    pub previous_assistant_context: Option<AssistantContext>,
    pub cosine_similarity: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BoundaryDecisionV2 {
    pub before_event_id: String,
    pub is_boundary: bool,
    pub reason: String,
    pub evidence_json: String,
    pub signals_json: String,
    pub generator: String,
}

#[derive(Debug, Clone)]
pub enum BoundaryClassification {
    Deterministic(BoundaryDecisionV2),
    Ambiguous,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelBoundaryOutput {
    before_event_id: String,
    is_boundary: bool,
    reason: String,
    evidence: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
struct FactRequestPayload<'a> {
    session_id: &'a str,
    batch_key: &'a str,
    source_clauses: &'a [SourceClause],
    assistant_contexts: &'a [AssistantContext],
    candidate_snapshot_sha256: &'a str,
    candidates: Value,
}

pub fn source_clauses(batch: &ConsolidationInputBatch) -> Vec<SourceClause> {
    batch
        .events
        .iter()
        .filter(|event| event.role == EventRole::User)
        .flat_map(|event| split_event_clauses(&event.event_id, &event.content))
        .collect()
}

pub fn assistant_contexts(batch: &ConsolidationInputBatch) -> Vec<AssistantContext> {
    let mut previous_assistant = None;
    let mut result = Vec::new();
    for event in &batch.events {
        match event.role {
            EventRole::Assistant => previous_assistant = Some(event),
            EventRole::User => {
                if let Some(context) = previous_assistant.take() {
                    result.push(AssistantContext {
                        context_id: assistant_context_id(&context.event_id, &context.content),
                        text: context.content.clone(),
                        extractable: false,
                        assistant_event_id: context.event_id.clone(),
                    });
                }
            }
            EventRole::System => {}
        }
    }
    result
}

pub fn assistant_context_id(event_id: &str, text: &str) -> String {
    deterministic_id(
        "hippocampus-assistant-context-v1",
        event_id,
        0,
        text.chars().count(),
        text,
    )
}

pub fn provenance_span_id(event_id: &str, start: usize, end: usize, text: &str) -> String {
    deterministic_id("hippocampus-span-v1", event_id, start, end, text)
}

pub fn prepare_fact_request(
    model: String,
    batch: &ConsolidationInputBatch,
    candidates: &ConsolidationCandidateSnapshot,
    extra_contexts: &[AssistantContext],
    clause_override: Option<&[SourceClause]>,
    num_ctx: u64,
    num_predict: u64,
) -> RetrievalResult<PreparedFactRequest> {
    let clauses = clause_override.map_or_else(|| source_clauses(batch), <[SourceClause]>::to_vec);
    let mut assistant_contexts = extra_contexts.to_vec();
    assistant_contexts.extend(self::assistant_contexts(batch));
    assistant_contexts.sort_by(|left, right| left.context_id.cmp(&right.context_id));
    assistant_contexts.dedup_by(|left, right| left.context_id == right.context_id);
    let payload = FactRequestPayload {
        session_id: &batch.session_id,
        batch_key: &batch.batch_key,
        source_clauses: &clauses,
        assistant_contexts: &assistant_contexts,
        candidate_snapshot_sha256: &candidates.snapshot_sha256,
        candidates: compact_candidates(candidates, &clauses),
    };
    let content = serde_json::to_string(&payload).map_err(|error| {
        RetrievalError::CorruptIndex(format!("无法序列化 fact v2 请求：{error}"))
    })?;
    Ok(PreparedFactRequest {
        request: StructuredChatRequest {
            model,
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: FACTS_SYSTEM_PROMPT.into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content,
                },
            ],
            schema: fact_schema(),
            think: true,
            num_ctx,
            num_predict,
        },
        clauses,
        assistant_contexts,
    })
}

pub fn split_clause_work(
    clauses: &[SourceClause],
) -> Option<(Vec<SourceClause>, Vec<SourceClause>)> {
    if clauses.len() > 1 {
        let total = clauses
            .iter()
            .map(|clause| clause.text.chars().count())
            .sum::<usize>();
        let mut accumulated = 0usize;
        let mut split = 1usize;
        for (index, clause) in clauses.iter().enumerate().take(clauses.len() - 1) {
            accumulated = accumulated.saturating_add(clause.text.chars().count());
            split = index + 1;
            if accumulated.saturating_mul(2) >= total {
                break;
            }
        }
        return Some((clauses[..split].to_vec(), clauses[split..].to_vec()));
    }
    let clause = clauses.first()?;
    split_single_clause(clause)
}

pub fn merge_fact_outputs(
    outputs: Vec<StructuredConsolidationOutput>,
) -> RetrievalResult<StructuredConsolidationOutput> {
    let mut entities = Vec::<ConsolidatedEntityOutput>::new();
    let mut entity_keys = HashMap::<String, String>::new();
    let mut claims = Vec::<ConsolidatedClaimOutput>::new();
    let mut claim_keys = HashMap::<String, usize>::new();
    for (output_index, output) in outputs.into_iter().enumerate() {
        let mut local_refs = HashMap::<String, String>::new();
        for mut entity in output.entities {
            let original_local_id = entity.local_id.clone();
            let key = serde_json::to_string(&json!({
                "name":entity.name,
                "kind":entity.kind,
                "resolution":entity.resolution,
                "existing_entity_id":entity.existing_entity_id,
                "name_evidence":entity.name_evidence,
            }))
            .map_err(|error| {
                RetrievalError::CorruptIndex(format!("无法合并 entity delta：{error}"))
            })?;
            let global_local = if let Some(existing) = entity_keys.get(&key) {
                existing.clone()
            } else {
                let local = format!(
                    "local_f{output_index}_{}",
                    entity.local_id.trim_start_matches("local_")
                );
                entity.local_id = local.clone();
                entity_keys.insert(key, local.clone());
                entities.push(entity);
                local
            };
            local_refs.insert(original_local_id, global_local);
        }
        for mut claim in output.claims {
            if let Some(mapped) = local_refs.get(&claim.subject_ref) {
                claim.subject_ref = mapped.clone();
            }
            if let Some(entity_ref) = claim.object.entity_ref.as_mut()
                && let Some(mapped) = local_refs.get(entity_ref)
            {
                *entity_ref = mapped.clone();
            }
            claim.local_id = format!(
                "local_f{output_index}_{}",
                claim.local_id.trim_start_matches("local_")
            );
            let key = serde_json::to_string(&json!({
                "subject_ref":claim.subject_ref,
                "predicate_key":claim.predicate_key,
                "object":claim.object,
                "polarity":claim.polarity,
                "cardinality":claim.cardinality,
                "certainty":claim.certainty,
                "disposition":claim.disposition,
            }))
            .map_err(|error| {
                RetrievalError::CorruptIndex(format!("无法合并 claim delta：{error}"))
            })?;
            if let Some(index) = claim_keys.get(&key).copied() {
                let existing = &mut claims[index];
                existing.evidence.extend(claim.evidence);
                existing
                    .evidence
                    .sort_by_key(|evidence| serde_json::to_string(evidence).unwrap_or_default());
                existing.evidence.dedup();
                existing.replaces_claim_ids.extend(claim.replaces_claim_ids);
                existing.replaces_claim_ids.sort();
                existing.replaces_claim_ids.dedup();
                existing
                    .conflicts_with_claim_ids
                    .extend(claim.conflicts_with_claim_ids);
                existing.conflicts_with_claim_ids.sort();
                existing.conflicts_with_claim_ids.dedup();
            } else {
                claim_keys.insert(key, claims.len());
                claims.push(claim);
            }
        }
    }
    Ok(StructuredConsolidationOutput {
        entities,
        claims,
        boundaries: Vec::new(),
    })
}

pub fn repair_fact_request(
    original: &StructuredChatRequest,
    invalid_response: &str,
    validation_json: &str,
) -> StructuredChatRequest {
    let mut repaired = original.clone();
    repaired.messages.push(ChatMessage {
        role: "user".into(),
        content: json!({
            "kind":"repair",
            "invalid_response":invalid_response,
            "validation_errors":serde_json::from_str::<Value>(validation_json)
                .unwrap_or_else(|_| json!([{"path":"$","code":"invalid","message":validation_json}])),
            "instruction":"Return one complete replacement object. Correct only the listed paths. Reuse only source and candidate IDs from the original request."
        }).to_string(),
    });
    repaired
}

pub fn classify_boundary(input: &BoundaryInput, gap_minutes: u64) -> BoundaryClassification {
    let mut signals = Map::new();
    if input.previous_user_event_id.is_none() {
        signals.insert("session_start".into(), Value::Bool(true));
        return BoundaryClassification::Deterministic(boundary_decision(
            input,
            true,
            "session_start",
            signals,
            Vec::new(),
            "rust",
        ));
    }
    let lowered = input.content.to_lowercase();
    let explicit = [
        "换个话题",
        "换一个话题",
        "说点别的",
        "另一个话题",
        "回到刚才",
        "change the subject",
        "new topic",
    ]
    .iter()
    .any(|cue| lowered.contains(cue));
    signals.insert("explicit_topic_transition".into(), Value::Bool(explicit));
    if explicit {
        return BoundaryClassification::Deterministic(boundary_decision(
            input,
            true,
            "explicit_topic_transition",
            signals,
            Vec::new(),
            "rust",
        ));
    }
    let gap = input
        .previous_user_created_at
        .as_deref()
        .and_then(|previous| chrono::DateTime::parse_from_rfc3339(previous).ok())
        .zip(chrono::DateTime::parse_from_rfc3339(&input.created_at).ok())
        .map(|(previous, current)| current.signed_duration_since(previous).num_minutes())
        .filter(|minutes| *minutes >= 0);
    signals.insert(
        "gap_minutes".into(),
        gap.map_or(Value::Null, |value| json!(value)),
    );
    if gap.is_some_and(|minutes| minutes > i64::try_from(gap_minutes).unwrap_or(i64::MAX)) {
        return BoundaryClassification::Deterministic(boundary_decision(
            input,
            true,
            "time_gap",
            signals,
            Vec::new(),
            "rust",
        ));
    }
    signals.insert(
        "embedding_cosine_similarity".into(),
        input
            .cosine_similarity
            .map_or(Value::Null, |value| json!(value)),
    );
    match input.cosine_similarity {
        Some(value) if value < 0.55 => BoundaryClassification::Deterministic(boundary_decision(
            input,
            true,
            "embedding_topic_shift",
            signals,
            Vec::new(),
            "rust",
        )),
        Some(value) if value >= 0.65 => BoundaryClassification::Deterministic(boundary_decision(
            input,
            false,
            "embedding_same_topic",
            signals,
            Vec::new(),
            "rust",
        )),
        _ => BoundaryClassification::Ambiguous,
    }
}

pub fn prepare_boundary_request(
    model: String,
    input: &BoundaryInput,
    num_ctx: u64,
    num_predict: u64,
) -> StructuredChatRequest {
    let batch = ConsolidationInputBatch {
        batch_key: format!("boundary:{}", input.event_id),
        session_id: input.session_id.clone(),
        watermark_before: input.watermark_before,
        from_sequence: input.sequence,
        through_sequence: input.sequence,
        through_event_id: input.event_id.clone(),
        through_event_sha256: input.content_sha256.clone(),
        turn_count: 1,
        char_count: input.content.chars().count(),
        events: vec![crate::consolidation::ConsolidationEvent {
            event_id: input.event_id.clone(),
            turn_id: "boundary".into(),
            sequence: input.sequence,
            role: EventRole::User,
            created_at: input.created_at.clone(),
            content: input.content.clone(),
            content_sha256: input.content_sha256.clone(),
        }],
    };
    let clauses = source_clauses(&batch);
    let payload = json!({
        "current_user_clauses":clauses,
        "previous_user":input.previous_user_content,
        "previous_assistant_context":input.previous_assistant_context,
        "cosine_similarity":input.cosine_similarity,
        "instruction":"Decide whether the current user event starts a new topic. Assistant text is context-only. Evidence may reference current user clauses only."
    });
    StructuredChatRequest {
        model,
        messages: vec![
            ChatMessage { role:"system".into(), content:"Classify one conversation boundary. Return only schema JSON. Never use assistant context as evidence.".into() },
            ChatMessage { role:"user".into(), content:payload.to_string() },
        ],
        schema: boundary_schema(),
        think: true,
        num_ctx,
        num_predict,
    }
}

pub fn parse_boundary_output(
    response: &str,
    input: &BoundaryInput,
) -> RetrievalResult<BoundaryDecisionV2> {
    let mut value: Value = serde_json::from_str(response).map_err(|error| {
        RetrievalError::CorruptIndex(format!("boundary response_json 不是有效 JSON：{error}"))
    })?;
    let batch = ConsolidationInputBatch {
        batch_key: "boundary".into(),
        session_id: input.session_id.clone(),
        watermark_before: input.watermark_before,
        from_sequence: input.sequence,
        through_sequence: input.sequence,
        through_event_id: input.event_id.clone(),
        through_event_sha256: input.content_sha256.clone(),
        turn_count: 1,
        char_count: input.content.chars().count(),
        events: vec![crate::consolidation::ConsolidationEvent {
            event_id: input.event_id.clone(),
            turn_id: "boundary".into(),
            sequence: input.sequence,
            role: EventRole::User,
            created_at: input.created_at.clone(),
            content: input.content.clone(),
            content_sha256: input.content_sha256.clone(),
        }],
    };
    let clauses = source_clauses(&batch);
    let map = clauses
        .iter()
        .map(|clause| (clause.clause_id.as_str(), clause))
        .collect();
    resolve_source_refs(&mut value, &map, "$")?;
    let output: ModelBoundaryOutput = serde_json::from_value(value).map_err(|error| {
        RetrievalError::CorruptIndex(format!("boundary response 结构无效：{error}"))
    })?;
    if output.before_event_id != input.event_id {
        return Err(RetrievalError::CorruptIndex(
            "boundary before_event_id 不是当前用户事件".into(),
        ));
    }
    if !matches!(
        output.reason.as_str(),
        "model_topic_shift" | "model_same_topic"
    ) {
        return Err(RetrievalError::CorruptIndex("boundary reason 无效".into()));
    }
    Ok(boundary_decision(
        input,
        output.is_boundary,
        &output.reason,
        Map::from_iter([("llm_ambiguous_resolution".into(), Value::Bool(true))]),
        output.evidence,
        "qwen",
    ))
}

pub fn parse_and_resolve_fact_output(
    response: &str,
    clauses: &[SourceClause],
    assistant_contexts: &[AssistantContext],
) -> RetrievalResult<StructuredConsolidationOutput> {
    let mut value: Value = serde_json::from_str(response).map_err(|error| {
        RetrievalError::CorruptIndex(format!("fact v2 response_json 不是有效 JSON：{error}"))
    })?;
    let clause_map = clauses
        .iter()
        .map(|clause| (clause.clause_id.as_str(), clause))
        .collect::<HashMap<_, _>>();
    let context_ids = assistant_contexts
        .iter()
        .map(|context| context.context_id.as_str())
        .collect::<HashSet<_>>();
    prepare_confirmation_evidence(&mut value, &context_ids)?;
    resolve_source_refs(&mut value, &clause_map, "$")?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| RetrievalError::CorruptIndex("fact v2 响应必须是对象".into()))?;
    object.insert("boundaries".into(), Value::Array(Vec::new()));
    serde_json::from_value(value)
        .map_err(|error| RetrievalError::CorruptIndex(format!("fact v2 响应结构无效：{error}")))
}

pub fn is_deterministic_empty(clauses: &[SourceClause], has_assistant_context: bool) -> bool {
    !clauses.is_empty()
        && clauses
            .iter()
            .all(|clause| clause_is_unambiguously_nonassertive(&clause.text, has_assistant_context))
}

pub fn split_batch(
    batch: &ConsolidationInputBatch,
) -> Option<(ConsolidationInputBatch, ConsolidationInputBatch)> {
    let mut turn_starts = batch
        .events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| (event.role == EventRole::User).then_some(index))
        .collect::<Vec<_>>();
    if turn_starts.len() < 2 {
        return None;
    }
    turn_starts.push(batch.events.len());
    let split_turn = (turn_starts.len() - 1).div_ceil(2);
    let split_index = turn_starts[split_turn];
    let left = child_batch(batch, &batch.events[..split_index], batch.watermark_before)?;
    let right = child_batch(batch, &batch.events[split_index..], left.through_sequence)?;
    Some((left, right))
}

fn child_batch(
    parent: &ConsolidationInputBatch,
    events: &[crate::consolidation::ConsolidationEvent],
    watermark_before: usize,
) -> Option<ConsolidationInputBatch> {
    let end = events
        .iter()
        .rposition(|event| event.role == EventRole::User)?
        + 1;
    let events = &events[..end];
    let first = events.first()?;
    let last = events.last()?;
    let turn_count = events
        .iter()
        .filter(|event| event.role == EventRole::User)
        .count();
    let char_count = events
        .iter()
        .map(|event| event.content.chars().count())
        .sum();
    let key_payload = json!({
        "domain":"hippocampus-fact-batch-v2",
        "session_id":parent.session_id,
        "watermark_before":watermark_before,
        "event_ids":events.iter().map(|event| event.event_id.as_str()).collect::<Vec<_>>(),
        "event_hashes":events.iter().map(|event| event.content_sha256.as_str()).collect::<Vec<_>>(),
    });
    Some(ConsolidationInputBatch {
        batch_key: format!(
            "batch_v2_{}",
            content_sha256(&serde_json::to_string(&key_payload).expect("JSON value serializes"))
        ),
        session_id: parent.session_id.clone(),
        watermark_before,
        from_sequence: first.sequence,
        through_sequence: last.sequence,
        through_event_id: last.event_id.clone(),
        through_event_sha256: last.content_sha256.clone(),
        turn_count,
        char_count,
        events: events.to_vec(),
    })
}

fn split_event_clauses(event_id: &str, content: &str) -> Vec<SourceClause> {
    let chars = content.chars().collect::<Vec<_>>();
    let mut ranges = Vec::new();
    let mut start = 0;
    for (index, character) in chars.iter().enumerate() {
        if matches!(
            character,
            '.' | '。' | '!' | '！' | '?' | '？' | ';' | '；' | '\n' | '\r'
        ) {
            ranges.push((start, index + 1));
            start = index + 1;
        }
    }
    if start < chars.len() {
        ranges.push((start, chars.len()));
    }
    ranges
        .into_iter()
        .filter_map(|(mut start, mut end)| {
            while start < end && chars[start].is_whitespace() {
                start += 1;
            }
            while end > start && chars[end - 1].is_whitespace() {
                end -= 1;
            }
            if start == end {
                return None;
            }
            let text = chars[start..end].iter().collect::<String>();
            Some(source_clause(event_id, start, end, text))
        })
        .collect()
}

fn split_single_clause(clause: &SourceClause) -> Option<(Vec<SourceClause>, Vec<SourceClause>)> {
    let chars = clause.text.chars().collect::<Vec<_>>();
    if chars.len() < 2 {
        return None;
    }
    let midpoint = chars.len() / 2;
    let weak = chars
        .iter()
        .enumerate()
        .filter(|(_, character)| matches!(character, '，' | ',' | '、' | ':' | '：'))
        .map(|(index, _)| index + 1)
        .min_by_key(|index| index.abs_diff(midpoint));
    let (left_end, right_start) =
        if let Some(split) = weak.filter(|split| *split > 0 && *split < chars.len()) {
            (split, split)
        } else {
            if chars.len() <= 64 {
                return None;
            }
            let overlap = 32.min(midpoint).min(chars.len().saturating_sub(midpoint));
            (midpoint + overlap, midpoint.saturating_sub(overlap))
        };
    let left_text = chars[..left_end].iter().collect::<String>();
    let right_text = chars[right_start..].iter().collect::<String>();
    if left_text == clause.text || right_text == clause.text {
        return None;
    }
    let left = source_clause(
        &clause.event_id,
        clause.start_char,
        clause.start_char + left_end,
        left_text,
    );
    let right = source_clause(
        &clause.event_id,
        clause.start_char + right_start,
        clause.end_char,
        right_text,
    );
    Some((vec![left], vec![right]))
}

fn source_clause(event_id: &str, start: usize, end: usize, text: String) -> SourceClause {
    SourceClause {
        clause_id: deterministic_id("hippocampus-clause-v1", event_id, start, end, &text),
        span_id: provenance_span_id(event_id, start, end, &text),
        event_id: event_id.to_owned(),
        start_char: start,
        end_char: end,
        text,
    }
}

fn deterministic_id(domain: &str, event_id: &str, start: usize, end: usize, text: &str) -> String {
    let mut hasher = Sha256::new();
    for value in [domain.as_bytes(), event_id.as_bytes(), text.as_bytes()] {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value);
    }
    hasher.update((start as u64).to_be_bytes());
    hasher.update((end as u64).to_be_bytes());
    let prefix = if domain.contains("clause") {
        "clause"
    } else if domain.contains("context") {
        "context"
    } else {
        "span"
    };
    format!("{prefix}_{:x}", hasher.finalize())
}

fn compact_candidates(
    candidates: &ConsolidationCandidateSnapshot,
    clauses: &[SourceClause],
) -> Value {
    let current = clauses
        .iter()
        .map(|clause| clause.text.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    let mut ranked_entities = candidates.entities.iter().collect::<Vec<_>>();
    ranked_entities.sort_by_key(|entity| {
        let matched = current.contains(&entity.canonical_name.to_lowercase())
            || entity
                .aliases
                .iter()
                .any(|alias| current.contains(&alias.text.to_lowercase()));
        std::cmp::Reverse(usize::from(matched))
    });
    let entities = ranked_entities
        .into_iter()
        .take(12)
        .map(|entity| {
            json!({
                "entity_id": entity.entity_id,
                "kind": entity.kind,
                "name": entity.canonical_name,
                "disambiguation": entity.disambiguation,
                "aliases": entity.aliases.iter().map(|alias| alias.text.as_str()).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let mut ranked_claims = candidates.claims.iter().collect::<Vec<_>>();
    ranked_claims.sort_by_key(|claim| {
        let matched = current.contains(&claim.normalized_relation.to_lowercase())
            || current.contains(&claim.predicate_key.to_lowercase())
            || claim
                .object_text
                .as_ref()
                .is_some_and(|object| current.contains(&object.to_lowercase()));
        std::cmp::Reverse(usize::from(matched))
    });
    let claims = ranked_claims
        .into_iter()
        .take(16)
        .map(|claim| {
            json!({
                "claim_id": claim.claim_id,
                "subject_entity_id": claim.subject_entity_id,
                "predicate_key": claim.predicate_key,
                "relation": claim.normalized_relation,
                "object_kind": claim.object_kind,
                "object_text": claim.object_text,
                "object_entity_id": claim.object_entity_id,
                "normalized_object": claim.normalized_object,
                "polarity": claim.polarity,
                "certainty": claim.certainty,
                "state": claim.state,
            })
        })
        .collect::<Vec<_>>();
    json!({"entities": entities, "claims": claims})
}

fn fact_schema() -> Value {
    let mut schema = structured_consolidation_schema();
    let object = schema.as_object_mut().expect("consolidation schema object");
    object.insert("required".into(), json!(["entities", "claims"]));
    object
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .expect("schema properties")
        .remove("boundaries");
    let defs = object
        .get_mut("$defs")
        .and_then(Value::as_object_mut)
        .expect("schema defs");
    defs.insert(
        "quote".into(),
        json!({
            "type":"object",
            "additionalProperties":false,
            "required":["clause_id","event_id","start_char","end_char","text"],
            "properties":{
                "clause_id":{"type":"string","minLength":1,"maxLength":96},
                "event_id":{"type":"string","minLength":1,"maxLength":128},
                "start_char":{"type":"integer","minimum":0},
                "end_char":{"type":"integer","minimum":0},
                "text":{"type":"string","minLength":1,"maxLength":4096}
            }
        }),
    );
    if let Some(evidence) = defs.get_mut("evidence").and_then(Value::as_object_mut) {
        evidence.insert(
            "required".into(),
            json!([
                "kind",
                "quote",
                "subject_span",
                "relation_span",
                "object_span",
                "speech_act_span",
                "context_id"
            ]),
        );
        if let Some(properties) = evidence
            .get_mut("properties")
            .and_then(Value::as_object_mut)
        {
            for name in ["subject_span", "relation_span", "object_span"] {
                properties.insert(
                    name.into(),
                    json!({"anyOf":[{"$ref":"#/$defs/quote"},{"type":"null"}]}),
                );
            }
            properties.insert(
                "context_id".into(),
                json!({"type":["string","null"],"maxLength":96}),
            );
        }
    }
    schema
}

fn prepare_confirmation_evidence(
    value: &mut Value,
    context_ids: &HashSet<&str>,
) -> RetrievalResult<()> {
    let claims = value
        .get_mut("claims")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| RetrievalError::CorruptIndex("$.claims 必须是数组".into()))?;
    for (claim_index, claim) in claims.iter_mut().enumerate() {
        let evidence = claim
            .get_mut("evidence")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                RetrievalError::CorruptIndex(format!("$.claims[{claim_index}].evidence 必须是数组"))
            })?;
        for (evidence_index, item) in evidence.iter_mut().enumerate() {
            let path = format!("$.claims[{claim_index}].evidence[{evidence_index}]");
            let object = item
                .as_object_mut()
                .ok_or_else(|| RetrievalError::CorruptIndex(format!("{path} 必须是对象")))?;
            let kind: ConsolidationEvidenceKind =
                serde_json::from_value(object.get("kind").cloned().unwrap_or(Value::Null))
                    .map_err(|error| {
                        RetrievalError::CorruptIndex(format!("{path}.kind 无效：{error}"))
                    })?;
            let context = object.get("context_id").cloned().unwrap_or(Value::Null);
            let context = context.as_str();
            if let Some(context) = context {
                if !context_ids.contains(context) {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "{path}.context_id 不在 assistant context catalog"
                    )));
                }
                if !matches!(
                    kind,
                    ConsolidationEvidenceKind::UserConfirmation
                        | ConsolidationEvidenceKind::Correction
                ) {
                    return Err(RetrievalError::CorruptIndex(format!(
                        "{path}.context_id 只允许确认或纠正证据使用"
                    )));
                }
            }
            let components = ["subject_span", "relation_span", "object_span"];
            let populated = components
                .iter()
                .filter(|component| {
                    object
                        .get(**component)
                        .is_some_and(|value| !value.is_null())
                })
                .count();
            if populated != 0 && populated != components.len() {
                return Err(RetrievalError::CorruptIndex(format!(
                    "{path} 的 subject/relation/object 必须全部提供或全部为 null"
                )));
            }
            if populated == 0
                && !matches!(
                    kind,
                    ConsolidationEvidenceKind::UserConfirmation
                        | ConsolidationEvidenceKind::Correction
                )
            {
                return Err(RetrievalError::CorruptIndex(format!(
                    "{path} 普通断言必须包含用户原文中的 subject/relation/object span"
                )));
            }
            if matches!(
                kind,
                ConsolidationEvidenceKind::UserConfirmation | ConsolidationEvidenceKind::Correction
            ) && object.get("speech_act_span").is_none_or(Value::is_null)
            {
                return Err(RetrievalError::CorruptIndex(format!(
                    "{path} 确认或纠正证据必须包含用户 speech-act span"
                )));
            }
        }
    }
    Ok(())
}

fn resolve_source_refs(
    value: &mut Value,
    clauses: &HashMap<&str, &SourceClause>,
    path: &str,
) -> RetrievalResult<()> {
    match value {
        Value::Array(values) => {
            for (index, item) in values.iter_mut().enumerate() {
                resolve_source_refs(item, clauses, &format!("{path}[{index}]"))?;
            }
        }
        Value::Object(object) if looks_like_source_ref(object) => {
            let clause_id = required_str(object, "clause_id", path)?;
            let event_id = required_str(object, "event_id", path)?;
            let text = required_str(object, "text", path)?;
            let start = required_usize(object, "start_char", path)?;
            let end = required_usize(object, "end_char", path)?;
            let clause = clauses.get(clause_id).ok_or_else(|| {
                RetrievalError::CorruptIndex(format!("{path}.clause_id 不在用户 source catalog"))
            })?;
            if event_id != clause.event_id
                || start < clause.start_char
                || end > clause.end_char
                || start >= end
            {
                return Err(RetrievalError::CorruptIndex(format!(
                    "{path} 偏移不属于指定用户 clause"
                )));
            }
            let relative_start = start - clause.start_char;
            let relative_end = end - clause.start_char;
            let actual = clause
                .text
                .chars()
                .skip(relative_start)
                .take(relative_end - relative_start)
                .collect::<String>();
            if actual != text {
                return Err(RetrievalError::CorruptIndex(format!(
                    "{path}.text 与用户原文切片不一致"
                )));
            }
            let event_id = event_id.to_owned();
            let hash = content_sha256(text);
            *object = Map::from_iter([
                ("event_id".into(), Value::String(event_id)),
                ("start_char".into(), json!(start)),
                ("end_char".into(), json!(end)),
                ("content_sha256".into(), Value::String(hash)),
            ]);
        }
        Value::Object(object) => {
            for (key, item) in object.iter_mut() {
                resolve_source_refs(item, clauses, &format!("{path}.{key}"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn looks_like_source_ref(object: &Map<String, Value>) -> bool {
    object.contains_key("clause_id")
        || (object.contains_key("event_id")
            && object.contains_key("start_char")
            && object.contains_key("end_char")
            && object.contains_key("text"))
}

fn required_str<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    path: &str,
) -> RetrievalResult<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| RetrievalError::CorruptIndex(format!("{path}.{key} 必须是非空字符串")))
}

fn required_usize(object: &Map<String, Value>, key: &str, path: &str) -> RetrievalResult<usize> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| RetrievalError::CorruptIndex(format!("{path}.{key} 必须是非负整数")))
}

fn clause_is_unambiguously_nonassertive(text: &str, has_assistant_context: bool) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    let normalized = trimmed
        .trim_matches(|character: char| {
            character.is_whitespace() || matches!(character, '。' | '.' | '！' | '!' | '？' | '?')
        })
        .to_lowercase();
    let confirmation = [
        "对", "是", "是的", "没错", "不是", "不对", "yes", "no", "correct",
    ];
    if has_assistant_context && confirmation.contains(&normalized.as_str()) {
        return false;
    }
    let greetings = [
        "你好",
        "您好",
        "嗨",
        "谢谢",
        "感谢",
        "再见",
        "hi",
        "hello",
        "thanks",
        "thank you",
        "bye",
    ];
    if greetings.contains(&normalized.as_str()) {
        return true;
    }
    (trimmed.ends_with('?') || trimmed.ends_with('？'))
        && !trimmed.contains(['，', ',', '；', ';', '。', '.'])
}

fn boundary_decision(
    input: &BoundaryInput,
    is_boundary: bool,
    reason: &str,
    signals: Map<String, Value>,
    evidence: Vec<Value>,
    generator: &str,
) -> BoundaryDecisionV2 {
    BoundaryDecisionV2 {
        before_event_id: input.event_id.clone(),
        is_boundary,
        reason: reason.into(),
        evidence_json: Value::Array(evidence).to_string(),
        signals_json: Value::Object(signals).to_string(),
        generator: generator.into(),
    }
}

fn boundary_schema() -> Value {
    json!({
        "type":"object",
        "additionalProperties":false,
        "required":["before_event_id","is_boundary","reason","evidence"],
        "properties":{
            "before_event_id":{"type":"string","minLength":1,"maxLength":128},
            "is_boundary":{"type":"boolean"},
            "reason":{"enum":["model_topic_shift","model_same_topic"]},
            "evidence":{"type":"array","maxItems":4,"items":{
                "type":"object","additionalProperties":false,
                "required":["clause_id","event_id","start_char","end_char","text"],
                "properties":{
                    "clause_id":{"type":"string","minLength":1,"maxLength":96},
                    "event_id":{"type":"string","minLength":1,"maxLength":128},
                    "start_char":{"type":"integer","minimum":0},
                    "end_char":{"type":"integer","minimum":0},
                    "text":{"type":"string","minLength":1,"maxLength":4096}
                }
            }}
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidation::{ConsolidationEvent, ConsolidationInputBatch};

    fn batch(content: &str) -> ConsolidationInputBatch {
        ConsolidationInputBatch {
            batch_key: "batch".into(),
            session_id: "session".into(),
            watermark_before: 0,
            from_sequence: 1,
            through_sequence: 1,
            through_event_id: "user-event".into(),
            through_event_sha256: content_sha256(content),
            turn_count: 1,
            char_count: content.chars().count(),
            events: vec![ConsolidationEvent {
                event_id: "user-event".into(),
                turn_id: "turn".into(),
                sequence: 1,
                role: EventRole::User,
                created_at: "2026-08-21T00:00:00Z".into(),
                content: content.into(),
                content_sha256: content_sha256(content),
            }],
        }
    }

    #[test]
    fn clauses_have_deterministic_host_ids_and_unicode_offsets() {
        let first = source_clauses(&batch("我住在北京。🙂很好！"));
        let second = source_clauses(&batch("我住在北京。🙂很好！"));
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
        assert_eq!((first[1].start_char, first[1].end_char), (6, 10));
        assert!(
            first
                .iter()
                .all(|item| item.clause_id.starts_with("clause_"))
        );
        assert!(first.iter().all(|item| item.span_id.starts_with("span_")));
    }

    #[test]
    fn fast_path_is_conservative_about_confirmations() {
        assert!(is_deterministic_empty(
            &source_clauses(&batch("你好！")),
            false
        ));
        assert!(is_deterministic_empty(
            &source_clauses(&batch("天气好吗？")),
            false
        ));
        assert!(!is_deterministic_empty(
            &source_clauses(&batch("对。")),
            true
        ));
        assert!(!is_deterministic_empty(
            &source_clauses(&batch("我住在北京。")),
            false
        ));
    }

    #[test]
    fn source_ref_is_verified_and_hash_is_computed_by_rust() {
        let clauses = source_clauses(&batch("Alice lives in Paris."));
        let clause = &clauses[0];
        let mut value = json!({
            "clause_id": clause.clause_id,
            "event_id": clause.event_id,
            "start_char": 15,
            "end_char": 20,
            "text": "Paris"
        });
        let map = HashMap::from([(clause.clause_id.as_str(), clause)]);
        resolve_source_refs(&mut value, &map, "$.quote").unwrap();
        assert_eq!(value["content_sha256"], content_sha256("Paris"));
        assert!(value.get("clause_id").is_none());
        value["content_sha256"] = Value::Null;
    }

    #[test]
    fn assistant_event_cannot_become_source_evidence() {
        let clauses = source_clauses(&batch("用户原文。"));
        let clause = &clauses[0];
        let mut value = json!({
            "clause_id": clause.clause_id,
            "event_id": "assistant-event",
            "start_char": 0,
            "end_char": 4,
            "text": "助手原文"
        });
        let map = HashMap::from([(clause.clause_id.as_str(), clause)]);
        assert!(resolve_source_refs(&mut value, &map, "$.quote").is_err());
    }
}
