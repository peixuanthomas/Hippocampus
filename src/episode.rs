//! Deterministic, auditable episode planning.
//!
//! This module deliberately contains no database or model calls.  Its input is
//! a validated projection of immutable message documents and consolidation
//! state, and its output contains only direct original-message members.

use std::collections::{BTreeMap, BTreeSet};

use chrono::DateTime;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model::{EventRole, SourceSpan};

pub const EPISODE_ALGORITHM_VERSION: u32 = 1;
pub const ENTITY_JACCARD_DISTANCE_THRESHOLD: f64 = 0.50;
pub const EMBEDDING_COSINE_SIMILARITY_THRESHOLD: f64 = 0.60;
pub const SOFT_BOUNDARY_VOTE_THRESHOLD: usize = 2;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeSignalState {
    True,
    False,
    Abstain,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeGapState {
    FirstMessage,
    GreaterThanThreshold,
    EqualToThreshold,
    BelowThreshold,
    InvalidOrOutOfOrder,
    MissingTimestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EpisodeSignal {
    pub name: String,
    pub state: EpisodeSignalState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EpisodeBoundaryDecision {
    pub before_event_id: String,
    pub is_boundary: bool,
    pub gap: EpisodeGapState,
    pub hard_signals: Vec<EpisodeSignal>,
    pub soft_signals: Vec<EpisodeSignal>,
    pub soft_true_votes: usize,
    pub input_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EpisodeMember {
    pub document_id: String,
    pub event_id: String,
    pub sequence: u64,
    pub role: EventRole,
    pub span: SourceSpan,
    pub content_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EpisodeDocument {
    pub document_id: String,
    pub session_id: String,
    pub granularity: String,
    pub source_sha256: String,
    pub start_sequence: u64,
    pub end_sequence: u64,
    pub members: Vec<EpisodeMember>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EpisodeMaterializationReport {
    pub session_id: String,
    pub source_session_sha256: String,
    pub plan_input_sha256: String,
    pub episode_documents: Vec<EpisodeDocument>,
    pub session_document_id: Option<String>,
    pub boundary_decisions: Vec<EpisodeBoundaryDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpisodeInputMessage {
    pub member: EpisodeMember,
    pub created_at: String,
    #[serde(default)]
    pub resolved_entity_ids: BTreeSet<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct EpisodeBoundarySuggestion {
    pub before_event_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpisodePlanInput {
    pub session_id: String,
    pub source_session_sha256: String,
    pub gap_minutes: u64,
    pub consolidation_watermark: Option<u64>,
    pub messages: Vec<EpisodeInputMessage>,
    pub suggestions: Vec<EpisodeBoundarySuggestion>,
}

pub fn plan_episodes(input: &EpisodePlanInput) -> Result<EpisodeMaterializationReport, String> {
    if input.session_id.trim().is_empty() || !is_sha256(&input.source_session_sha256) {
        return Err("episode plan input contains an invalid session identity".into());
    }
    let mut messages = input.messages.clone();
    messages.sort_by(|left, right| {
        left.member
            .sequence
            .cmp(&right.member.sequence)
            .then_with(|| left.member.event_id.cmp(&right.member.event_id))
    });
    if messages.iter().any(|message| {
        message.member.document_id.trim().is_empty()
            || message.member.event_id.trim().is_empty()
            || message.member.span.event_id != message.member.event_id
            || message.member.span.start_char != 0
            || message.member.span.end_char == 0
            || !is_sha256(&message.member.content_sha256)
            || !matches!(message.member.role, EventRole::User | EventRole::Assistant)
            || message
                .embedding
                .as_ref()
                .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    }) {
        return Err("episode plan contains a non-message or invalid direct member".into());
    }
    if messages
        .windows(2)
        .any(|items| items[0].member.sequence == items[1].member.sequence)
    {
        return Err("episode plan contains duplicate message sequences".into());
    }

    let suggestion_map = dedupe_suggestions(&input.suggestions);
    let plan_input_sha256 = plan_hash(input, &messages, &suggestion_map);
    let mut decisions = Vec::new();
    let mut episodes: Vec<Vec<EpisodeMember>> = Vec::new();
    let mut current = Vec::new();
    let mut previous_eligible: Option<&EpisodeInputMessage> = None;
    let mut previous_user: Option<&EpisodeInputMessage> = None;

    for message in &messages {
        let starts_candidate = current.is_empty() || message.member.role == EventRole::User;
        if starts_candidate {
            let decision = decide_boundary(
                input,
                message,
                previous_eligible,
                previous_user,
                suggestion_map.get(&message.member.event_id),
            );
            if decision.is_boundary && !current.is_empty() {
                episodes.push(std::mem::take(&mut current));
            }
            decisions.push(decision);
        }
        current.push(message.member.clone());
        previous_eligible = Some(message);
        if message.member.role == EventRole::User {
            previous_user = Some(message);
        }
    }
    if !current.is_empty() {
        episodes.push(current);
    }
    let episode_documents = episodes
        .into_iter()
        .map(|members| make_document(&input.session_id, "episode", members))
        .collect::<Result<Vec<_>, _>>()?;
    let session_document_id =
        (!messages.is_empty()).then(|| session_document_id(&input.session_id));
    Ok(EpisodeMaterializationReport {
        session_id: input.session_id.clone(),
        source_session_sha256: input.source_session_sha256.clone(),
        plan_input_sha256,
        episode_documents,
        session_document_id,
        boundary_decisions: decisions,
    })
}

pub fn session_document_id(session_id: &str) -> String {
    format!(
        "session_{}",
        hash_fields(b"hippocampus-session-document-v1", &[session_id.as_bytes()])
    )
}

fn make_document(
    session_id: &str,
    granularity: &str,
    members: Vec<EpisodeMember>,
) -> Result<EpisodeDocument, String> {
    let first = members
        .first()
        .ok_or_else(|| "empty aggregate members".to_string())?;
    let document_id = if granularity == "episode" {
        format!(
            "episode_{}",
            hash_fields(
                b"hippocampus-episode-document-v1",
                &[session_id.as_bytes(), first.event_id.as_bytes()],
            )
        )
    } else {
        session_document_id(session_id)
    };
    let source_sha256 = members_hash(granularity, session_id, &members);
    Ok(EpisodeDocument {
        document_id,
        session_id: session_id.to_owned(),
        granularity: granularity.to_owned(),
        source_sha256,
        start_sequence: first.sequence,
        end_sequence: members.last().expect("nonempty checked").sequence,
        members,
    })
}

fn decide_boundary(
    input: &EpisodePlanInput,
    current: &EpisodeInputMessage,
    previous_eligible: Option<&EpisodeInputMessage>,
    previous_user: Option<&EpisodeInputMessage>,
    suggestions: Option<&BTreeSet<String>>,
) -> EpisodeBoundaryDecision {
    let (gap, time_hard) = gap_signal(previous_eligible, current, input.gap_minutes);
    let explicit = suggestion_state(suggestions, "explicit_topic_transition");
    let model = model_signal(
        suggestions,
        current.member.sequence,
        input.consolidation_watermark,
    );
    let entity = entity_signal(previous_user, current);
    let embedding = embedding_signal(previous_user, current);
    let hard_signals = vec![
        EpisodeSignal {
            name: "session_start".into(),
            state: if previous_eligible.is_none() {
                EpisodeSignalState::True
            } else {
                EpisodeSignalState::False
            },
        },
        EpisodeSignal {
            name: "time_gap".into(),
            state: time_hard,
        },
        EpisodeSignal {
            name: "explicit_topic_transition".into(),
            state: explicit,
        },
    ];
    let soft_signals = vec![
        EpisodeSignal {
            name: "entity_jaccard_distance".into(),
            state: entity,
        },
        EpisodeSignal {
            name: "embedding_cosine_similarity".into(),
            state: embedding,
        },
        EpisodeSignal {
            name: "model_topic_shift".into(),
            state: model,
        },
    ];
    let votes = soft_signals
        .iter()
        .filter(|signal| signal.state == EpisodeSignalState::True)
        .count();
    let boundary = hard_signals
        .iter()
        .any(|signal| signal.state == EpisodeSignalState::True)
        || votes >= SOFT_BOUNDARY_VOTE_THRESHOLD;
    let input_sha256 = decision_hash(
        input,
        current,
        previous_eligible,
        previous_user,
        suggestions,
        gap,
        &hard_signals,
        &soft_signals,
        votes,
    );
    EpisodeBoundaryDecision {
        before_event_id: current.member.event_id.clone(),
        is_boundary: boundary,
        gap,
        hard_signals,
        soft_signals,
        soft_true_votes: votes,
        input_sha256,
    }
}

fn gap_signal(
    previous: Option<&EpisodeInputMessage>,
    current: &EpisodeInputMessage,
    gap_minutes: u64,
) -> (EpisodeGapState, EpisodeSignalState) {
    let Some(previous) = previous else {
        return (EpisodeGapState::FirstMessage, EpisodeSignalState::Abstain);
    };
    let Ok(previous_time) = DateTime::parse_from_rfc3339(&previous.created_at) else {
        return (
            EpisodeGapState::MissingTimestamp,
            EpisodeSignalState::Abstain,
        );
    };
    let Ok(current_time) = DateTime::parse_from_rfc3339(&current.created_at) else {
        return (
            EpisodeGapState::MissingTimestamp,
            EpisodeSignalState::Abstain,
        );
    };
    let seconds = current_time
        .signed_duration_since(previous_time)
        .num_seconds();
    if seconds < 0 {
        return (
            EpisodeGapState::InvalidOrOutOfOrder,
            EpisodeSignalState::Abstain,
        );
    }
    let threshold = i64::try_from(gap_minutes).unwrap_or(i64::MAX / 60) * 60;
    if seconds > threshold {
        (
            EpisodeGapState::GreaterThanThreshold,
            EpisodeSignalState::True,
        )
    } else if seconds == threshold {
        (EpisodeGapState::EqualToThreshold, EpisodeSignalState::False)
    } else {
        (EpisodeGapState::BelowThreshold, EpisodeSignalState::False)
    }
}

fn suggestion_state(suggestions: Option<&BTreeSet<String>>, wanted: &str) -> EpisodeSignalState {
    if suggestions.is_some_and(|values| values.contains(wanted)) {
        EpisodeSignalState::True
    } else {
        EpisodeSignalState::False
    }
}

fn model_signal(
    suggestions: Option<&BTreeSet<String>>,
    sequence: u64,
    watermark: Option<u64>,
) -> EpisodeSignalState {
    if suggestions.is_some_and(|values| values.contains("model_topic_shift")) {
        EpisodeSignalState::True
    } else if watermark.is_some_and(|through| sequence <= through) {
        EpisodeSignalState::False
    } else {
        EpisodeSignalState::Abstain
    }
}

fn entity_signal(
    previous: Option<&EpisodeInputMessage>,
    current: &EpisodeInputMessage,
) -> EpisodeSignalState {
    let Some(previous) = previous else {
        return EpisodeSignalState::Abstain;
    };
    if previous.resolved_entity_ids.is_empty() || current.resolved_entity_ids.is_empty() {
        return EpisodeSignalState::Abstain;
    }
    let intersection = previous
        .resolved_entity_ids
        .intersection(&current.resolved_entity_ids)
        .count();
    let union = previous
        .resolved_entity_ids
        .union(&current.resolved_entity_ids)
        .count();
    let distance = 1.0 - intersection as f64 / union as f64;
    if distance >= ENTITY_JACCARD_DISTANCE_THRESHOLD {
        EpisodeSignalState::True
    } else {
        EpisodeSignalState::False
    }
}

fn embedding_signal(
    previous: Option<&EpisodeInputMessage>,
    current: &EpisodeInputMessage,
) -> EpisodeSignalState {
    let Some(previous) = previous else {
        return EpisodeSignalState::Abstain;
    };
    let (Some(left), Some(right)) = (&previous.embedding, &current.embedding) else {
        return EpisodeSignalState::Abstain;
    };
    if left.is_empty()
        || left.len() != right.len()
        || left.iter().chain(right).any(|value| !value.is_finite())
    {
        return EpisodeSignalState::Abstain;
    }
    let dot: f64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let left_norm: f64 = left.iter().map(|value| f64::from(*value).powi(2)).sum();
    let right_norm: f64 = right.iter().map(|value| f64::from(*value).powi(2)).sum();
    if left_norm <= 0.0 || right_norm <= 0.0 {
        return EpisodeSignalState::Abstain;
    }
    let cosine = dot / (left_norm.sqrt() * right_norm.sqrt());
    if !cosine.is_finite() {
        EpisodeSignalState::Abstain
    } else if cosine < EMBEDDING_COSINE_SIMILARITY_THRESHOLD {
        EpisodeSignalState::True
    } else {
        EpisodeSignalState::False
    }
}

fn dedupe_suggestions(values: &[EpisodeBoundarySuggestion]) -> BTreeMap<String, BTreeSet<String>> {
    let mut result = BTreeMap::new();
    for value in values {
        if matches!(
            value.reason.as_str(),
            "explicit_topic_transition" | "model_topic_shift"
        ) {
            result
                .entry(value.before_event_id.clone())
                .or_insert_with(BTreeSet::new)
                .insert(value.reason.clone());
        }
    }
    result
}

fn members_hash(granularity: &str, session_id: &str, members: &[EpisodeMember]) -> String {
    let mut hasher = Sha256::new();
    hash_part(&mut hasher, b"hippocampus-aggregate-source-v1");
    hash_part(&mut hasher, granularity.as_bytes());
    hash_part(&mut hasher, session_id.as_bytes());
    hash_part(&mut hasher, &(members.len() as u64).to_le_bytes());
    for member in members {
        for value in [
            member.document_id.as_bytes(),
            member.event_id.as_bytes(),
            &member.sequence.to_le_bytes(),
            member.role.as_str().as_bytes(),
            &u64::try_from(member.span.start_char)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
            &u64::try_from(member.span.end_char)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
            member.content_sha256.as_bytes(),
        ] {
            hash_part(&mut hasher, value);
        }
    }
    format!("{:x}", hasher.finalize())
}

fn plan_hash(
    input: &EpisodePlanInput,
    messages: &[EpisodeInputMessage],
    suggestions: &BTreeMap<String, BTreeSet<String>>,
) -> String {
    canonical_input_hash(
        b"hippocampus-episode-plan-input-v2",
        &input.session_id,
        Some(&input.source_session_sha256),
        input.gap_minutes,
        input.consolidation_watermark,
        messages,
        suggestions,
    )
}

#[allow(clippy::too_many_arguments)]
fn decision_hash(
    input: &EpisodePlanInput,
    current: &EpisodeInputMessage,
    previous_eligible: Option<&EpisodeInputMessage>,
    previous_user: Option<&EpisodeInputMessage>,
    suggestions: Option<&BTreeSet<String>>,
    gap: EpisodeGapState,
    hard: &[EpisodeSignal],
    soft: &[EpisodeSignal],
    votes: usize,
) -> String {
    let mut encoder = CanonicalEncoder::new(b"hippocampus-episode-boundary-decision-v2");
    encoder.text("session_id", &input.session_id);
    encoder.u64("gap_minutes", input.gap_minutes);
    encoder.option_u64("watermark", input.consolidation_watermark);
    encoder.message("current", current);
    encoder.option_message("previous_eligible", previous_eligible);
    encoder.option_message("previous_user", previous_user);
    encoder.string_set("suggestion_reasons", suggestions);
    encoder.byte("output_gap", gap_tag(gap));
    encoder.signals("hard", hard);
    encoder.signals("soft", soft);
    encoder.u64(
        "soft_true_votes",
        u64::try_from(votes).expect("usize fits u64"),
    );
    encoder.finish()
}

pub(crate) fn ledger_snapshot_hash(
    session_id: &str,
    watermark: Option<u64>,
    messages: &[EpisodeInputMessage],
    suggestions: &[EpisodeBoundarySuggestion],
) -> String {
    let mut ordered = messages.to_vec();
    ordered.sort_by(|left, right| {
        left.member
            .sequence
            .cmp(&right.member.sequence)
            .then_with(|| left.member.event_id.cmp(&right.member.event_id))
    });
    let suggestions = dedupe_suggestions(suggestions);
    canonical_input_hash(
        b"hippocampus-episode-ledger-snapshot-v2",
        session_id,
        None,
        0,
        watermark,
        &ordered,
        &suggestions,
    )
}

fn canonical_input_hash(
    domain: &[u8],
    session_id: &str,
    source_hash: Option<&str>,
    gap_minutes: u64,
    watermark: Option<u64>,
    messages: &[EpisodeInputMessage],
    suggestions: &BTreeMap<String, BTreeSet<String>>,
) -> String {
    let mut encoder = CanonicalEncoder::new(domain);
    encoder.text("session_id", session_id);
    encoder.option_text("source_session_sha256", source_hash);
    encoder.u64("gap_minutes", gap_minutes);
    encoder.option_u64("watermark", watermark);
    encoder.messages("messages", messages);
    encoder.suggestions("suggestions", suggestions);
    encoder.finish()
}

struct CanonicalEncoder(Sha256);
impl CanonicalEncoder {
    fn new(domain: &[u8]) -> Self {
        let mut hash = Sha256::new();
        hash_part(&mut hash, domain);
        Self(hash)
    }
    fn field(&mut self, tag: &str, value: &[u8]) {
        hash_part(&mut self.0, tag.as_bytes());
        hash_part(&mut self.0, value);
    }
    fn byte(&mut self, tag: &str, value: u8) {
        self.field(tag, &[value]);
    }
    fn u64(&mut self, tag: &str, value: u64) {
        self.field(tag, &value.to_le_bytes());
    }
    fn text(&mut self, tag: &str, value: &str) {
        self.field(tag, value.as_bytes());
    }
    fn option_u64(&mut self, tag: &str, value: Option<u64>) {
        self.byte(&format!("{tag}_present"), u8::from(value.is_some()));
        if let Some(value) = value {
            self.u64(tag, value);
        }
    }
    fn option_text(&mut self, tag: &str, value: Option<&str>) {
        self.byte(&format!("{tag}_present"), u8::from(value.is_some()));
        if let Some(value) = value {
            self.text(tag, value);
        }
    }
    fn member(&mut self, tag: &str, member: &EpisodeMember) {
        self.text(&format!("{tag}.document_id"), &member.document_id);
        self.text(&format!("{tag}.event_id"), &member.event_id);
        self.u64(&format!("{tag}.sequence"), member.sequence);
        self.byte(&format!("{tag}.role"), role_tag(member.role));
        self.text(&format!("{tag}.span.event_id"), &member.span.event_id);
        self.u64(
            &format!("{tag}.span.start"),
            u64::try_from(member.span.start_char).expect("usize fits u64"),
        );
        self.u64(
            &format!("{tag}.span.end"),
            u64::try_from(member.span.end_char).expect("usize fits u64"),
        );
        self.text(&format!("{tag}.content_sha256"), &member.content_sha256);
    }
    fn message(&mut self, tag: &str, message: &EpisodeInputMessage) {
        self.member(tag, &message.member);
        self.text(&format!("{tag}.created_at"), &message.created_at);
        self.u64(
            &format!("{tag}.entities.count"),
            u64::try_from(message.resolved_entity_ids.len()).expect("usize fits u64"),
        );
        for entity in &message.resolved_entity_ids {
            self.text(&format!("{tag}.entity"), entity);
        }
        self.byte(
            &format!("{tag}.embedding.present"),
            u8::from(message.embedding.is_some()),
        );
        if let Some(values) = &message.embedding {
            self.u64(
                &format!("{tag}.embedding.count"),
                u64::try_from(values.len()).expect("usize fits u64"),
            );
            for value in values {
                self.field(
                    &format!("{tag}.embedding.value"),
                    &value.to_bits().to_le_bytes(),
                );
            }
        }
    }
    fn option_message(&mut self, tag: &str, message: Option<&EpisodeInputMessage>) {
        self.byte(&format!("{tag}.present"), u8::from(message.is_some()));
        if let Some(message) = message {
            self.message(tag, message);
        }
    }
    fn messages(&mut self, tag: &str, messages: &[EpisodeInputMessage]) {
        self.u64(
            &format!("{tag}.count"),
            u64::try_from(messages.len()).expect("usize fits u64"),
        );
        for message in messages {
            self.message(&format!("{tag}.item"), message);
        }
    }
    fn string_set(&mut self, tag: &str, values: Option<&BTreeSet<String>>) {
        let values = values.cloned().unwrap_or_default();
        self.u64(
            &format!("{tag}.count"),
            u64::try_from(values.len()).expect("usize fits u64"),
        );
        for value in values {
            self.text(&format!("{tag}.item"), &value);
        }
    }
    fn suggestions(&mut self, tag: &str, suggestions: &BTreeMap<String, BTreeSet<String>>) {
        self.u64(
            &format!("{tag}.count"),
            u64::try_from(suggestions.len()).expect("usize fits u64"),
        );
        for (event, reasons) in suggestions {
            self.text(&format!("{tag}.event"), event);
            self.string_set(&format!("{tag}.reasons"), Some(reasons));
        }
    }
    fn signals(&mut self, tag: &str, signals: &[EpisodeSignal]) {
        self.u64(
            &format!("{tag}.count"),
            u64::try_from(signals.len()).expect("usize fits u64"),
        );
        for signal in signals {
            self.text(&format!("{tag}.name"), &signal.name);
            self.byte(&format!("{tag}.state"), signal_tag(signal.state));
        }
    }
    fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
}
fn role_tag(role: EventRole) -> u8 {
    match role {
        EventRole::System => 0,
        EventRole::User => 1,
        EventRole::Assistant => 2,
    }
}
fn gap_tag(gap: EpisodeGapState) -> u8 {
    match gap {
        EpisodeGapState::FirstMessage => 0,
        EpisodeGapState::GreaterThanThreshold => 1,
        EpisodeGapState::EqualToThreshold => 2,
        EpisodeGapState::BelowThreshold => 3,
        EpisodeGapState::InvalidOrOutOfOrder => 4,
        EpisodeGapState::MissingTimestamp => 5,
    }
}
fn signal_tag(state: EpisodeSignalState) -> u8 {
    match state {
        EpisodeSignalState::True => 1,
        EpisodeSignalState::False => 2,
        EpisodeSignalState::Abstain => 3,
    }
}

fn hash_fields(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hash_part(&mut hasher, domain);
    for field in fields {
        hash_part(&mut hasher, field);
    }
    format!("{:x}", hasher.finalize())
}
fn hash_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}
fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn message(sequence: u64, role: EventRole, time: &str) -> EpisodeInputMessage {
        let id = format!("e{sequence}");
        EpisodeInputMessage {
            member: EpisodeMember {
                document_id: format!("{id}:0:1"),
                event_id: id.clone(),
                sequence,
                role,
                span: SourceSpan {
                    event_id: id,
                    start_char: 0,
                    end_char: 1,
                },
                content_sha256: "0".repeat(64),
            },
            created_at: time.into(),
            resolved_entity_ids: BTreeSet::new(),
            embedding: None,
        }
    }
    fn plan(messages: Vec<EpisodeInputMessage>) -> EpisodeMaterializationReport {
        plan_episodes(&EpisodePlanInput {
            session_id: "s".into(),
            source_session_sha256: "1".repeat(64),
            gap_minutes: 30,
            consolidation_watermark: None,
            messages,
            suggestions: vec![],
        })
        .unwrap()
    }
    #[test]
    fn episode_thresholds_soft_votes_and_assistant_behavior() {
        let mut one = message(1, EventRole::User, "2026-01-01T00:00:00Z");
        one.resolved_entity_ids.insert("a".into());
        one.embedding = Some(vec![1.0, 0.0]);
        let mut two = message(2, EventRole::Assistant, "2026-01-01T00:30:00Z");
        two.embedding = Some(vec![0.0, 1.0]);
        let mut three = message(3, EventRole::User, "2026-01-01T00:30:00Z");
        three.resolved_entity_ids.insert("b".into());
        three.embedding = Some(vec![0.0, 1.0]);
        let report = plan(vec![one, two, three]);
        assert_eq!(report.episode_documents.len(), 2);
        assert_eq!(report.boundary_decisions.len(), 2);
        assert_eq!(
            report.boundary_decisions[1].gap,
            EpisodeGapState::BelowThreshold
        );
    }
    #[test]
    fn episode_ids_are_stable_and_member_hashes_change() {
        let a = message(1, EventRole::User, "2026-01-01T00:00:00Z");
        let b = message(2, EventRole::Assistant, "2026-01-01T00:01:00Z");
        let short = plan(vec![a.clone()]);
        let long = plan(vec![a, b]);
        assert_eq!(
            short.episode_documents[0].document_id,
            long.episode_documents[0].document_id
        );
        assert_ne!(
            short.episode_documents[0].source_sha256,
            long.episode_documents[0].source_sha256
        );
    }

    #[test]
    fn decision_and_ledger_hashes_cover_raw_inputs_canonically() {
        let first = message(1, EventRole::User, "2026-01-01T00:00:00Z");
        let mut second = message(2, EventRole::User, "2026-01-01T00:01:00Z");
        second.embedding = Some(vec![1.0, 0.0]);
        let input = EpisodePlanInput {
            session_id: "s".into(),
            source_session_sha256: "1".repeat(64),
            gap_minutes: 30,
            consolidation_watermark: Some(2),
            messages: vec![first.clone(), second.clone()],
            suggestions: vec![],
        };
        let first_plan = plan_episodes(&input).unwrap();
        let mut changed = input.clone();
        changed.messages[1].created_at = "2026-01-01T00:01:01Z".into();
        assert_ne!(
            first_plan.plan_input_sha256,
            plan_episodes(&changed).unwrap().plan_input_sha256
        );
        assert_ne!(
            first_plan.boundary_decisions[1].input_sha256,
            plan_episodes(&changed).unwrap().boundary_decisions[1].input_sha256
        );
        assert_ne!(
            ledger_snapshot_hash("s", Some(2), &input.messages, &[]),
            ledger_snapshot_hash("s", None, &input.messages, &[])
        );
    }
}
