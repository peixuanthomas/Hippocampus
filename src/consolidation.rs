use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
        let from_sequence = attempt_usize_to_sql(record.from_sequence, "from_sequence")?;
        let through_sequence = attempt_usize_to_sql(record.through_sequence, "through_sequence")?;
        let input_tokens = record
            .input_tokens
            .map(|value| attempt_u64_to_sql(value, "input_tokens"))
            .transpose()?;
        let output_tokens = record
            .output_tokens
            .map(|value| attempt_u64_to_sql(value, "output_tokens"))
            .transpose()?;
        let latency_ms = attempt_u64_to_sql(record.latency_ms, "latency_ms")?;
        let input_event_ids = canonical_string_array(&record.input_event_ids)?;
        let input_event_hashes = canonical_string_array(&record.input_event_hashes)?;

        let connection = self.open_connection()?;
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
            )
            .map_err(|error| self.database_error(error))?;
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
            validate_json("response_json", response)?;
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

fn canonical_string_array(values: &[String]) -> RetrievalResult<String> {
    serde_json::to_string(values).map_err(|_| invalid_attempt("无法编码输入事件来源数组"))
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
        let mut malformed_response = base.clone();
        malformed_response.response_json = Some("[".into());
        malformed_response.response_sha256 = malformed_response
            .response_json
            .as_ref()
            .map(|response| sha256_bytes(response.as_bytes()));
        invalid.push(malformed_response);
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
    fn malformed_request_and_response_json_are_rejected_when_read() {
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
        assert!(matches!(
            response_store.consolidation_attempts("session"),
            Err(RetrievalError::CorruptIndex(message))
                if message == "巩固失败记录无效：response_json 不是有效 JSON"
        ));
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
}
