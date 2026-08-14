use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

use crate::model::{EventRole, Session, TurnStatus, utc_now};
use crate::retrieval::derive_events;

pub const CONTROL_LOG_VERSION: u32 = 1;
const CONTROL_DIRECTORY: &str = ".hippocampus-control";
const MAX_SEGMENT_BYTES: u64 = 16 * 1024;
const MAX_APPEND_ATTEMPTS: usize = 8;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlAction {
    Exclude,
    Restore,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlTargetKind {
    Session,
    Event,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ControlTarget {
    pub kind: ControlTargetKind,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ControlRecord {
    pub version: u32,
    pub sequence: u64,
    pub action: ControlAction,
    pub target_kind: ControlTargetKind,
    pub target_id: String,
    pub target_session_id: Option<String>,
    pub target_content_sha256: Option<String>,
    pub target_created_at: String,
    pub timestamp: String,
    pub previous_record_sha256: Option<String>,
    pub record_sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ControlState {
    excluded_sessions: BTreeSet<String>,
    excluded_events: BTreeSet<String>,
    last_sequence: u64,
    last_record_sha256: Option<String>,
}

impl ControlState {
    /// Stable token binding every derived projection to one exact control-log replay.
    pub fn generation_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"hippocampus-control-generation-v1\0");
        hasher.update(self.last_sequence.to_be_bytes());
        if let Some(hash) = &self.last_record_sha256 {
            hasher.update(hash.as_bytes());
        }
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    pub fn allows_session(&self, session_id: &str) -> bool {
        !self.session_is_excluded(session_id)
    }

    pub fn session_is_excluded(&self, session_id: &str) -> bool {
        self.excluded_sessions.contains(session_id)
    }

    pub fn event_is_excluded(&self, event_id: &str) -> bool {
        self.excluded_events.contains(event_id)
    }

    pub fn allows_event(&self, session_id: &str, event_id: &str) -> bool {
        self.allows_session(session_id) && !self.event_is_excluded(event_id)
    }

    pub fn allows_turn(&self, session_id: &str, turn_id: &str) -> bool {
        self.allows_event(
            session_id,
            &crate::model::event_id(session_id, Some(turn_id), EventRole::User),
        ) && self.allows_event(
            session_id,
            &crate::model::event_id(session_id, Some(turn_id), EventRole::Assistant),
        )
    }

    pub(crate) fn session_has_excluded_event(&self, session: &Session) -> bool {
        session
            .turns
            .iter()
            .any(|turn| !self.allows_turn(&session.id, &turn.id))
    }

    pub fn excluded_sessions(&self) -> impl ExactSizeIterator<Item = &str> {
        self.excluded_sessions.iter().map(String::as_str)
    }

    pub fn excluded_events(&self) -> impl ExactSizeIterator<Item = &str> {
        self.excluded_events.iter().map(String::as_str)
    }

    pub fn last_sequence(&self) -> u64 {
        self.last_sequence
    }

    pub fn last_record_sha256(&self) -> Option<&str> {
        self.last_record_sha256.as_deref()
    }
}

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("control log I/O failed at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("control log is corrupt: {0}")]
    CorruptLog(String),
    #[error("control source is corrupt at {path}: {message}")]
    CorruptSource { path: PathBuf, message: String },
    #[error("invalid control target: {0}")]
    InvalidTarget(String),
    #[error("invalid control transition: {0}")]
    InvalidTransition(String),
    #[error("control serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("canonical control record is too large: {size} bytes exceeds {max} bytes")]
    RecordTooLarge { size: usize, max: u64 },
    #[error("could not lock the session root: {0}")]
    Locking(String),
}

pub type ControlResult<T> = Result<T, ControlError>;

#[derive(Debug, Clone)]
pub struct ControlLog {
    root: PathBuf,
    directory: PathBuf,
}

impl ControlLog {
    pub fn new(root: impl AsRef<Path>) -> ControlResult<Self> {
        let root = root.as_ref().to_path_buf();
        let log = Self {
            directory: root.join(CONTROL_DIRECTORY),
            root,
        };
        log.replay()?;
        Ok(log)
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn replay(&self) -> ControlResult<ControlState> {
        let mut state = ControlState::default();
        match fs::symlink_metadata(&self.directory) {
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => {
                return Err(ControlError::CorruptLog(format!(
                    "{} is not a real directory",
                    self.directory.display()
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(state),
            Err(source) => return Err(self.io(&self.directory, source)),
        }
        let mut segments = Vec::new();
        for entry in
            fs::read_dir(&self.directory).map_err(|source| self.io(&self.directory, source))?
        {
            let entry = entry.map_err(|source| self.io(&self.directory, source))?;
            let file_type = entry
                .file_type()
                .map_err(|source| self.io(&entry.path(), source))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                return Err(ControlError::CorruptLog(
                    "non-UTF-8 segment filename".into(),
                ));
            };
            if is_stale_temp_name(name) {
                if !file_type.is_file() {
                    return Err(ControlError::CorruptLog(format!(
                        "stale temporary entry {name} is not a regular file"
                    )));
                }
                continue;
            }
            let sequence = parse_segment_name(name)?;
            if !file_type.is_file() {
                return Err(ControlError::CorruptLog(format!("unexpected entry {name}")));
            }
            segments.push((sequence, entry.path()));
        }
        segments.sort_by_key(|(sequence, _)| *sequence);
        for (index, (sequence, path)) in segments.into_iter().enumerate() {
            let expected = u64::try_from(index).unwrap_or(u64::MAX) + 1;
            if sequence != expected {
                return Err(ControlError::CorruptLog(format!(
                    "expected segment {expected:020}.json, found {sequence:020}.json"
                )));
            }
            let metadata = fs::symlink_metadata(&path).map_err(|source| self.io(&path, source))?;
            if !metadata.file_type().is_file() {
                return Err(ControlError::CorruptLog(format!(
                    "segment {} is not a regular file",
                    path.display()
                )));
            }
            if metadata.len() > MAX_SEGMENT_BYTES {
                return Err(ControlError::CorruptLog(format!(
                    "segment {} is too large",
                    path.display()
                )));
            }
            let bytes = fs::read(&path).map_err(|source| self.io(&path, source))?;
            let record: ControlRecord = serde_json::from_slice(&bytes).map_err(|error| {
                ControlError::CorruptLog(format!(
                    "segment {} is invalid JSON: {error}",
                    path.display()
                ))
            })?;
            self.validate_record(&record, sequence, &state)?;
            let canonical = canonical_record_bytes(&record)?;
            if bytes != canonical {
                return Err(ControlError::CorruptLog(format!(
                    "segment {} is not canonical JSON",
                    path.display()
                )));
            }
            self.validate_binding(&record)?;
            apply_transition(
                &mut state,
                record.action,
                record.target_kind,
                &record.target_id,
            )
            .map_err(|error| ControlError::CorruptLog(error.to_string()))?;
            state.last_sequence = record.sequence;
            state.last_record_sha256 = Some(record.record_sha256);
        }
        Ok(state)
    }

    pub(crate) fn append(
        &self,
        action: ControlAction,
        target: &ControlTarget,
    ) -> ControlResult<ControlRecord> {
        validate_safe_id(&target.id)?;
        for _ in 0..MAX_APPEND_ATTEMPTS {
            let mut state = self.replay()?;
            let binding = self.resolve_target(target.kind, &target.id)?;
            apply_transition(&mut state, action, target.kind, &target.id)?;
            let mut record = ControlRecord {
                version: CONTROL_LOG_VERSION,
                sequence: state.last_sequence + 1,
                action,
                target_kind: target.kind,
                target_id: target.id.clone(),
                target_session_id: binding.session_id,
                target_content_sha256: binding.content_sha256,
                target_created_at: binding.created_at,
                timestamp: utc_now(),
                previous_record_sha256: state.last_record_sha256,
                record_sha256: String::new(),
            };
            record.record_sha256 = record_hash(&record)?;
            let bytes = canonical_record_bytes(&record)?;
            if bytes.len() as u64 > MAX_SEGMENT_BYTES {
                return Err(ControlError::RecordTooLarge {
                    size: bytes.len(),
                    max: MAX_SEGMENT_BYTES,
                });
            }
            let created_directory = match fs::create_dir(&self.directory) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
                Err(source) => return Err(self.io(&self.directory, source)),
            };
            if created_directory {
                sync_directory(&self.root).map_err(|source| self.io(&self.root, source))?;
            }
            let temporary = self
                .directory
                .join(format!(".control-{}.tmp", Uuid::new_v4().simple()));
            let final_path = self.directory.join(segment_name(record.sequence));
            let preparation = (|| -> ControlResult<()> {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)
                    .map_err(|source| self.io(&temporary, source))?;
                file.write_all(&bytes)
                    .map_err(|source| self.io(&temporary, source))?;
                file.flush().map_err(|source| self.io(&temporary, source))?;
                file.sync_all()
                    .map_err(|source| self.io(&temporary, source))?;
                Ok(())
            })();
            if let Err(error) = preparation {
                let _ = fs::remove_file(&temporary);
                return Err(error);
            }
            match fs::hard_link(&temporary, &final_path) {
                Ok(()) => {
                    let durability = sync_directory(&self.directory)
                        .map_err(|source| self.io(&self.directory, source));
                    let _ = fs::remove_file(&temporary);
                    durability?;
                    return Ok(record);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let _ = fs::remove_file(&temporary);
                    continue;
                }
                Err(source) => {
                    let _ = fs::remove_file(&temporary);
                    return Err(self.io(&final_path, source));
                }
            }
        }
        Err(ControlError::InvalidTransition(
            "control log changed too often while appending".into(),
        ))
    }

    fn validate_record(
        &self,
        record: &ControlRecord,
        sequence: u64,
        state: &ControlState,
    ) -> ControlResult<()> {
        if record.version != CONTROL_LOG_VERSION || record.sequence != sequence {
            return Err(ControlError::CorruptLog(
                "version or sequence mismatch".into(),
            ));
        }
        validate_safe_id(&record.target_id)
            .map_err(|error| ControlError::CorruptLog(error.to_string()))?;
        validate_utc_timestamp(&record.target_created_at, "target_created_at")?;
        validate_utc_timestamp(&record.timestamp, "timestamp")?;
        validate_optional_hash(
            record.previous_record_sha256.as_deref(),
            "previous_record_sha256",
        )?;
        validate_hash(&record.record_sha256, "record_sha256")?;
        if record.previous_record_sha256.as_deref() != state.last_record_sha256.as_deref() {
            return Err(ControlError::CorruptLog(
                "previous record hash mismatch".into(),
            ));
        }
        if record.record_sha256 != record_hash(record)? {
            return Err(ControlError::CorruptLog("record hash mismatch".into()));
        }
        match record.target_kind {
            ControlTargetKind::Session => {
                if record.target_session_id.is_some() || record.target_content_sha256.is_some() {
                    return Err(ControlError::CorruptLog(
                        "session record has event-only binding fields".into(),
                    ));
                }
            }
            ControlTargetKind::Event => {
                let session_id = record.target_session_id.as_deref().ok_or_else(|| {
                    ControlError::CorruptLog("event record lacks session binding".into())
                })?;
                validate_safe_id(session_id)
                    .map_err(|error| ControlError::CorruptLog(error.to_string()))?;
                validate_hash(
                    record.target_content_sha256.as_deref().unwrap_or(""),
                    "target_content_sha256",
                )?;
            }
        }
        Ok(())
    }

    fn validate_binding(&self, record: &ControlRecord) -> ControlResult<()> {
        let binding = self.resolve_target(record.target_kind, &record.target_id)?;
        if binding.session_id != record.target_session_id
            || binding.content_sha256 != record.target_content_sha256
            || binding.created_at != record.target_created_at
        {
            return Err(ControlError::CorruptLog(format!(
                "target binding changed for {}",
                record.target_id
            )));
        }
        Ok(())
    }

    fn resolve_target(&self, kind: ControlTargetKind, id: &str) -> ControlResult<TargetBinding> {
        match kind {
            ControlTargetKind::Session => {
                let session = self.read_exact_session(id)?;
                validate_utc_timestamp(&session.created_at, "session created_at")?;
                Ok(TargetBinding {
                    session_id: None,
                    content_sha256: None,
                    created_at: session.created_at,
                })
            }
            ControlTargetKind::Event => self.resolve_event(id),
        }
    }

    fn read_exact_session(&self, id: &str) -> ControlResult<Session> {
        validate_safe_id(id)?;
        let path = self.root.join(format!("{id}.json"));
        let metadata = fs::symlink_metadata(&path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                ControlError::InvalidTarget(format!("session {id} does not exist"))
            } else {
                self.io(&path, source)
            }
        })?;
        if !metadata.file_type().is_file() {
            return Err(ControlError::InvalidTarget(format!(
                "session {id} is not a regular source file"
            )));
        }
        let mut session = read_session(&path)?;
        session.normalize_legacy_provenance();
        session
            .validate()
            .map_err(|error| ControlError::CorruptSource {
                path: path.clone(),
                message: error.to_string(),
            })?;
        validate_source_timestamps(&session).map_err(|message| ControlError::CorruptSource {
            path: path.clone(),
            message,
        })?;
        if session.id != id
            || path.file_name().and_then(|name| name.to_str())
                != Some(&format!("{}.json", session.id))
        {
            return Err(ControlError::CorruptSource {
                path,
                message: "source filename and session id differ".into(),
            });
        }
        Ok(session)
    }

    fn resolve_event(&self, id: &str) -> ControlResult<TargetBinding> {
        let mut matches = Vec::new();
        for entry in fs::read_dir(&self.root).map_err(|source| self.io(&self.root, source))? {
            let entry = entry.map_err(|source| self.io(&self.root, source))?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                return Err(ControlError::CorruptSource {
                    path,
                    message: "non-UTF-8 source filename".into(),
                });
            };
            let session = self.read_exact_session(stem)?;
            for event in derive_events(&session)
                .into_iter()
                .filter(|event| event.id == id)
            {
                matches.push(event);
            }
        }
        if matches.len() != 1 {
            return Err(ControlError::InvalidTarget(format!(
                "event {id} matched {} raw sources",
                matches.len()
            )));
        }
        let event = matches.pop().expect("one event match");
        if event.role == EventRole::System {
            return Err(ControlError::InvalidTarget(
                "system events cannot be controlled".into(),
            ));
        }
        if event.turn_status == Some(TurnStatus::Pending) {
            return Err(ControlError::InvalidTarget(
                "pending events cannot be controlled".into(),
            ));
        }
        validate_utc_timestamp(&event.created_at, "event created_at")?;
        validate_hash(&event.content_sha256, "event content_sha256")?;
        Ok(TargetBinding {
            session_id: Some(event.session_id),
            content_sha256: Some(event.content_sha256),
            created_at: event.created_at,
        })
    }

    fn io(&self, path: &Path, source: std::io::Error) -> ControlError {
        ControlError::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

struct TargetBinding {
    session_id: Option<String>,
    content_sha256: Option<String>,
    created_at: String,
}

fn apply_transition(
    state: &mut ControlState,
    action: ControlAction,
    kind: ControlTargetKind,
    id: &str,
) -> ControlResult<()> {
    let targets = match kind {
        ControlTargetKind::Session => &mut state.excluded_sessions,
        ControlTargetKind::Event => &mut state.excluded_events,
    };
    match action {
        ControlAction::Exclude if targets.insert(id.to_owned()) => Ok(()),
        ControlAction::Restore if targets.remove(id) => Ok(()),
        ControlAction::Exclude => Err(ControlError::InvalidTransition(format!(
            "{id} is already excluded"
        ))),
        ControlAction::Restore => Err(ControlError::InvalidTransition(format!(
            "{id} is not excluded"
        ))),
    }
}

#[derive(Serialize)]
struct RecordHashPayload<'a> {
    version: u32,
    sequence: u64,
    action: ControlAction,
    target_kind: ControlTargetKind,
    target_id: &'a str,
    target_session_id: &'a Option<String>,
    target_content_sha256: &'a Option<String>,
    target_created_at: &'a str,
    timestamp: &'a str,
    previous_record_sha256: &'a Option<String>,
}

fn record_hash(record: &ControlRecord) -> ControlResult<String> {
    let payload = RecordHashPayload {
        version: record.version,
        sequence: record.sequence,
        action: record.action,
        target_kind: record.target_kind,
        target_id: &record.target_id,
        target_session_id: &record.target_session_id,
        target_content_sha256: &record.target_content_sha256,
        target_created_at: &record.target_created_at,
        timestamp: &record.timestamp,
        previous_record_sha256: &record.previous_record_sha256,
    };
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&payload)?)
    ))
}

fn canonical_record_bytes(record: &ControlRecord) -> ControlResult<Vec<u8>> {
    let mut bytes = serde_json::to_vec(record)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn read_session(path: &Path) -> ControlResult<Session> {
    let bytes = fs::read(path).map_err(|source| ControlError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|error| ControlError::CorruptSource {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

fn validate_safe_id(id: &str) -> ControlResult<()> {
    let mut characters = id.chars();
    let Some(first) = characters.next() else {
        return Err(ControlError::InvalidTarget("empty target id".into()));
    };
    if !first.is_ascii_alphanumeric()
        || !characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
    {
        return Err(ControlError::InvalidTarget(format!(
            "unsafe target id {id:?}"
        )));
    }
    Ok(())
}

fn validate_utc_timestamp(value: &str, field: &str) -> ControlResult<()> {
    let parsed: DateTime<FixedOffset> = DateTime::parse_from_rfc3339(value)
        .map_err(|_| ControlError::CorruptLog(format!("{field} is not RFC3339")))?;
    if parsed.offset().local_minus_utc() != 0 {
        return Err(ControlError::CorruptLog(format!("{field} is not UTC")));
    }
    Ok(())
}

fn validate_source_timestamps(session: &Session) -> Result<(), String> {
    validate_source_timestamp(&session.created_at, "session.created_at")?;
    validate_source_timestamp(&session.updated_at, "session.updated_at")?;
    for turn in &session.turns {
        validate_source_timestamp(&turn.created_at, &format!("turn {} created_at", turn.id))?;
        validate_source_timestamp(&turn.updated_at, &format!("turn {} updated_at", turn.id))?;
        if let Some(timestamp) = &turn.request_started_at {
            validate_source_timestamp(timestamp, &format!("turn {} request_started_at", turn.id))?;
        }
    }
    Ok(())
}

fn validate_source_timestamp(value: &str, field: &str) -> Result<(), String> {
    let parsed =
        DateTime::parse_from_rfc3339(value).map_err(|_| format!("{field} is not RFC3339"))?;
    if parsed.offset().local_minus_utc() != 0 {
        return Err(format!("{field} is not UTC"));
    }
    Ok(())
}

fn validate_hash(value: &str, field: &str) -> ControlResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ControlError::CorruptLog(format!(
            "{field} is not a lowercase SHA-256"
        )));
    }
    Ok(())
}

fn validate_optional_hash(value: Option<&str>, field: &str) -> ControlResult<()> {
    if let Some(value) = value {
        validate_hash(value, field)?;
    }
    Ok(())
}

fn segment_name(sequence: u64) -> String {
    format!("{sequence:020}.json")
}

fn parse_segment_name(name: &str) -> ControlResult<u64> {
    if name.len() != 25
        || !name.ends_with(".json")
        || !name[..20].bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(ControlError::CorruptLog(format!(
            "unexpected control-log file {name}"
        )));
    }
    name[..20]
        .parse()
        .map_err(|_| ControlError::CorruptLog(format!("invalid segment filename {name}")))
}

fn is_stale_temp_name(name: &str) -> bool {
    name.strip_prefix(".control-")
        .and_then(|rest| rest.strip_suffix(".tmp"))
        .is_some_and(|uuid| {
            uuid.len() == 32
                && uuid
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> std::io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> std::io::Result<()> {
    Ok(())
}
