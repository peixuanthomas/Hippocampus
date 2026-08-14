use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use rusqlite::{Connection, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::MemoryConfig;
use crate::consolidation::{normalize_match, validate_full_derived_integrity};
use crate::model::{
    EventRole, EvidenceKind, GraphPathTrace, RetrievalChannel, RetrievalDocumentGranularity,
    TurnStatus,
};
use crate::retrieval::{
    RetrievalError, RetrievalResult, RetrievalStore, load_aggregate_embedding_snapshot,
    load_leaf_embedding_snapshot, parse_status, query_terms,
};
use crate::vector::{StoredEmbedding, VectorIndexSpec};

pub const GRAPH_SCHEMA_VERSION: i64 = 1;
pub const GRAPH_ALGORITHM_VERSION: i64 = 1;
pub const EMBEDDING_MUTUAL_TOP_K: usize = 5;
pub const EMBEDDING_MUTUAL_MIN_COSINE: f64 = 0.80;
const KEYWORD_MIN_NORMALIZED_CHARS: usize = 2;
const EDGE_RULES_VERSION: i64 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKind {
    Document,
    Entity,
    Claim,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeType {
    Reply,
    Adjacent,
    EpisodeMember,
    EntityMention,
    SharedEntity,
    KeywordCooccurrence,
    EmbeddingMutualTopK,
    CommonRecall,
    Support,
    Conflict,
    Replacement,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphMaterializationReport {
    pub changed: bool,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_counts: BTreeMap<GraphNodeKind, usize>,
    pub edge_counts: BTreeMap<GraphEdgeType, usize>,
    pub source_sha256: String,
    pub catalog_sha256: String,
    pub vector_index_fingerprint: String,
    pub materialized_at: String,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RawProvenance {
    session_id: String,
    event_id: String,
    start_char: usize,
    end_char: usize,
    content_sha256: String,
    role: EventRole,
}

#[derive(Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Provenance {
    record_ids: Vec<String>,
    spans: Vec<RawProvenance>,
    terms: Vec<String>,
    reasons: Vec<String>,
}

#[derive(Clone, PartialEq, Eq)]
struct Node {
    id: String,
    kind: GraphNodeKind,
    source_id: String,
    session_id: Option<String>,
    granularity: Option<RetrievalDocumentGranularity>,
    source_sha256: String,
}

#[derive(Clone)]
struct Edge {
    id: String,
    kind: GraphEdgeType,
    source: String,
    target: String,
    weight: f64,
    directed: bool,
    provenance_json: String,
    provenance_sha256: String,
}

#[derive(Debug, Clone)]
pub(crate) struct GraphRecallSeed {
    pub channel: RetrievalChannel,
    pub source_id: String,
    pub document_id: Option<String>,
    pub rank: usize,
    pub score: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GraphRecallResult {
    pub paths: Vec<GraphPathTrace>,
    pub warning: Option<String>,
}

type RepresentativePath<'a> = (
    f64,
    usize,
    usize,
    String,
    String,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    &'a GraphRecallSeed,
    f64,
);

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.kind == other.kind
            && self.source == other.source
            && self.target == other.target
            && self.weight.to_bits() == other.weight.to_bits()
            && self.directed == other.directed
            && self.provenance_json == other.provenance_json
            && self.provenance_sha256 == other.provenance_sha256
    }
}
impl Eq for Edge {}

#[derive(Default)]
struct EdgeAccumulator {
    weight: f64,
    provenance: Provenance,
}

#[derive(Clone)]
struct Leaf {
    node_id: String,
    document_id: String,
    session_id: String,
    event_id: String,
    start: usize,
    end: usize,
    hash: String,
    role: EventRole,
    content: String,
    granularity: RetrievalDocumentGranularity,
}

fn hash_parts(domain: &str, parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    let domain = domain.as_bytes();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    format!("{:x}", hasher.finalize())
}

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn node_name(kind: GraphNodeKind) -> &'static str {
    match kind {
        GraphNodeKind::Document => "document",
        GraphNodeKind::Entity => "entity",
        GraphNodeKind::Claim => "claim",
    }
}

fn edge_name(kind: GraphEdgeType) -> &'static str {
    match kind {
        GraphEdgeType::Reply => "reply",
        GraphEdgeType::Adjacent => "adjacent",
        GraphEdgeType::EpisodeMember => "episode_member",
        GraphEdgeType::EntityMention => "entity_mention",
        GraphEdgeType::SharedEntity => "shared_entity",
        GraphEdgeType::KeywordCooccurrence => "keyword_cooccurrence",
        GraphEdgeType::EmbeddingMutualTopK => "embedding_mutual_top_k",
        GraphEdgeType::CommonRecall => "common_recall",
        GraphEdgeType::Support => "support",
        GraphEdgeType::Conflict => "conflict",
        GraphEdgeType::Replacement => "replacement",
    }
}

fn granularity_name(value: RetrievalDocumentGranularity) -> &'static str {
    match value {
        RetrievalDocumentGranularity::Message => "message",
        RetrievalDocumentGranularity::Fragment => "fragment",
        RetrievalDocumentGranularity::Episode => "episode",
        RetrievalDocumentGranularity::Session => "session",
    }
}

fn node_id(kind: GraphNodeKind, source: &str) -> String {
    format!(
        "gnode_{}",
        hash_parts(
            "hippocampus.graph.node-id.v1",
            &[node_name(kind).as_bytes(), source.as_bytes()]
        )
    )
}

fn parse_role(value: &str) -> RetrievalResult<EventRole> {
    match value {
        "user" => Ok(EventRole::User),
        "assistant" => Ok(EventRole::Assistant),
        "system" => Ok(EventRole::System),
        _ => Err(RetrievalError::CorruptIndex(format!(
            "无效事件角色 {value}"
        ))),
    }
}

fn role_name(value: EventRole) -> &'static str {
    match value {
        EventRole::System => "system",
        EventRole::User => "user",
        EventRole::Assistant => "assistant",
    }
}

fn canonical_pair(left: &str, right: &str) -> (String, String) {
    if left < right {
        (left.to_owned(), right.to_owned())
    } else {
        (right.to_owned(), left.to_owned())
    }
}

fn add_edge(
    map: &mut BTreeMap<(GraphEdgeType, String, String), EdgeAccumulator>,
    kind: GraphEdgeType,
    source: &str,
    target: &str,
    directed: bool,
    weight: f64,
    provenance: Provenance,
) -> RetrievalResult<()> {
    if source == target || !weight.is_finite() || weight <= 0.0 {
        return Err(RetrievalError::CorruptIndex("图边权重或端点无效".into()));
    }
    let (source, target) = if directed {
        (source.to_owned(), target.to_owned())
    } else {
        canonical_pair(source, target)
    };
    let entry = map.entry((kind, source, target)).or_default();
    entry.weight += weight;
    entry.provenance.record_ids.extend(provenance.record_ids);
    entry.provenance.spans.extend(provenance.spans);
    entry.provenance.terms.extend(provenance.terms);
    entry.provenance.reasons.extend(provenance.reasons);
    Ok(())
}

fn finish_edges(
    map: BTreeMap<(GraphEdgeType, String, String), EdgeAccumulator>,
) -> RetrievalResult<Vec<Edge>> {
    map.into_iter()
        .map(|((kind, source, target), mut value)| {
            value.provenance.record_ids.sort();
            value.provenance.record_ids.dedup();
            value.provenance.spans.sort_by(|left, right| {
                (
                    &left.session_id,
                    &left.event_id,
                    left.start_char,
                    left.end_char,
                    &left.content_sha256,
                    role_name(left.role),
                )
                    .cmp(&(
                        &right.session_id,
                        &right.event_id,
                        right.start_char,
                        right.end_char,
                        &right.content_sha256,
                        role_name(right.role),
                    ))
            });
            value.provenance.spans.dedup();
            value.provenance.terms.sort();
            value.provenance.terms.dedup();
            value.provenance.reasons.sort();
            value.provenance.reasons.dedup();
            let provenance_json = serde_json::to_string(&value.provenance)
                .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
            let provenance_sha256 = bytes_sha256(provenance_json.as_bytes());
            let id = format!(
                "gedge_{}",
                hash_parts(
                    "hippocampus.graph.edge-id.v1",
                    &[
                        edge_name(kind).as_bytes(),
                        source.as_bytes(),
                        target.as_bytes()
                    ],
                )
            );
            if !value.weight.is_finite() || value.weight <= 0.0 {
                return Err(RetrievalError::CorruptIndex("图聚合边权重无效".into()));
            }
            Ok(Edge {
                id,
                kind,
                source,
                target,
                weight: value.weight,
                directed: kind == GraphEdgeType::Replacement,
                provenance_json,
                provenance_sha256,
            })
        })
        .collect()
}

fn shortest_leaf<'a>(
    leaves: &'a [Leaf],
    event: &str,
    start: usize,
    end: usize,
) -> Option<&'a Leaf> {
    leaves
        .iter()
        .filter(|leaf| leaf.event_id == event && leaf.start <= start && leaf.end >= end)
        .min_by_key(|leaf| (leaf.end - leaf.start, leaf.document_id.as_str()))
}

fn raw_span(leaf: &Leaf, start: usize, end: usize, hash: String) -> RawProvenance {
    RawProvenance {
        session_id: leaf.session_id.clone(),
        event_id: leaf.event_id.clone(),
        start_char: start,
        end_char: end,
        content_sha256: hash,
        role: leaf.role,
    }
}

pub(crate) fn refresh_graph(
    store: &RetrievalStore,
    config: &MemoryConfig,
) -> RetrievalResult<GraphMaterializationReport> {
    config
        .validate()
        .map_err(|error| RetrievalError::CorruptIndex(format!("memory 配置无效：{error}")))?;
    let spec = VectorIndexSpec::from_config(config)
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
    let fingerprint = spec
        .fingerprint()
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
    let _guard = store.acquire_root_write()?;
    let control = store.replay_control_state_under_guard()?;
    let mut connection = store.open_connection()?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| store.database_error(error))?;
    store.require_current_control_projection(&transaction, &control)?;

    let leaf_snapshot = load_leaf_embedding_snapshot(store, &transaction, &spec)?;
    let aggregate_snapshot = load_aggregate_embedding_snapshot(store, &transaction, &spec)?;
    let control_generation_sha256 = control.generation_sha256();
    if leaf_snapshot.control_generation_sha256 != control_generation_sha256
        || aggregate_snapshot.control_generation_sha256 != control_generation_sha256
        || leaf_snapshot.control_generation_sha256 != aggregate_snapshot.control_generation_sha256
    {
        return Err(RetrievalError::ControlStateChanged);
    }
    validate_full_derived_integrity(&transaction)?;
    let embeddings =
        store.compatible_embeddings_from_connection(&transaction, &spec, &fingerprint, None)?;
    validate_graph_embeddings(&embeddings)?;
    let document_count: i64 = transaction
        .query_row("SELECT count(*) FROM memory_documents", [], |row| {
            row.get(0)
        })
        .map_err(|error| store.database_error(error))?;
    if usize::try_from(document_count).ok() != Some(embeddings.len()) {
        return Err(RetrievalError::CorruptIndex(
            "memory_documents 未被 compatible embeddings 完整覆盖".into(),
        ));
    }
    let runs = store.validated_retrieval_runs_from_connection(&transaction)?;
    let (nodes, leaves, event_rows) =
        build_nodes(&transaction, &leaf_snapshot, &aggregate_snapshot)?;
    let edges = build_edges(
        &transaction,
        config,
        &nodes,
        &leaves,
        &event_rows,
        &embeddings,
        &runs,
    )?;
    let config_sha256 = config_hash(config, &fingerprint);
    let source_sha256 = source_hash(
        &transaction,
        &control_generation_sha256,
        &leaf_snapshot.catalog_sha256,
        &aggregate_snapshot.catalog_sha256,
        &fingerprint,
        &embeddings,
        &runs,
    )?;
    let catalog_sha256 = catalog_hash(&nodes, &edges);
    let existing_time = exact_existing_catalog(
        &transaction,
        &nodes,
        &edges,
        &fingerprint,
        &config_sha256,
        &source_sha256,
        &catalog_sha256,
    )?;
    let (changed, materialized_at) = if let Some(time) = existing_time {
        (false, time)
    } else {
        let time = Utc::now().to_rfc3339();
        replace_catalog(
            &transaction,
            &nodes,
            &edges,
            (
                &fingerprint,
                &config_sha256,
                &source_sha256,
                &catalog_sha256,
            ),
            &time,
        )?;
        let audited = exact_existing_catalog(
            &transaction,
            &nodes,
            &edges,
            &fingerprint,
            &config_sha256,
            &source_sha256,
            &catalog_sha256,
        )?;
        if audited.as_deref() != Some(time.as_str()) {
            return Err(RetrievalError::CorruptIndex("图写后审计失败".into()));
        }
        (true, time)
    };
    store.require_unchanged_control_state(&control)?;
    transaction
        .commit()
        .map_err(|error| store.database_error(error))?;
    Ok(report(
        changed,
        &nodes,
        &edges,
        source_sha256,
        catalog_sha256,
        fingerprint,
        materialized_at,
    ))
}

pub(crate) fn recall_graph_from_connection(
    store: &RetrievalStore,
    connection: &Connection,
    config: &MemoryConfig,
    seeds: &[GraphRecallSeed],
    session_filter: Option<&str>,
) -> RetrievalResult<GraphRecallResult> {
    config
        .validate()
        .map_err(|error| RetrievalError::CorruptIndex(format!("memory 配置无效：{error}")))?;
    if seeds
        .iter()
        .any(|seed| seed.rank == 0 || !seed.score.is_finite())
    {
        return Err(RetrievalError::CorruptIndex(
            "图 seed rank/score 无效".into(),
        ));
    }
    let spec = VectorIndexSpec::from_config(config)
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
    let fingerprint = spec
        .fingerprint()
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
    let leaf_snapshot = load_leaf_embedding_snapshot(store, connection, &spec)?;
    let aggregate_snapshot = load_aggregate_embedding_snapshot(store, connection, &spec)?;
    if leaf_snapshot.control_generation_sha256 != aggregate_snapshot.control_generation_sha256 {
        return Err(RetrievalError::ControlStateChanged);
    }
    validate_full_derived_integrity(connection)?;
    let embeddings =
        store.compatible_embeddings_from_connection(connection, &spec, &fingerprint, None)?;
    validate_graph_embeddings(&embeddings)?;
    let document_count: i64 = connection
        .query_row("SELECT count(*) FROM memory_documents", [], |row| {
            row.get(0)
        })
        .map_err(|error| store.database_error(error))?;
    if usize::try_from(document_count).ok() != Some(embeddings.len()) {
        return Err(RetrievalError::CorruptIndex(
            "memory_documents 未被 compatible embeddings 完整覆盖".into(),
        ));
    }
    let runs = store.validated_retrieval_runs_from_connection(connection)?;
    let (expected_nodes, leaves, event_rows) =
        build_nodes(connection, &leaf_snapshot, &aggregate_snapshot)?;
    let expected_edges = build_edges(
        connection,
        config,
        &expected_nodes,
        &leaves,
        &event_rows,
        &embeddings,
        &runs,
    )?;
    let config_sha256 = config_hash(config, &fingerprint);
    let source_sha256 = source_hash(
        connection,
        &leaf_snapshot.control_generation_sha256,
        &leaf_snapshot.catalog_sha256,
        &aggregate_snapshot.catalog_sha256,
        &fingerprint,
        &embeddings,
        &runs,
    )?;
    let catalog_sha256 = catalog_hash(&expected_nodes, &expected_edges);
    if exact_existing_catalog(
        connection,
        &expected_nodes,
        &expected_edges,
        &fingerprint,
        &config_sha256,
        &source_sha256,
        &catalog_sha256,
    )?
    .is_none()
    {
        return Err(RetrievalError::CorruptIndex(
            "图 materialization 缺失或已过期".into(),
        ));
    }

    let nodes = read_nodes(connection)?;
    let edges = read_edges(connection)?;
    if seeds.is_empty() {
        return Ok(GraphRecallResult::default());
    }
    let by_id = nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let allowed = |node: &Node| {
        node.kind == GraphNodeKind::Entity
            || session_filter.is_none_or(|scope| node.session_id.as_deref() == Some(scope))
    };
    let channel_key = |channel| match channel {
        RetrievalChannel::Bm25 => 0_u8,
        RetrievalChannel::Vector => 1,
        RetrievalChannel::Entity => 2,
        RetrievalChannel::State => 3,
        RetrievalChannel::Episode => 4,
        RetrievalChannel::Graph => 5,
    };
    let mut channel_counts = BTreeMap::<u8, f64>::new();
    for seed in seeds {
        *channel_counts.entry(channel_key(seed.channel)).or_default() += 1.0 / seed.rank as f64;
    }
    let active_channels = channel_counts.len() as f64;
    let mut seed_rows = Vec::new();
    let mut personalization = BTreeMap::<String, f64>::new();
    for seed in seeds {
        let kind = if seed.channel == RetrievalChannel::Entity {
            GraphNodeKind::Entity
        } else {
            GraphNodeKind::Document
        };
        let id = node_id(kind, &seed.source_id);
        let node = by_id
            .get(id.as_str())
            .ok_or_else(|| RetrievalError::CorruptIndex(format!("图 seed 节点缺失：{id}")))?;
        if !allowed(node) {
            return Err(RetrievalError::CorruptIndex(format!(
                "图 seed 节点超出会话范围：{id}"
            )));
        }
        let mass =
            (1.0 / seed.rank as f64) / channel_counts[&channel_key(seed.channel)] / active_channels;
        *personalization.entry(id.clone()).or_default() += mass;
        seed_rows.push((seed, id, mass));
    }
    let total = personalization.values().sum::<f64>();
    if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
        return Err(RetrievalError::CorruptIndex(
            "图 personalization 未严格归一".into(),
        ));
    }

    let mut incident = BTreeMap::<String, Vec<&Edge>>::new();
    for edge in &edges {
        let source = by_id
            .get(edge.source.as_str())
            .ok_or_else(|| RetrievalError::CorruptIndex(format!("图边 {} source 缺失", edge.id)))?;
        let target = by_id
            .get(edge.target.as_str())
            .ok_or_else(|| RetrievalError::CorruptIndex(format!("图边 {} target 缺失", edge.id)))?;
        if allowed(source) && allowed(target) {
            incident.entry(edge.source.clone()).or_default().push(edge);
            incident.entry(edge.target.clone()).or_default().push(edge);
        }
    }
    let mut local = personalization.keys().cloned().collect::<BTreeSet<_>>();
    let mut frontier = local.clone();
    for _ in 0..config.max_graph_depth {
        let mut next = BTreeSet::new();
        for id in &frontier {
            for edge in incident.get(id).into_iter().flatten() {
                let other = if edge.source == *id {
                    &edge.target
                } else {
                    &edge.source
                };
                if local.insert(other.clone()) {
                    next.insert(other.clone());
                }
            }
        }
        frontier = next;
    }
    let mut rank = personalization.clone();
    for id in &local {
        rank.entry(id.clone()).or_default();
    }
    let mut converged = false;
    for _ in 0..50 {
        let mut next = personalization
            .iter()
            .map(|(id, mass)| (id.clone(), 0.5 * mass))
            .collect::<BTreeMap<_, _>>();
        let mut dangling = 0.0;
        for (id, mass) in &rank {
            let local_edges = incident
                .get(id)
                .into_iter()
                .flatten()
                .filter(|edge| {
                    local.contains(if edge.source == *id {
                        &edge.target
                    } else {
                        &edge.source
                    })
                })
                .collect::<Vec<_>>();
            let weight = local_edges.iter().map(|edge| edge.weight).sum::<f64>();
            if !weight.is_finite() || weight < 0.0 {
                return Err(RetrievalError::CorruptIndex("图局部转移权重无效".into()));
            }
            if weight == 0.0 {
                dangling += mass;
            } else {
                for edge in local_edges {
                    let other = if edge.source == *id {
                        &edge.target
                    } else {
                        &edge.source
                    };
                    *next.entry(other.clone()).or_default() += 0.5 * mass * edge.weight / weight;
                }
            }
        }
        for (id, mass) in &personalization {
            *next.entry(id.clone()).or_default() += 0.5 * dangling * mass;
        }
        let delta = local
            .iter()
            .map(|id| {
                (next.get(id).copied().unwrap_or(0.0) - rank.get(id).copied().unwrap_or(0.0)).abs()
            })
            .sum::<f64>();
        if next
            .values()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(RetrievalError::CorruptIndex("图 PPR 产生无效数值".into()));
        }
        let next_total = next.values().sum::<f64>();
        if !next_total.is_finite() || (next_total - 1.0).abs() > 1e-9 {
            return Err(RetrievalError::CorruptIndex("图 PPR 质量未守恒".into()));
        }
        rank = next;
        if delta <= 1e-6 {
            converged = true;
            break;
        }
    }
    let mut targets = local
        .iter()
        .filter_map(|id| {
            let node = by_id[id.as_str()];
            (node.kind == GraphNodeKind::Document)
                .then(|| (id.clone(), node, rank.get(id).copied().unwrap_or(0.0)))
        })
        .collect::<Vec<_>>();
    targets.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.1.source_id.cmp(&right.1.source_id))
    });
    let mut paths = Vec::new();
    for (target_rank, (target_id, target, score)) in targets.into_iter().enumerate() {
        let mut best: Option<RepresentativePath<'_>> = None;
        for (seed, seed_id, seed_mass) in &seed_rows {
            let channel_priority = match seed.channel {
                RetrievalChannel::Bm25 => 0,
                RetrievalChannel::Vector => 1,
                _ => 2,
            };
            let mut candidates = Vec::new();
            if seed_id == &target_id {
                candidates.push((Vec::new(), vec![seed_id.clone()], 1.0));
            }
            for first in incident.get(seed_id).into_iter().flatten() {
                let mid = if first.source == *seed_id {
                    &first.target
                } else {
                    &first.source
                };
                let denom = incident
                    .get(seed_id)
                    .into_iter()
                    .flatten()
                    .filter(|edge| {
                        local.contains(if edge.source == *seed_id {
                            &edge.target
                        } else {
                            &edge.source
                        })
                    })
                    .map(|edge| edge.weight)
                    .sum::<f64>();
                if mid == &target_id {
                    candidates.push((
                        vec![*first],
                        vec![seed_id.clone(), mid.clone()],
                        first.weight / denom,
                    ));
                }
                if config.max_graph_depth == 2 && mid != seed_id {
                    for second in incident.get(mid).into_iter().flatten() {
                        let end = if second.source == *mid {
                            &second.target
                        } else {
                            &second.source
                        };
                        if end == &target_id && end != seed_id {
                            let mid_denom = incident
                                .get(mid)
                                .into_iter()
                                .flatten()
                                .filter(|edge| {
                                    local.contains(if edge.source == *mid {
                                        &edge.target
                                    } else {
                                        &edge.source
                                    })
                                })
                                .map(|edge| edge.weight)
                                .sum::<f64>();
                            candidates.push((
                                vec![*first, *second],
                                vec![seed_id.clone(), mid.clone(), end.clone()],
                                first.weight / denom * second.weight / mid_denom,
                            ));
                        }
                    }
                }
            }
            for (path_edges, path_nodes, probability) in candidates {
                let quality = seed_mass * probability;
                let edge_ids = path_edges
                    .iter()
                    .map(|edge| edge.id.clone())
                    .collect::<Vec<_>>();
                let edge_types = path_edges
                    .iter()
                    .map(|edge| edge_name(edge.kind).to_owned())
                    .collect::<Vec<_>>();
                let key = (
                    quality,
                    path_edges.len(),
                    channel_priority,
                    seed_id.clone(),
                    seed.source_id.clone(),
                    edge_ids.clone(),
                    path_nodes.clone(),
                    edge_types,
                    *seed,
                    *seed_mass,
                );
                let replace = best.as_ref().is_none_or(|old| {
                    quality > old.0
                        || (quality == old.0
                            && (key.1, key.2, &key.3, &key.4, &key.5, &key.6)
                                < (old.1, old.2, &old.3, &old.4, &old.5, &old.6))
                });
                if replace {
                    best = Some(key);
                }
            }
        }
        let Some((quality, _, _, seed_node_id, _, edge_ids, node_ids, edge_types, seed, seed_mass)) =
            best
        else {
            continue;
        };
        paths.push(GraphPathTrace {
            seed_document_id: seed.document_id.clone().unwrap_or_default(),
            target_document_id: target.source_id.clone(),
            edge_types,
            node_ids,
            score,
            path_quality: quality,
            seed_channel: seed.channel,
            seed_node_id,
            seed_source_id: seed.source_id.clone(),
            seed_rank: seed.rank,
            seed_score: seed.score,
            seed_mass,
            edge_ids,
            target_rank: target_rank + 1,
            target_granularity: target.granularity,
            target_session_id: target.session_id.clone().unwrap_or_default(),
            reason: if seed.document_id.as_deref() == Some(target.source_id.as_str()) {
                "seed_document".into()
            } else {
                String::new()
            },
            ..Default::default()
        });
    }
    Ok(GraphRecallResult {
        paths,
        warning: (!converged).then(|| "graph PPR did not converge after 50 iterations".into()),
    })
}

fn validate_graph_embeddings(embeddings: &[StoredEmbedding]) -> RetrievalResult<()> {
    for embedding in embeddings {
        if embedding.vector.is_empty() {
            return Err(RetrievalError::CorruptIndex(format!(
                "图输入文档 {} 的向量为空",
                embedding.document_id
            )));
        }
        let norm_squared = embedding.vector.iter().try_fold(0.0_f64, |sum, value| {
            value
                .is_finite()
                .then_some(sum + f64::from(*value) * f64::from(*value))
        });
        let Some(norm_squared) = norm_squared.filter(|value| value.is_finite() && *value > 0.0)
        else {
            return Err(RetrievalError::CorruptIndex(format!(
                "图输入文档 {} 的向量包含非有限值或零范数",
                embedding.document_id
            )));
        };
        let norm = norm_squared.sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 1e-5 {
            return Err(RetrievalError::CorruptIndex(format!(
                "图输入文档 {} 的向量不是单位向量",
                embedding.document_id
            )));
        }
    }
    Ok(())
}

type EventRow = (
    String,
    usize,
    EventRole,
    String,
    String,
    Option<String>,
    Option<TurnStatus>,
);
type EventRows = BTreeMap<String, EventRow>;

fn build_nodes(
    connection: &Connection,
    leaf_snapshot: &crate::retrieval::LeafEmbeddingSnapshot,
    aggregate_snapshot: &crate::retrieval::AggregateEmbeddingSnapshot,
) -> RetrievalResult<(Vec<Node>, Vec<Leaf>, EventRows)> {
    let mut events = BTreeMap::new();
    let mut statement = connection.prepare("SELECT event_id,session_id,sequence,role,content,content_sha256,reply_to_event_id,turn_status FROM events ORDER BY event_id")
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
    for row in rows {
        let (id, session, sequence, role, content, hash, reply, turn_status) =
            row.map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        if hash != crate::model::content_sha256(&content) {
            return Err(RetrievalError::CorruptIndex(format!("事件 {id} hash 损坏")));
        }
        events.insert(
            id,
            (
                session,
                usize::try_from(sequence)
                    .map_err(|_| RetrievalError::CorruptIndex("事件 sequence 无效".into()))?,
                parse_role(&role)?,
                content,
                hash,
                reply,
                turn_status
                    .as_deref()
                    .map(parse_status)
                    .transpose()
                    .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?,
            ),
        );
    }
    let mut nodes = Vec::new();
    let mut leaves = Vec::new();
    for document in &leaf_snapshot.documents {
        let event = events
            .get(&document.source_event_id)
            .ok_or_else(|| RetrievalError::CorruptIndex("leaf 缺少事件".into()))?;
        if event.0 != document.session_id {
            return Err(RetrievalError::CorruptIndex("leaf session 绑定错误".into()));
        }
        let id = node_id(GraphNodeKind::Document, &document.document_id);
        nodes.push(Node {
            id: id.clone(),
            kind: GraphNodeKind::Document,
            source_id: document.document_id.clone(),
            session_id: Some(document.session_id.clone()),
            granularity: Some(document.granularity),
            source_sha256: document.source_sha256.clone(),
        });
        leaves.push(Leaf {
            node_id: id,
            document_id: document.document_id.clone(),
            session_id: document.session_id.clone(),
            event_id: document.source_event_id.clone(),
            start: document.start_char,
            end: document.end_char,
            hash: document.source_sha256.clone(),
            role: event.2,
            content: document.content.clone(),
            granularity: document.granularity,
        });
    }
    for document in &aggregate_snapshot.documents {
        nodes.push(Node {
            id: node_id(GraphNodeKind::Document, &document.document_id),
            kind: GraphNodeKind::Document,
            source_id: document.document_id.clone(),
            session_id: Some(document.session_id.clone()),
            granularity: Some(document.granularity),
            source_sha256: document.source_sha256.clone(),
        });
    }
    append_sql_nodes(connection, &mut nodes, GraphNodeKind::Entity)?;
    append_sql_nodes(connection, &mut nodes, GraphNodeKind::Claim)?;
    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    let mut ids = BTreeSet::new();
    if nodes.iter().any(|node| !ids.insert(node.id.clone())) {
        return Err(RetrievalError::CorruptIndex("图节点 ID 冲突".into()));
    }
    Ok((nodes, leaves, events))
}

fn append_sql_nodes(
    connection: &Connection,
    nodes: &mut Vec<Node>,
    kind: GraphNodeKind,
) -> RetrievalResult<()> {
    let sql = match kind {
        GraphNodeKind::Entity => {
            "SELECT entity_id,NULL,kind,canonical_name,normalized_name,disambiguation,created_session_id,created_batch_key,created_event_id,created_start,created_end,created_hash,created_at,updated_at FROM memory_entities WHERE disambiguation='resolved' ORDER BY entity_id"
        }
        GraphNodeKind::Claim => {
            "SELECT claim_id,session_id,subject_entity_id,predicate_key,normalized_relation,object_kind,coalesce(object_text,''),coalesce(object_entity_id,''),normalized_object,polarity,cardinality,certainty,state,asserted_at,coalesce(event_time,''),valid_from,coalesce(valid_to,''),reference_time,created_batch_key,updated_batch_key,created_at,updated_at FROM memory_claims ORDER BY claim_id"
        }
        GraphNodeKind::Document => unreachable!(),
    };
    let mut statement = connection
        .prepare(sql)
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
    let column_count = statement.column_count();
    let rows = statement
        .query_map([], |row| {
            let source = row.get::<_, String>(0)?;
            let session = row.get::<_, Option<String>>(1)?;
            let mut fields = Vec::new();
            for index in 2..column_count {
                let value = row.get_ref(index)?;
                fields.push(match value {
                    rusqlite::types::ValueRef::Null => Vec::new(),
                    rusqlite::types::ValueRef::Integer(v) => v.to_le_bytes().to_vec(),
                    rusqlite::types::ValueRef::Real(v) => v.to_bits().to_le_bytes().to_vec(),
                    rusqlite::types::ValueRef::Text(v) | rusqlite::types::ValueRef::Blob(v) => {
                        v.to_vec()
                    }
                });
            }
            Ok((source, session, fields))
        })
        .map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
    for row in rows {
        let (source, session, fields) =
            row.map_err(|error| RetrievalError::CorruptIndex(error.to_string()))?;
        let mut bound_fields = vec![source.as_bytes().to_vec()];
        bound_fields.push(session.as_deref().unwrap_or("").as_bytes().to_vec());
        bound_fields.extend(fields);
        let refs = bound_fields.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let hash = hash_parts(
            match kind {
                GraphNodeKind::Entity => "hippocampus.graph.entity-source.v1",
                GraphNodeKind::Claim => "hippocampus.graph.claim-source.v1",
                _ => unreachable!(),
            },
            &refs,
        );
        nodes.push(Node {
            id: node_id(kind, &source),
            kind,
            source_id: source,
            session_id: session,
            granularity: None,
            source_sha256: hash,
        });
    }
    Ok(())
}

fn build_edges(
    connection: &Connection,
    config: &MemoryConfig,
    nodes: &[Node],
    leaves: &[Leaf],
    events: &EventRows,
    embeddings: &[StoredEmbedding],
    runs: &[(String, crate::model::RetrievalTrace)],
) -> RetrievalResult<Vec<Edge>> {
    let node_by_source = nodes
        .iter()
        .map(|node| ((node.kind, node.source_id.as_str()), node.id.as_str()))
        .collect::<BTreeMap<_, _>>();
    let message_by_event = leaves
        .iter()
        .filter(|leaf| leaf.granularity == RetrievalDocumentGranularity::Message)
        .map(|leaf| (leaf.event_id.as_str(), leaf))
        .collect::<BTreeMap<_, _>>();
    let mut acc = BTreeMap::new();
    let mut by_session = BTreeMap::<&str, Vec<(&String, &EventRow)>>::new();
    for row in events {
        if row.1.2 != EventRole::System && !row.1.3.trim().is_empty() {
            by_session.entry(&row.1.0).or_default().push(row);
        }
    }
    for rows in by_session.values_mut() {
        rows.sort_by_key(|row| row.1.1);
    }
    for rows in by_session.values() {
        for pair in rows.windows(2) {
            let left = message_by_event
                .get(pair[0].0.as_str())
                .ok_or_else(|| RetrievalError::CorruptIndex("相邻事件缺少 message doc".into()))?;
            let right = message_by_event
                .get(pair[1].0.as_str())
                .ok_or_else(|| RetrievalError::CorruptIndex("相邻事件缺少 message doc".into()))?;
            add_edge(
                &mut acc,
                GraphEdgeType::Adjacent,
                &left.node_id,
                &right.node_id,
                false,
                1.0,
                Provenance {
                    record_ids: vec![pair[0].0.clone(), pair[1].0.clone()],
                    spans: vec![
                        raw_span(left, 0, left.end, left.hash.clone()),
                        raw_span(right, 0, right.end, right.hash.clone()),
                    ],
                    ..Default::default()
                },
            )?;
        }
    }
    for (id, event) in events {
        if event.2 == EventRole::System || event.3.trim().is_empty() {
            continue;
        }
        if let Some(reply) = &event.5 {
            let target = events
                .get(reply)
                .ok_or_else(|| RetrievalError::CorruptIndex(format!("回复 {id} 目标缺失")))?;
            if target.0 != event.0 || target.2 == EventRole::System {
                return Err(RetrievalError::CorruptIndex(format!(
                    "回复 {id} 跨会话或指向 system"
                )));
            }
            if target.3.trim().is_empty() {
                if target.2 == EventRole::Assistant && target.6 == Some(TurnStatus::Failed) {
                    continue;
                }
                return Err(RetrievalError::CorruptIndex(format!(
                    "回复 {id} 指向非失败状态的空消息"
                )));
            }
            let left = message_by_event
                .get(id.as_str())
                .ok_or_else(|| RetrievalError::CorruptIndex("回复缺少 message doc".into()))?;
            let right = message_by_event
                .get(reply.as_str())
                .ok_or_else(|| RetrievalError::CorruptIndex("回复目标缺少 message doc".into()))?;
            add_edge(
                &mut acc,
                GraphEdgeType::Reply,
                &left.node_id,
                &right.node_id,
                false,
                1.0,
                Provenance {
                    record_ids: vec![id.clone(), reply.clone()],
                    spans: vec![
                        raw_span(left, 0, left.end, left.hash.clone()),
                        raw_span(right, 0, right.end, right.hash.clone()),
                    ],
                    ..Default::default()
                },
            )?;
        }
    }
    add_episode_edges(connection, &node_by_source, &message_by_event, &mut acc)?;
    let chosen_mentions = add_entity_edges(connection, config, &node_by_source, leaves, &mut acc)?;
    add_shared_edges(config, &chosen_mentions, &mut acc)?;
    add_keyword_edges(config, leaves, &mut acc)?;
    add_embedding_edges(&node_by_source, embeddings, &mut acc)?;
    add_recall_edges(runs, leaves, &mut acc)?;
    add_claim_edges(connection, &node_by_source, leaves, &mut acc)?;
    finish_edges(acc)
}

type Acc = BTreeMap<(GraphEdgeType, String, String), EdgeAccumulator>;

fn add_episode_edges(
    connection: &Connection,
    nodes: &BTreeMap<(GraphNodeKind, &str), &str>,
    messages: &BTreeMap<&str, &Leaf>,
    acc: &mut Acc,
) -> RetrievalResult<()> {
    let mut statement=connection.prepare("SELECT d.document_id,m.event_id,m.start_char,m.end_char,m.content_sha256,e.session_id,e.role FROM memory_documents d JOIN memory_document_members m ON m.document_id=d.document_id JOIN events e ON e.event_id=m.event_id WHERE d.granularity='episode' ORDER BY d.document_id,m.ordinal").map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    let rows = statement
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
            ))
        })
        .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
    for row in rows {
        let (doc, event, start, end, hash, session, role) =
            row.map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        let episode = *nodes
            .get(&(GraphNodeKind::Document, doc.as_str()))
            .ok_or_else(|| RetrievalError::CorruptIndex("episode node缺失".into()))?;
        let message = *messages
            .get(event.as_str())
            .ok_or_else(|| RetrievalError::CorruptIndex("episode member message缺失".into()))?;
        if start != 0
            || usize::try_from(end).ok() != Some(message.end)
            || session != message.session_id
            || parse_role(&role)? != message.role
            || hash != message.hash
        {
            return Err(RetrievalError::CorruptIndex(
                "episode member provenance错误".into(),
            ));
        }
        add_edge(
            acc,
            GraphEdgeType::EpisodeMember,
            episode,
            &message.node_id,
            false,
            1.0,
            Provenance {
                record_ids: vec![doc, event],
                spans: vec![raw_span(message, 0, message.end, message.hash.clone())],
                ..Default::default()
            },
        )?;
    }
    Ok(())
}

#[derive(Clone)]
struct MentionLink {
    entity: String,
    document: String,
    mention: String,
}
fn add_entity_edges(
    connection: &Connection,
    _config: &MemoryConfig,
    nodes: &BTreeMap<(GraphNodeKind, &str), &str>,
    leaves: &[Leaf],
    acc: &mut Acc,
) -> RetrievalResult<Vec<MentionLink>> {
    let mut out = Vec::new();
    let mut statement=connection.prepare("SELECT mention_id,entity_id,event_id,start_char,end_char,content_sha256,session_id,role,mention_kind FROM memory_entity_mentions WHERE entity_status='resolved' ORDER BY mention_id").map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    let rows = statement
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, String>(7)?,
                r.get::<_, String>(8)?,
            ))
        })
        .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
    for row in rows {
        let (mention, entity, event, start, end, hash, session, role, reason) =
            row.map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        let start = usize::try_from(start)
            .map_err(|_| RetrievalError::CorruptIndex("mention span无效".into()))?;
        let end = usize::try_from(end)
            .map_err(|_| RetrievalError::CorruptIndex("mention span无效".into()))?;
        let leaf = shortest_leaf(leaves, &event, start, end)
            .ok_or_else(|| RetrievalError::CorruptIndex(format!("mention {mention} 无leaf映射")))?;
        let entity_node = *nodes
            .get(&(GraphNodeKind::Entity, entity.as_str()))
            .ok_or_else(|| {
                RetrievalError::CorruptIndex(format!(
                    "resolved mention {mention} 指向非resolved entity"
                ))
            })?;
        if session != leaf.session_id || parse_role(&role)? != leaf.role {
            return Err(RetrievalError::CorruptIndex(
                "mention raw binding错误".into(),
            ));
        }
        add_edge(
            acc,
            GraphEdgeType::EntityMention,
            entity_node,
            &leaf.node_id,
            false,
            1.0,
            Provenance {
                record_ids: vec![mention.clone()],
                spans: vec![raw_span(leaf, start, end, hash)],
                reasons: vec![reason],
                ..Default::default()
            },
        )?;
        out.push(MentionLink {
            entity,
            document: leaf.node_id.clone(),
            mention,
        });
    }
    Ok(out)
}

fn add_shared_edges(
    config: &MemoryConfig,
    mentions: &[MentionLink],
    acc: &mut Acc,
) -> RetrievalResult<()> {
    let mut by = BTreeMap::<&str, BTreeMap<&str, BTreeSet<&str>>>::new();
    for m in mentions {
        by.entry(&m.entity)
            .or_default()
            .entry(&m.document)
            .or_default()
            .insert(&m.mention);
    }
    for (entity, docs) in by {
        if !(2..=config.graph_candidate_limit).contains(&docs.len()) {
            continue;
        }
        let docs = docs.into_iter().collect::<Vec<_>>();
        for i in 0..docs.len() {
            for j in i + 1..docs.len() {
                let records = docs[i]
                    .1
                    .union(&docs[j].1)
                    .map(|v| (*v).to_owned())
                    .collect();
                add_edge(
                    acc,
                    GraphEdgeType::SharedEntity,
                    docs[i].0,
                    docs[j].0,
                    false,
                    1.0,
                    Provenance {
                        record_ids: records,
                        terms: vec![entity.to_owned()],
                        ..Default::default()
                    },
                )?;
            }
        }
    }
    Ok(())
}

fn add_keyword_edges(config: &MemoryConfig, leaves: &[Leaf], acc: &mut Acc) -> RetrievalResult<()> {
    let docs = leaves
        .iter()
        .filter(|l| l.granularity == RetrievalDocumentGranularity::Message)
        .map(|leaf| {
            let terms = query_terms(&leaf.content)
                .into_iter()
                .map(|t| normalize_match(&t))
                .filter(|t| !t.is_empty() && t.chars().count() >= KEYWORD_MIN_NORMALIZED_CHARS)
                .collect::<BTreeSet<_>>();
            (leaf, terms)
        })
        .collect::<Vec<_>>();
    let mut df = BTreeMap::<String, usize>::new();
    for (_, terms) in &docs {
        for term in terms {
            *df.entry(term.clone()).or_default() += 1;
        }
    }
    for i in 0..docs.len() {
        for j in i + 1..docs.len() {
            let intersection = docs[i]
                .1
                .intersection(&docs[j].1)
                .filter(|t| {
                    df.get(*t)
                        .is_some_and(|n| (2..=config.graph_candidate_limit).contains(n))
                })
                .cloned()
                .collect::<Vec<_>>();
            if intersection.is_empty() {
                continue;
            }
            let eligible_i = docs[i]
                .1
                .iter()
                .filter(|t| {
                    df.get(*t)
                        .is_some_and(|n| (2..=config.graph_candidate_limit).contains(n))
                })
                .collect::<BTreeSet<_>>();
            let eligible_j = docs[j]
                .1
                .iter()
                .filter(|t| {
                    df.get(*t)
                        .is_some_and(|n| (2..=config.graph_candidate_limit).contains(n))
                })
                .collect::<BTreeSet<_>>();
            let union = eligible_i.union(&eligible_j).count();
            add_edge(
                acc,
                GraphEdgeType::KeywordCooccurrence,
                &docs[i].0.node_id,
                &docs[j].0.node_id,
                false,
                intersection.len() as f64 / union as f64,
                Provenance {
                    terms: intersection,
                    ..Default::default()
                },
            )?;
        }
    }
    Ok(())
}

fn add_embedding_edges(
    nodes: &BTreeMap<(GraphNodeKind, &str), &str>,
    rows: &[StoredEmbedding],
    acc: &mut Acc,
) -> RetrievalResult<()> {
    let mut tops = BTreeMap::<&str, Vec<(&str, f64)>>::new();
    for left in rows {
        let mut ranks = Vec::new();
        for right in rows {
            if left.document_id == right.document_id {
                continue;
            }
            let score = left
                .vector
                .iter()
                .zip(&right.vector)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>();
            if !score.is_finite() {
                return Err(RetrievalError::CorruptIndex("cosine计算非有限".into()));
            }
            if score >= EMBEDDING_MUTUAL_MIN_COSINE {
                ranks.push((right.document_id.as_str(), score));
            }
        }
        ranks.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        ranks.truncate(EMBEDDING_MUTUAL_TOP_K);
        tops.insert(&left.document_id, ranks);
    }
    for (left, ranks) in &tops {
        for (right, score) in ranks {
            if left >= right {
                continue;
            }
            if tops
                .get(right)
                .is_some_and(|v| v.iter().any(|(id, _)| id == left))
            {
                let l = *nodes
                    .get(&(GraphNodeKind::Document, *left))
                    .ok_or_else(|| RetrievalError::CorruptIndex("embedding doc node缺失".into()))?;
                let r = *nodes
                    .get(&(GraphNodeKind::Document, *right))
                    .ok_or_else(|| RetrievalError::CorruptIndex("embedding doc node缺失".into()))?;
                add_edge(
                    acc,
                    GraphEdgeType::EmbeddingMutualTopK,
                    l,
                    r,
                    false,
                    *score,
                    Provenance {
                        record_ids: vec![(*left).to_owned(), (*right).to_owned()],
                        ..Default::default()
                    },
                )?;
            }
        }
    }
    Ok(())
}

fn add_recall_edges(
    runs: &[(String, crate::model::RetrievalTrace)],
    leaves: &[Leaf],
    acc: &mut Acc,
) -> RetrievalResult<()> {
    for (answer, trace) in runs {
        let mut docs = BTreeMap::<String, &Leaf>::new();
        for evidence in trace
            .selected_evidence
            .iter()
            .filter(|e| e.kind == EvidenceKind::Core)
        {
            let leaf = shortest_leaf(
                leaves,
                &evidence.span.event_id,
                evidence.span.start_char,
                evidence.span.end_char,
            )
            .ok_or_else(|| {
                RetrievalError::CorruptIndex(format!("run {answer} evidence无leaf映射"))
            })?;
            if evidence.content_sha256
                != crate::model::content_sha256(
                    &leaf
                        .content
                        .chars()
                        .skip(evidence.span.start_char - leaf.start)
                        .take(evidence.span.end_char - evidence.span.start_char)
                        .collect::<String>(),
                )
                || evidence.role != leaf.role
            {
                return Err(RetrievalError::CorruptIndex(format!(
                    "run {answer} evidence binding错误"
                )));
            }
            docs.insert(leaf.node_id.clone(), leaf);
        }
        let docs = docs.into_values().collect::<Vec<_>>();
        for i in 0..docs.len() {
            for j in i + 1..docs.len() {
                add_edge(
                    acc,
                    GraphEdgeType::CommonRecall,
                    &docs[i].node_id,
                    &docs[j].node_id,
                    false,
                    1.0,
                    Provenance {
                        record_ids: vec![answer.clone()],
                        ..Default::default()
                    },
                )?;
            }
        }
    }
    Ok(())
}

fn add_claim_edges(
    connection: &Connection,
    nodes: &BTreeMap<(GraphNodeKind, &str), &str>,
    leaves: &[Leaf],
    acc: &mut Acc,
) -> RetrievalResult<()> {
    let mut statement=connection.prepare("SELECT evidence_id,claim_id,event_id,start_char,end_char,content_sha256,session_id,role,kind FROM memory_claim_evidence ORDER BY evidence_id").map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    let rows = statement
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, String>(7)?,
                r.get::<_, String>(8)?,
            ))
        })
        .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
    for row in rows {
        let (id, claim, event, start, end, hash, session, role, reason) =
            row.map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        let start = usize::try_from(start)
            .map_err(|_| RetrievalError::CorruptIndex("evidence span无效".into()))?;
        let end = usize::try_from(end)
            .map_err(|_| RetrievalError::CorruptIndex("evidence span无效".into()))?;
        let leaf = shortest_leaf(leaves, &event, start, end)
            .ok_or_else(|| RetrievalError::CorruptIndex(format!("evidence {id} 无leaf")))?;
        if session != leaf.session_id || parse_role(&role)? != leaf.role {
            return Err(RetrievalError::CorruptIndex("evidence binding错误".into()));
        }
        let claim_node = *nodes
            .get(&(GraphNodeKind::Claim, claim.as_str()))
            .ok_or_else(|| RetrievalError::CorruptIndex("evidence claim node缺失".into()))?;
        add_edge(
            acc,
            GraphEdgeType::Support,
            claim_node,
            &leaf.node_id,
            false,
            1.0,
            Provenance {
                record_ids: vec![id],
                spans: vec![raw_span(leaf, start, end, hash)],
                reasons: vec![reason],
                ..Default::default()
            },
        )?;
    }
    let mut statement=connection.prepare("SELECT transition_id,claim_id,reason,related_claim_id FROM memory_claim_transitions WHERE related_claim_id IS NOT NULL AND reason IN ('conflicted','corrected','replaced') ORDER BY transition_id").map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    let rows = statement
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
    for row in rows {
        let (id, old, reason, new) =
            row.map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        let old_node = *nodes
            .get(&(GraphNodeKind::Claim, old.as_str()))
            .ok_or_else(|| RetrievalError::CorruptIndex("transition old claim缺失".into()))?;
        let new_node = *nodes
            .get(&(GraphNodeKind::Claim, new.as_str()))
            .ok_or_else(|| RetrievalError::CorruptIndex("transition related claim缺失".into()))?;
        let kind = if reason == "conflicted" {
            GraphEdgeType::Conflict
        } else {
            GraphEdgeType::Replacement
        };
        add_edge(
            acc,
            kind,
            old_node,
            new_node,
            kind == GraphEdgeType::Replacement,
            1.0,
            Provenance {
                record_ids: vec![id],
                reasons: vec![reason],
                ..Default::default()
            },
        )?;
    }
    Ok(())
}

fn config_hash(config: &MemoryConfig, fingerprint: &str) -> String {
    let schema = GRAPH_SCHEMA_VERSION.to_le_bytes();
    let algorithm = GRAPH_ALGORITHM_VERSION.to_le_bytes();
    let limit = (config.graph_candidate_limit as u64).to_le_bytes();
    let top = (EMBEDDING_MUTUAL_TOP_K as u64).to_le_bytes();
    let cosine = EMBEDDING_MUTUAL_MIN_COSINE.to_bits().to_le_bytes();
    let keyword = (KEYWORD_MIN_NORMALIZED_CHARS as u64).to_le_bytes();
    let rules = EDGE_RULES_VERSION.to_le_bytes();
    hash_parts(
        "hippocampus.graph.config.v1",
        &[
            &schema,
            &algorithm,
            fingerprint.as_bytes(),
            &limit,
            &top,
            &cosine,
            &keyword,
            &rules,
        ],
    )
}

fn source_hash(
    connection: &Connection,
    control_generation_sha256: &str,
    leaf: &str,
    aggregate: &str,
    fingerprint: &str,
    embeddings: &[StoredEmbedding],
    runs: &[(String, crate::model::RetrievalTrace)],
) -> RetrievalResult<String> {
    let mut fields = vec![
        control_generation_sha256.as_bytes().to_vec(),
        leaf.as_bytes().to_vec(),
        aggregate.as_bytes().to_vec(),
        fingerprint.as_bytes().to_vec(),
    ];
    for row in embeddings {
        fields.push(row.document_id.as_bytes().to_vec());
        fields.push(row.session_id.as_bytes().to_vec());
        fields.push(row.source_sha256.as_bytes().to_vec());
        fields.push(row.model.as_bytes().to_vec());
        fields.push((row.dimensions as u64).to_le_bytes().to_vec());
        for value in &row.vector {
            fields.push(value.to_bits().to_le_bytes().to_vec());
        }
    }
    for (id, trace) in runs {
        fields.push(id.as_bytes().to_vec());
        fields.push(
            serde_json::to_vec(trace).map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?,
        );
    }
    for table in [
        "events",
        "memory_entities",
        "memory_entity_mentions",
        "memory_claims",
        "memory_claim_evidence",
        "memory_claim_transitions",
    ] {
        let mut statement = connection
            .prepare(&format!("SELECT * FROM {table} ORDER BY 1"))
            .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        let count = statement.column_count();
        let rows = statement
            .query_map([], |row| {
                let mut values = Vec::new();
                for i in 0..count {
                    let value = row.get_ref(i)?;
                    values.push(match value {
                        rusqlite::types::ValueRef::Null => vec![],
                        rusqlite::types::ValueRef::Integer(v) => v.to_le_bytes().to_vec(),
                        rusqlite::types::ValueRef::Real(v) => v.to_bits().to_le_bytes().to_vec(),
                        rusqlite::types::ValueRef::Text(v) | rusqlite::types::ValueRef::Blob(v) => {
                            v.to_vec()
                        }
                    });
                }
                Ok(values)
            })
            .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        for row in rows {
            fields.extend(row.map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?);
        }
    }
    let refs = fields.iter().map(Vec::as_slice).collect::<Vec<_>>();
    Ok(hash_parts("hippocampus.graph.source.v1", &refs))
}

fn catalog_hash(nodes: &[Node], edges: &[Edge]) -> String {
    let mut fields = Vec::new();
    for n in nodes {
        fields.extend([
            n.id.as_bytes().to_vec(),
            node_name(n.kind).as_bytes().to_vec(),
            n.source_id.as_bytes().to_vec(),
            n.session_id.as_deref().unwrap_or("").as_bytes().to_vec(),
            n.granularity
                .map(granularity_name)
                .unwrap_or("")
                .as_bytes()
                .to_vec(),
            n.source_sha256.as_bytes().to_vec(),
        ]);
    }
    for e in edges {
        fields.extend([
            e.id.as_bytes().to_vec(),
            edge_name(e.kind).as_bytes().to_vec(),
            e.source.as_bytes().to_vec(),
            e.target.as_bytes().to_vec(),
            e.weight.to_bits().to_le_bytes().to_vec(),
            vec![u8::from(e.directed)],
            e.provenance_json.as_bytes().to_vec(),
            e.provenance_sha256.as_bytes().to_vec(),
        ]);
    }
    let refs = fields.iter().map(Vec::as_slice).collect::<Vec<_>>();
    hash_parts("hippocampus.graph.catalog.v1", &refs)
}

fn exact_existing_catalog(
    c: &Connection,
    nodes: &[Node],
    edges: &[Edge],
    fingerprint: &str,
    config: &str,
    source: &str,
    catalog: &str,
) -> RetrievalResult<Option<String>> {
    let meta=c.query_row("SELECT algorithm_version,vector_index_fingerprint,config_sha256,source_sha256,catalog_sha256,node_count,edge_count,materialized_at FROM memory_graph_materializations WHERE singleton=1",[],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,String>(3)?,r.get::<_,String>(4)?,r.get::<_,i64>(5)?,r.get::<_,i64>(6)?,r.get::<_,String>(7)?))).optional().map_err(|e|RetrievalError::CorruptIndex(format!("读取图 materialization 失败：{e}")))?;
    let Some(meta) = meta else { return Ok(None) };
    if meta.1 != fingerprint || meta.2 != config || meta.3 != source {
        return Ok(None);
    }
    if meta.0 != GRAPH_ALGORITHM_VERSION {
        return Err(RetrievalError::CorruptIndex(format!(
            "图 materialization algorithm_version 异常：{}",
            meta.0
        )));
    }
    if meta.4 != catalog
        || usize::try_from(meta.5).ok() != Some(nodes.len())
        || usize::try_from(meta.6).ok() != Some(edges.len())
    {
        return Err(RetrievalError::CorruptIndex(
            "图 materialization catalog 或计数损坏".into(),
        ));
    }
    let actual_nodes = read_nodes(c)?;
    let actual_edges = read_edges(c)?;
    if actual_nodes != nodes || actual_edges != edges {
        return Err(RetrievalError::CorruptIndex(
            "持久化图节点或边与当前来源不一致".into(),
        ));
    }
    if catalog_hash(&actual_nodes, &actual_edges) != catalog {
        return Err(RetrievalError::CorruptIndex(
            "持久化图 catalog hash 损坏".into(),
        ));
    }
    if DateTime::parse_from_rfc3339(&meta.7).is_err() {
        return Err(RetrievalError::CorruptIndex(
            "图 materialized_at 不是 RFC3339 时间".into(),
        ));
    }
    Ok(Some(meta.7))
}

fn read_nodes(c: &Connection) -> RetrievalResult<Vec<Node>> {
    let mut s=c.prepare("SELECT node_id,node_kind,source_id,session_id,granularity,source_sha256 FROM memory_graph_nodes ORDER BY node_id").map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    s.query_map([], |r| {
        let kind = match r.get::<_, String>(1)?.as_str() {
            "document" => GraphNodeKind::Document,
            "entity" => GraphNodeKind::Entity,
            "claim" => GraphNodeKind::Claim,
            _ => return Err(rusqlite::Error::InvalidQuery),
        };
        let granularity = match r.get::<_, Option<String>>(4)?.as_deref() {
            None => None,
            Some("message") => Some(RetrievalDocumentGranularity::Message),
            Some("fragment") => Some(RetrievalDocumentGranularity::Fragment),
            Some("episode") => Some(RetrievalDocumentGranularity::Episode),
            Some("session") => Some(RetrievalDocumentGranularity::Session),
            _ => return Err(rusqlite::Error::InvalidQuery),
        };
        Ok(Node {
            id: r.get(0)?,
            kind,
            source_id: r.get(2)?,
            session_id: r.get(3)?,
            granularity,
            source_sha256: r.get(5)?,
        })
    })
    .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?
    .collect::<rusqlite::Result<Vec<_>>>()
    .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))
}
fn read_edges(c: &Connection) -> RetrievalResult<Vec<Edge>> {
    let mut s=c.prepare("SELECT edge_id,edge_type,source_node_id,target_node_id,weight,directed,provenance_json,provenance_sha256 FROM memory_graph_edges").map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    let mut out = Vec::new();
    let rows = s
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, f64>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, String>(7)?,
            ))
        })
        .map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
    for row in rows {
        let (id, name, source, target, weight, directed_raw, json, hash) =
            row.map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        let kind = parse_edge(&name)?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(RetrievalError::CorruptIndex(format!(
                "图边 {id} 权重不是有限正数"
            )));
        }
        if source == target {
            return Err(RetrievalError::CorruptIndex(format!("图边 {id} 包含自环")));
        }
        let directed = match directed_raw {
            0 => false,
            1 => true,
            _ => {
                return Err(RetrievalError::CorruptIndex(format!(
                    "图边 {id} directed 值无效"
                )));
            }
        };
        if directed != (kind == GraphEdgeType::Replacement) {
            return Err(RetrievalError::CorruptIndex(format!(
                "图边 {id} directed 与边类型不匹配"
            )));
        }
        if !directed && source >= target {
            return Err(RetrievalError::CorruptIndex(format!(
                "无向图边 {id} 端点未规范排序"
            )));
        }
        let _: Provenance =
            serde_json::from_str(&json).map_err(|e| RetrievalError::CorruptIndex(e.to_string()))?;
        if hash != bytes_sha256(json.as_bytes()) {
            return Err(RetrievalError::CorruptIndex(
                "图 provenance hash损坏".into(),
            ));
        }
        let expected_id = format!(
            "gedge_{}",
            hash_parts(
                "hippocampus.graph.edge-id.v1",
                &[
                    edge_name(kind).as_bytes(),
                    source.as_bytes(),
                    target.as_bytes(),
                ],
            )
        );
        if id != expected_id {
            return Err(RetrievalError::CorruptIndex(format!(
                "图边 {id} 的确定性 ID 损坏"
            )));
        }
        out.push(Edge {
            id,
            kind,
            source,
            target,
            weight,
            directed,
            provenance_json: json,
            provenance_sha256: hash,
        });
    }
    out.sort_by(|left, right| {
        (left.kind, &left.source, &left.target).cmp(&(right.kind, &right.source, &right.target))
    });
    Ok(out)
}
fn parse_edge(v: &str) -> RetrievalResult<GraphEdgeType> {
    [
        GraphEdgeType::Reply,
        GraphEdgeType::Adjacent,
        GraphEdgeType::EpisodeMember,
        GraphEdgeType::EntityMention,
        GraphEdgeType::SharedEntity,
        GraphEdgeType::KeywordCooccurrence,
        GraphEdgeType::EmbeddingMutualTopK,
        GraphEdgeType::CommonRecall,
        GraphEdgeType::Support,
        GraphEdgeType::Conflict,
        GraphEdgeType::Replacement,
    ]
    .into_iter()
    .find(|k| edge_name(*k) == v)
    .ok_or_else(|| RetrievalError::CorruptIndex(format!("未知图边类型 {v}")))
}

fn replace_catalog(
    c: &Connection,
    nodes: &[Node],
    edges: &[Edge],
    metadata: (&str, &str, &str, &str),
    time: &str,
) -> RetrievalResult<()> {
    let (fingerprint, config, source, catalog) = metadata;
    c.execute_batch("DELETE FROM memory_graph_edges;DELETE FROM memory_graph_nodes;DELETE FROM memory_graph_materializations;").map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    for n in nodes {
        c.execute("INSERT INTO memory_graph_nodes(node_id,node_kind,source_id,session_id,granularity,source_sha256)VALUES(?1,?2,?3,?4,?5,?6)",params![n.id,node_name(n.kind),n.source_id,n.session_id,n.granularity.map(granularity_name),n.source_sha256]).map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    }
    for e in edges {
        c.execute("INSERT INTO memory_graph_edges(edge_id,edge_type,source_node_id,target_node_id,weight,directed,provenance_json,provenance_sha256)VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",params![e.id,edge_name(e.kind),e.source,e.target,e.weight,e.directed,e.provenance_json,e.provenance_sha256]).map_err(|x|RetrievalError::CorruptIndex(x.to_string()))?;
    }
    c.execute("INSERT INTO memory_graph_materializations(singleton,algorithm_version,vector_index_fingerprint,config_sha256,source_sha256,catalog_sha256,node_count,edge_count,materialized_at)VALUES(1,1,?1,?2,?3,?4,?5,?6,?7)",params![fingerprint,config,source,catalog,nodes.len() as i64,edges.len() as i64,time]).map_err(|e|RetrievalError::CorruptIndex(e.to_string()))?;
    Ok(())
}

fn report(
    changed: bool,
    nodes: &[Node],
    edges: &[Edge],
    source_sha256: String,
    catalog_sha256: String,
    vector_index_fingerprint: String,
    materialized_at: String,
) -> GraphMaterializationReport {
    let mut node_counts = BTreeMap::new();
    for kind in [
        GraphNodeKind::Document,
        GraphNodeKind::Entity,
        GraphNodeKind::Claim,
    ] {
        node_counts.insert(kind, 0);
    }
    for node in nodes {
        *node_counts.entry(node.kind).or_default() += 1;
    }
    let mut edge_counts = BTreeMap::new();
    for kind in [
        GraphEdgeType::Reply,
        GraphEdgeType::Adjacent,
        GraphEdgeType::EpisodeMember,
        GraphEdgeType::EntityMention,
        GraphEdgeType::SharedEntity,
        GraphEdgeType::KeywordCooccurrence,
        GraphEdgeType::EmbeddingMutualTopK,
        GraphEdgeType::CommonRecall,
        GraphEdgeType::Support,
        GraphEdgeType::Conflict,
        GraphEdgeType::Replacement,
    ] {
        edge_counts.insert(kind, 0);
    }
    for edge in edges {
        *edge_counts.entry(edge.kind).or_default() += 1;
    }
    GraphMaterializationReport {
        changed,
        node_count: nodes.len(),
        edge_count: edges.len(),
        node_counts,
        edge_counts,
        source_sha256,
        catalog_sha256,
        vector_index_fingerprint,
        materialized_at,
    }
}

use rusqlite::OptionalExtension;
