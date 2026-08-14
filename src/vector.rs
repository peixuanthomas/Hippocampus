use std::collections::HashSet;
use std::fmt;

use hnsw_rs::prelude::{DistCosine, Hnsw};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::config::MemoryConfig;
use crate::model::RetrievalDocumentGranularity;

pub const EMBEDDING_PREPROCESSING_VERSION: u32 = 2;

/// The complete compatibility contract for a persisted vector index.
///
/// HNSW tuning is included deliberately: changing it requires rebuilding the
/// in-memory index, so embeddings written under a different setting are not
/// considered compatible with the current derived index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorIndexSpec {
    pub model: String,
    pub dimensions: usize,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VectorError {
    #[error("memory 配置无效：{0}")]
    InvalidConfig(String),
    #[error("向量索引模型不能为空")]
    EmptyModel,
    #[error("向量维度必须为非零值")]
    ZeroDimensions,
    #[error("向量维度必须在 32..=4096 之间，实际为 {0}")]
    DimensionsOutOfRange(usize),
    #[error("HNSW M 必须在 2..=64 之间，实际为 {0}")]
    HnswMOutOfRange(usize),
    #[error("HNSW ef_construction 必须在 M..=4096 之间，实际为 {0}")]
    HnswEfConstructionOutOfRange(usize),
    #[error("HNSW ef_search 必须在 1..=4096 之间，实际为 {0}")]
    HnswEfSearchOutOfRange(usize),
    #[error("向量不能为空")]
    EmptyVector,
    #[error("向量维度不匹配：期望 {expected}，实际 {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("向量包含非有限值")]
    NonFiniteValue,
    #[error("向量范数必须大于零")]
    ZeroNorm,
    #[error("向量池不能为空")]
    EmptyPool,
    #[error("向量权重必须为正整数")]
    ZeroWeight,
    #[error("向量运算结果超出有限数值范围")]
    NonFiniteComputation,
    #[error("向量字节长度无效：期望 {expected}，实际 {actual}")]
    InvalidByteLength { expected: usize, actual: usize },
    #[error("向量维度溢出")]
    DimensionOverflow,
    #[error("持久化向量包含重复文档 ID：{0}")]
    DuplicateDocumentId(String),
    #[error("持久化向量与索引不兼容（文档 {document_id}）：{reason}")]
    IncompatibleStoredEmbedding { document_id: String, reason: String },
    #[error("HNSW 返回了无效的外部 ID：{0}")]
    InvalidExternalId(usize),
}

impl VectorIndexSpec {
    pub fn from_config(config: &MemoryConfig) -> Result<Self, VectorError> {
        config
            .validate()
            .map_err(|error| VectorError::InvalidConfig(error.to_string()))?;
        let spec = Self {
            model: config.embedding_model.clone(),
            dimensions: config.embedding_dimensions,
            hnsw_m: config.hnsw_m,
            hnsw_ef_construction: config.hnsw_ef_construction,
            hnsw_ef_search: config.hnsw_ef_search,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), VectorError> {
        if self.model.trim().is_empty() {
            return Err(VectorError::EmptyModel);
        }
        if self.dimensions == 0 {
            return Err(VectorError::ZeroDimensions);
        }
        if !(32..=4_096).contains(&self.dimensions) {
            return Err(VectorError::DimensionsOutOfRange(self.dimensions));
        }
        if !(2..=64).contains(&self.hnsw_m) {
            return Err(VectorError::HnswMOutOfRange(self.hnsw_m));
        }
        if !(self.hnsw_m..=4_096).contains(&self.hnsw_ef_construction) {
            return Err(VectorError::HnswEfConstructionOutOfRange(
                self.hnsw_ef_construction,
            ));
        }
        if !(1..=4_096).contains(&self.hnsw_ef_search) {
            return Err(VectorError::HnswEfSearchOutOfRange(self.hnsw_ef_search));
        }
        self.dimensions
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(VectorError::DimensionOverflow)?;
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String, VectorError> {
        self.validate()?;
        let dimensions =
            u64::try_from(self.dimensions).map_err(|_| VectorError::DimensionOverflow)?;
        let hnsw_m = u64::try_from(self.hnsw_m).map_err(|_| VectorError::DimensionOverflow)?;
        let hnsw_ef_construction =
            u64::try_from(self.hnsw_ef_construction).map_err(|_| VectorError::DimensionOverflow)?;
        let hnsw_ef_search =
            u64::try_from(self.hnsw_ef_search).map_err(|_| VectorError::DimensionOverflow)?;
        let mut hasher = Sha256::new();
        hasher.update(b"hippocampus-vector-index-v2\0");
        for value in [
            &EMBEDDING_PREPROCESSING_VERSION.to_le_bytes()[..],
            self.model.as_bytes(),
            &dimensions.to_le_bytes(),
            &hnsw_m.to_le_bytes(),
            &hnsw_ef_construction.to_le_bytes(),
            &hnsw_ef_search.to_le_bytes(),
        ] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

fn normalized_f64(vector: &[f32]) -> Result<Vec<f64>, VectorError> {
    if vector.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    let mut squared_norm = 0.0_f64;
    for &value in vector {
        if !value.is_finite() {
            return Err(VectorError::NonFiniteValue);
        }
        let value = f64::from(value);
        squared_norm += value * value;
        if !squared_norm.is_finite() {
            return Err(VectorError::NonFiniteComputation);
        }
    }
    if squared_norm == 0.0 {
        return Err(VectorError::ZeroNorm);
    }
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(VectorError::NonFiniteComputation);
    }
    Ok(vector
        .iter()
        .map(|&value| f64::from(value) / norm)
        .collect())
}

fn normalized_f32(vector: &[f64]) -> Result<Vec<f32>, VectorError> {
    let mut squared_norm = 0.0_f64;
    for &value in vector {
        if !value.is_finite() {
            return Err(VectorError::NonFiniteComputation);
        }
        squared_norm += value * value;
        if !squared_norm.is_finite() {
            return Err(VectorError::NonFiniteComputation);
        }
    }
    if squared_norm == 0.0 {
        return Err(VectorError::ZeroNorm);
    }
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(VectorError::NonFiniteComputation);
    }
    vector
        .iter()
        .map(|value| {
            let normalized = (value / norm) as f32;
            if normalized.is_finite() {
                Ok(normalized)
            } else {
                Err(VectorError::NonFiniteComputation)
            }
        })
        .collect()
}

pub fn l2_normalize(vector: &[f32]) -> Result<Vec<f32>, VectorError> {
    normalized_f32(&normalized_f64(vector)?)
}

pub fn weighted_pool(vectors: &[(&[f32], usize)]) -> Result<Vec<f32>, VectorError> {
    let Some((first, _)) = vectors.first() else {
        return Err(VectorError::EmptyPool);
    };
    if first.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    let dimensions = first.len();
    let mut pooled = vec![0.0_f64; dimensions];
    for &(vector, weight) in vectors {
        if weight == 0 {
            return Err(VectorError::ZeroWeight);
        }
        if vector.len() != dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: dimensions,
                actual: vector.len(),
            });
        }
        let normalized = normalized_f64(vector)?;
        let weight = weight as f64;
        if !weight.is_finite() {
            return Err(VectorError::NonFiniteComputation);
        }
        for (output, value) in pooled.iter_mut().zip(normalized) {
            *output += value * weight;
            if !output.is_finite() {
                return Err(VectorError::NonFiniteComputation);
            }
        }
    }
    normalized_f32(&pooled)
}

pub fn equal_mean(vectors: &[&[f32]]) -> Result<Vec<f32>, VectorError> {
    let weighted = vectors
        .iter()
        .map(|vector| (*vector, 1_usize))
        .collect::<Vec<_>>();
    weighted_pool(&weighted)
}

const HNSW_MAX_LAYER: usize = 16;
const UNIT_NORM_TOLERANCE: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchHit {
    pub document_id: String,
    pub session_id: String,
    pub granularity: RetrievalDocumentGranularity,
    pub source_sha256: String,
    pub cosine_similarity: f32,
    pub cosine_distance: f32,
}

#[derive(Clone)]
struct IndexedEmbedding {
    document_id: String,
    session_id: String,
    granularity: RetrievalDocumentGranularity,
    source_sha256: String,
    vector: Vec<f32>,
}

pub struct HnswVectorIndex {
    spec: VectorIndexSpec,
    fingerprint: String,
    embeddings: Vec<IndexedEmbedding>,
    index: Hnsw<'static, f32, DistCosine>,
}

impl fmt::Debug for HnswVectorIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HnswVectorIndex")
            .field("spec", &self.spec)
            .field("fingerprint", &self.fingerprint)
            .field("len", &self.len())
            .finish()
    }
}

impl HnswVectorIndex {
    pub fn rebuild(
        spec: VectorIndexSpec,
        mut embeddings: Vec<StoredEmbedding>,
    ) -> Result<Self, VectorError> {
        spec.validate()?;
        let fingerprint = spec.fingerprint()?;
        embeddings.sort_by(|left, right| left.document_id.cmp(&right.document_id));

        for rows in embeddings.windows(2) {
            if rows[0].document_id == rows[1].document_id {
                return Err(VectorError::DuplicateDocumentId(
                    rows[0].document_id.clone(),
                ));
            }
        }

        let mut retained = Vec::with_capacity(embeddings.len());
        for row in embeddings {
            validate_stored_embedding(&spec, &fingerprint, &row)?;
            retained.push(IndexedEmbedding {
                document_id: row.document_id,
                session_id: row.session_id,
                granularity: row.granularity,
                source_sha256: row.source_sha256,
                vector: row.vector,
            });
        }

        let mut index = Hnsw::new(
            spec.hnsw_m,
            retained.len().max(1),
            HNSW_MAX_LAYER,
            spec.hnsw_ef_construction,
            DistCosine,
        );
        for (external_id, embedding) in retained.iter().enumerate() {
            index.insert((&embedding.vector, external_id));
        }
        index.set_searching_mode(true);

        Ok(Self {
            spec,
            fingerprint,
            embeddings: retained,
            index,
        })
    }

    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<VectorSearchHit>, VectorError> {
        if limit == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        if query.len() != self.spec.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.spec.dimensions,
                actual: query.len(),
            });
        }
        let query = l2_normalize(query)?;
        let requested = limit.min(self.len());
        let neighbours = self
            .index
            .search(&query, requested, self.spec.hnsw_ef_search);
        let mut seen = HashSet::with_capacity(neighbours.len());
        let mut hits = Vec::with_capacity(neighbours.len());

        for neighbour in neighbours {
            let external_id = neighbour.d_id;
            let embedding = self
                .embeddings
                .get(external_id)
                .ok_or(VectorError::InvalidExternalId(external_id))?;
            if !seen.insert(external_id) {
                continue;
            }
            let similarity = exact_cosine(&query, &embedding.vector)?;
            hits.push(VectorSearchHit {
                document_id: embedding.document_id.clone(),
                session_id: embedding.session_id.clone(),
                granularity: embedding.granularity,
                source_sha256: embedding.source_sha256.clone(),
                cosine_similarity: similarity,
                cosine_distance: 1.0 - similarity,
            });
        }

        hits.sort_by(|left, right| {
            right
                .cosine_similarity
                .total_cmp(&left.cosine_similarity)
                .then_with(|| left.document_id.cmp(&right.document_id))
        });
        hits.truncate(requested);
        Ok(hits)
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    pub fn spec(&self) -> &VectorIndexSpec {
        &self.spec
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

fn validate_stored_embedding(
    spec: &VectorIndexSpec,
    fingerprint: &str,
    row: &StoredEmbedding,
) -> Result<(), VectorError> {
    let incompatible = |reason: &str| VectorError::IncompatibleStoredEmbedding {
        document_id: row.document_id.clone(),
        reason: reason.to_owned(),
    };
    if row.model != spec.model {
        return Err(incompatible("模型不匹配"));
    }
    if row.dimensions != spec.dimensions {
        return Err(incompatible("声明维度不匹配"));
    }
    if row.index_fingerprint != fingerprint {
        return Err(incompatible("索引指纹不匹配"));
    }
    if row.vector.len() != spec.dimensions {
        return Err(incompatible("向量长度不匹配"));
    }

    let mut squared_norm = 0.0_f64;
    for &value in &row.vector {
        if !value.is_finite() {
            return Err(incompatible("向量包含非有限值"));
        }
        let value = f64::from(value);
        squared_norm += value * value;
        if !squared_norm.is_finite() {
            return Err(incompatible("向量范数计算结果非有限"));
        }
    }
    if squared_norm == 0.0 {
        return Err(incompatible("向量范数为零"));
    }
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || (norm - 1.0).abs() > UNIT_NORM_TOLERANCE {
        return Err(incompatible("向量未按单位 L2 范数归一化"));
    }
    Ok(())
}

fn exact_cosine(left: &[f32], right: &[f32]) -> Result<f32, VectorError> {
    let mut dot = 0.0_f64;
    let mut left_squared_norm = 0.0_f64;
    let mut right_squared_norm = 0.0_f64;
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        left_squared_norm += left * left;
        right_squared_norm += right * right;
        if !dot.is_finite() || !left_squared_norm.is_finite() || !right_squared_norm.is_finite() {
            return Err(VectorError::NonFiniteComputation);
        }
    }
    let denominator = (left_squared_norm * right_squared_norm).sqrt();
    if !denominator.is_finite() || denominator == 0.0 {
        return Err(VectorError::NonFiniteComputation);
    }
    let similarity = (dot / denominator).clamp(-1.0, 1.0) as f32;
    if similarity.is_finite() {
        Ok(similarity)
    } else {
        Err(VectorError::NonFiniteComputation)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingWrite {
    pub document_id: String,
    pub expected_source_sha256: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredEmbedding {
    pub document_id: String,
    pub session_id: String,
    pub granularity: RetrievalDocumentGranularity,
    pub source_sha256: String,
    pub model: String,
    pub dimensions: usize,
    pub index_fingerprint: String,
    pub vector: Vec<f32>,
    pub embedded_at: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbeddingCoverage {
    pub total: usize,
    pub compatible: usize,
    pub stale: usize,
}

pub fn encode_f32_le(vector: &[f32]) -> Result<Vec<u8>, VectorError> {
    if vector.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    let byte_len = vector
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(VectorError::DimensionOverflow)?;
    let mut bytes = Vec::with_capacity(byte_len);
    for value in vector {
        if !value.is_finite() {
            return Err(VectorError::NonFiniteValue);
        }
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

pub fn decode_f32_le(bytes: &[u8], expected_dimensions: usize) -> Result<Vec<f32>, VectorError> {
    if expected_dimensions == 0 {
        return Err(VectorError::ZeroDimensions);
    }
    let expected = expected_dimensions
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(VectorError::DimensionOverflow)?;
    if bytes.len() != expected {
        return Err(VectorError::InvalidByteLength {
            expected,
            actual: bytes.len(),
        });
    }
    let mut vector = Vec::with_capacity(expected_dimensions);
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(VectorError::NonFiniteValue);
        }
        vector.push(value);
    }
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn little_endian_codec_preserves_bits() {
        let input = [1.0_f32, -0.0, f32::from_bits(0x3eaaaaab)];
        let bytes = encode_f32_le(&input).unwrap();
        assert_eq!(bytes, [0, 0, 128, 63, 0, 0, 0, 128, 171, 170, 170, 62]);
        let output = decode_f32_le(&bytes, input.len()).unwrap();
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            input
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn codec_rejects_invalid_vectors() {
        assert!(matches!(encode_f32_le(&[]), Err(VectorError::EmptyVector)));
        assert!(matches!(
            decode_f32_le(&[], 0),
            Err(VectorError::ZeroDimensions)
        ));
        assert!(matches!(
            decode_f32_le(&[0, 0, 0, 0], 2),
            Err(VectorError::InvalidByteLength { .. })
        ));
        assert!(matches!(
            encode_f32_le(&[f32::NAN]),
            Err(VectorError::NonFiniteValue)
        ));
        assert!(matches!(
            encode_f32_le(&[f32::INFINITY]),
            Err(VectorError::NonFiniteValue)
        ));
        assert!(matches!(
            decode_f32_le(&f32::NAN.to_le_bytes(), 1),
            Err(VectorError::NonFiniteValue)
        ));
    }

    #[test]
    fn fingerprint_changes_for_every_compatibility_field() {
        let base = VectorIndexSpec {
            model: "qwen3-embedding:8b".into(),
            dimensions: 1024,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
        };
        let fingerprint = base.fingerprint().unwrap();
        assert_eq!(
            fingerprint,
            "90cff08d6771a56d92b5bd472bcb215b4cc144dc39280dc26edbe2b9c80c7e81"
        );
        assert_ne!(
            fingerprint,
            "c591bf40346074f1407af6274f5be8e6889b73f549bd0c9a20a288d870fe16de"
        );
        let mut changed = base.clone();
        changed.model = "other".into();
        assert_ne!(fingerprint, changed.fingerprint().unwrap());
        changed = base.clone();
        changed.dimensions = 512;
        assert_ne!(fingerprint, changed.fingerprint().unwrap());
        changed = base.clone();
        changed.hnsw_m = 24;
        assert_ne!(fingerprint, changed.fingerprint().unwrap());
        changed = base.clone();
        changed.hnsw_ef_construction = 300;
        assert_ne!(fingerprint, changed.fingerprint().unwrap());
        changed = base.clone();
        changed.hnsw_ef_search = 96;
        assert_ne!(fingerprint, changed.fingerprint().unwrap());
        assert_eq!(fingerprint, base.fingerprint().unwrap());
    }

    fn assert_unit(vector: &[f32]) {
        assert!(vector.iter().all(|value| value.is_finite()));
        let norm = vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() <= 1e-5, "norm was {norm}");
    }

    #[test]
    fn normalization_handles_basis_and_extreme_finite_values() {
        assert_eq!(l2_normalize(&[0.0, 1.0]).unwrap(), vec![0.0, 1.0]);
        assert_eq!(l2_normalize(&[3.0, 4.0]).unwrap(), vec![0.6, 0.8]);
        for input in [[f32::MAX, f32::MAX], [f32::MIN_POSITIVE, f32::MIN_POSITIVE]] {
            assert_unit(&l2_normalize(&input).unwrap());
        }
    }

    #[test]
    fn normalization_rejects_zero_and_non_finite_vectors() {
        assert_eq!(l2_normalize(&[]), Err(VectorError::EmptyVector));
        assert_eq!(l2_normalize(&[0.0, -0.0]), Err(VectorError::ZeroNorm));
        assert_eq!(
            l2_normalize(&[f32::INFINITY]),
            Err(VectorError::NonFiniteValue)
        );
        assert_eq!(l2_normalize(&[f32::NAN]), Err(VectorError::NonFiniteValue));
    }

    #[test]
    fn pooling_validates_inputs() {
        assert_eq!(weighted_pool(&[]), Err(VectorError::EmptyPool));
        assert_eq!(weighted_pool(&[(&[], 1)]), Err(VectorError::EmptyVector));
        assert_eq!(weighted_pool(&[(&[1.0], 0)]), Err(VectorError::ZeroWeight));
        assert_eq!(
            weighted_pool(&[(&[1.0, 0.0], 1), (&[1.0], 1)]),
            Err(VectorError::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn weighted_pool_reflects_unequal_coverage() {
        let pooled = weighted_pool(&[(&[1.0, 0.0], 240), (&[0.0, 1.0], 1)]).unwrap();
        assert!(pooled[0] > 0.999 && pooled[1] > 0.0 && pooled[1] < 0.005);
        assert_unit(&pooled);
    }

    #[test]
    fn equal_mean_ignores_input_magnitude() {
        let pooled = equal_mean(&[&[10.0, 0.0], &[0.0, 0.001]]).unwrap();
        let expected = 0.5_f32.sqrt();
        assert!((pooled[0] - expected).abs() <= 1e-6);
        assert!((pooled[1] - expected).abs() <= 1e-6);
        assert_unit(&pooled);
    }

    #[test]
    fn spec_validation_matches_memory_config_bounds() {
        let base = VectorIndexSpec {
            model: "qwen3-embedding:8b".into(),
            dimensions: 1024,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
        };
        for (spec, expected) in [
            (
                VectorIndexSpec {
                    dimensions: 31,
                    ..base.clone()
                },
                VectorError::DimensionsOutOfRange(31),
            ),
            (
                VectorIndexSpec {
                    hnsw_m: 1,
                    ..base.clone()
                },
                VectorError::HnswMOutOfRange(1),
            ),
            (
                VectorIndexSpec {
                    hnsw_ef_construction: 15,
                    ..base.clone()
                },
                VectorError::HnswEfConstructionOutOfRange(15),
            ),
            (
                VectorIndexSpec {
                    hnsw_ef_search: 4_097,
                    ..base.clone()
                },
                VectorError::HnswEfSearchOutOfRange(4_097),
            ),
        ] {
            assert_eq!(spec.validate(), Err(expected));
        }
    }
}
