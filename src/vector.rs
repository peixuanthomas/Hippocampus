use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::config::MemoryConfig;
use crate::model::RetrievalDocumentGranularity;

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
    #[error("向量字节长度无效：期望 {expected}，实际 {actual}")]
    InvalidByteLength { expected: usize, actual: usize },
    #[error("向量维度溢出")]
    DimensionOverflow,
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
        hasher.update(b"hippocampus-vector-index-v1\0");
        for value in [
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
