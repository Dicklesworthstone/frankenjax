#![forbid(unsafe_code)]

use asupersync::EncodingConfig;
use asupersync::decoding::{DecodingConfig, DecodingPipeline};
use asupersync::encoding::EncodingPipeline;
use asupersync::security::{AuthenticatedSymbol, AuthenticationTag};
use asupersync::types::resource::{PoolConfig, SymbolPool};
use asupersync::types::{ObjectId, ObjectParams, Symbol, SymbolId, SymbolKind};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SidecarConfig {
    pub symbol_size: u16,
    pub max_block_size: usize,
    pub repair_overhead: f64,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            symbol_size: 256,
            max_block_size: 1024 * 1024,
            repair_overhead: 1.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarSymbolRecord {
    pub sbn: u8,
    pub esi: u32,
    pub kind: String,
    pub data_b64: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SidecarManifest {
    pub schema_version: String,
    pub artifact_path: String,
    pub artifact_sha256_hex: String,
    pub artifact_size: usize,
    pub generated_at_unix_ms: u128,
    pub object_id_high: u64,
    pub object_id_low: u64,
    pub symbol_size: u16,
    pub max_block_size: usize,
    pub repair_overhead: f64,
    pub source_blocks: u8,
    pub symbols_per_block: u16,
    pub total_symbols: usize,
    pub source_symbols: usize,
    pub repair_symbols: usize,
    pub symbols: Vec<SidecarSymbolRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScrubReport {
    pub sidecar_path: String,
    pub artifact_path: String,
    pub expected_sha256_hex: String,
    pub decoded_sha256_hex: String,
    pub decoded_matches_expected: bool,
    pub total_symbols: usize,
    pub source_symbols: usize,
    pub repair_symbols: usize,
    pub generated_at_unix_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeProof {
    pub proof_id: String,
    pub sidecar_path: String,
    pub artifact_path: String,
    pub expected_sha256_hex: String,
    pub decoded_sha256_hex: Option<String>,
    pub dropped_symbols: Vec<String>,
    pub attempted_symbol_count: usize,
    pub recovered: bool,
    pub details: String,
    pub generated_at_unix_ms: u128,
}

#[derive(Debug)]
pub enum DurabilityError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidConfig(String),
    Encode(String),
    Decode(String),
    Integrity(String),
}

impl std::fmt::Display for DurabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::InvalidConfig(detail) => write!(f, "invalid durability config: {detail}"),
            Self::Encode(detail) => write!(f, "encode error: {detail}"),
            Self::Decode(detail) => write!(f, "decode error: {detail}"),
            Self::Integrity(detail) => write!(f, "integrity error: {detail}"),
        }
    }
}

impl std::error::Error for DurabilityError {}

impl From<std::io::Error> for DurabilityError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for DurabilityError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

pub fn encode_artifact_to_sidecar(
    artifact_path: &Path,
    sidecar_path: &Path,
    config: &SidecarConfig,
) -> Result<SidecarManifest, DurabilityError> {
    validate_config(config)?;

    let data = fs::read(artifact_path)?;
    let artifact_sha256_hex = sha256_hex(&data);
    let object_id = object_id_from_sha256(&artifact_sha256_hex)?;

    let source_blocks = compute_source_blocks(data.len(), config.max_block_size)?;
    let symbols_per_block =
        compute_symbols_per_block(data.len(), config.max_block_size, config.symbol_size)?;

    let mut encoder = EncodingPipeline::new(
        EncodingConfig {
            repair_overhead: config.repair_overhead,
            max_block_size: config.max_block_size,
            symbol_size: config.symbol_size,
            ..EncodingConfig::default()
        },
        SymbolPool::new(PoolConfig::default()),
    );

    let encoded_symbols = encoder
        .encode(object_id, &data)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| DurabilityError::Encode(err.to_string()))?;

    let stats = encoder.stats();
    let mut symbols = encoded_symbols
        .into_iter()
        .map(|encoded| {
            let symbol = encoded.into_symbol();
            SidecarSymbolRecord {
                sbn: symbol.sbn(),
                esi: symbol.esi(),
                kind: match symbol.kind() {
                    SymbolKind::Source => "source".to_owned(),
                    SymbolKind::Repair => "repair".to_owned(),
                },
                data_b64: BASE64_STANDARD.encode(symbol.data()),
            }
        })
        .collect::<Vec<_>>();
    symbols.sort_by_key(|record| (record.sbn, record.esi));

    let manifest = SidecarManifest {
        schema_version: "frankenjax.sidecar.v1".to_owned(),
        artifact_path: artifact_path.display().to_string(),
        artifact_sha256_hex,
        artifact_size: data.len(),
        generated_at_unix_ms: now_unix_ms(),
        object_id_high: object_id.high(),
        object_id_low: object_id.low(),
        symbol_size: config.symbol_size,
        max_block_size: config.max_block_size,
        repair_overhead: config.repair_overhead,
        source_blocks,
        symbols_per_block,
        total_symbols: symbols.len(),
        source_symbols: stats.source_symbols,
        repair_symbols: stats.repair_symbols,
        symbols,
    };

    let json = serde_json::to_string_pretty(&manifest)?;
    if let Some(parent) = sidecar_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(sidecar_path, json)?;

    Ok(manifest)
}

pub fn scrub_sidecar(
    sidecar_path: &Path,
    artifact_path: &Path,
    scrub_report_path: &Path,
) -> Result<ScrubReport, DurabilityError> {
    let manifest = read_sidecar_manifest(sidecar_path)?;
    let original_data = fs::read(artifact_path)?;

    let decoded = decode_from_sidecar_records(&manifest, &manifest.symbols)?;
    let decoded_hash = sha256_hex(&decoded);
    let expected_hash = sha256_hex(&original_data);

    let report = ScrubReport {
        sidecar_path: sidecar_path.display().to_string(),
        artifact_path: artifact_path.display().to_string(),
        expected_sha256_hex: expected_hash.clone(),
        decoded_sha256_hex: decoded_hash.clone(),
        decoded_matches_expected: decoded_hash == expected_hash,
        total_symbols: manifest.total_symbols,
        source_symbols: manifest.source_symbols,
        repair_symbols: manifest.repair_symbols,
        generated_at_unix_ms: now_unix_ms(),
    };

    if let Some(parent) = scrub_report_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(scrub_report_path, serde_json::to_string_pretty(&report)?)?;

    if !report.decoded_matches_expected {
        return Err(DurabilityError::Integrity(
            "decoded bytes do not match artifact bytes".to_owned(),
        ));
    }

    Ok(report)
}

pub fn generate_decode_proof(
    sidecar_path: &Path,
    artifact_path: &Path,
    proof_path: &Path,
    drop_source_count: usize,
) -> Result<DecodeProof, DurabilityError> {
    let manifest = read_sidecar_manifest(sidecar_path)?;
    let original_data = fs::read(artifact_path)?;
    let expected_hash = sha256_hex(&original_data);

    let (mut dropped_symbols, mut retained) =
        drop_symbols_by_kind(&manifest.symbols, "source", drop_source_count);
    let mut details_prefix = format!("drop-kind=source count={}", dropped_symbols.len());

    let mut decode_attempt = decode_from_sidecar_records(&manifest, &retained);
    if decode_attempt.is_err() {
        let repair_drop_count = drop_source_count.min(
            manifest
                .symbols
                .iter()
                .filter(|record| record.kind == "repair")
                .count(),
        );
        if repair_drop_count > 0 {
            let (repair_dropped, repair_retained) =
                drop_symbols_by_kind(&manifest.symbols, "repair", repair_drop_count);
            if let Ok(decoded) = decode_from_sidecar_records(&manifest, &repair_retained) {
                dropped_symbols = repair_dropped;
                retained = repair_retained;
                decode_attempt = Ok(decoded);
                details_prefix = format!("drop-kind=repair count={}", dropped_symbols.len());
            }
        }
    }

    let (decoded_sha256_hex, recovered, details) = match decode_attempt {
        Ok(decoded) => {
            let decoded_hash = sha256_hex(&decoded);
            (
                Some(decoded_hash.clone()),
                decoded_hash == expected_hash,
                if decoded_hash == expected_hash {
                    format!("{details_prefix}; recovered successfully from retained symbols")
                } else {
                    format!("{details_prefix}; decoded bytes differed from expected hash")
                },
            )
        }
        Err(err) => (
            None,
            false,
            format!("{details_prefix}; decode failed from retained symbols: {err}"),
        ),
    };

    let proof = DecodeProof {
        proof_id: format!(
            "decode-proof-{:016x}",
            fnv1a_64(format!("{}:{}", manifest.artifact_sha256_hex, drop_source_count).as_bytes())
        ),
        sidecar_path: sidecar_path.display().to_string(),
        artifact_path: artifact_path.display().to_string(),
        expected_sha256_hex: expected_hash,
        decoded_sha256_hex,
        dropped_symbols,
        attempted_symbol_count: retained.len(),
        recovered,
        details,
        generated_at_unix_ms: now_unix_ms(),
    };

    if let Some(parent) = proof_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(proof_path, serde_json::to_string_pretty(&proof)?)?;

    Ok(proof)
}

fn drop_symbols_by_kind(
    symbols: &[SidecarSymbolRecord],
    kind: &str,
    drop_count: usize,
) -> (Vec<String>, Vec<SidecarSymbolRecord>) {
    let mut dropped_symbols = Vec::new();
    let mut retained = Vec::with_capacity(symbols.len());
    let mut dropped = 0_usize;

    for symbol in symbols {
        if symbol.kind == kind && dropped < drop_count {
            dropped_symbols.push(format!(
                "kind:{}:sbn:{}:esi:{}",
                kind, symbol.sbn, symbol.esi
            ));
            dropped += 1;
        } else {
            retained.push(symbol.clone());
        }
    }

    (dropped_symbols, retained)
}

pub fn read_sidecar_manifest(path: &Path) -> Result<SidecarManifest, DurabilityError> {
    let raw = fs::read_to_string(path)?;
    let manifest = serde_json::from_str::<SidecarManifest>(&raw)?;
    Ok(manifest)
}

fn decode_from_sidecar_records(
    manifest: &SidecarManifest,
    records: &[SidecarSymbolRecord],
) -> Result<Vec<u8>, DurabilityError> {
    validate_manifest(manifest)?;

    let object_id = ObjectId::new(manifest.object_id_high, manifest.object_id_low);
    let object_params = ObjectParams::new(
        object_id,
        manifest.artifact_size as u64,
        manifest.symbol_size,
        manifest.source_blocks as _,
        manifest.symbols_per_block,
    );

    let mut decoder = DecodingPipeline::new(DecodingConfig {
        symbol_size: manifest.symbol_size,
        max_block_size: manifest.max_block_size,
        repair_overhead: 1.0,
        min_overhead: 0,
        max_buffered_symbols: 0,
        block_timeout: Duration::from_secs(30),
        verify_auth: false,
    });
    decoder
        .set_object_params(object_params)
        .map_err(|err| DurabilityError::Decode(err.to_string()))?;

    for record in records {
        let kind = match record.kind.as_str() {
            "source" => SymbolKind::Source,
            "repair" => SymbolKind::Repair,
            other => {
                return Err(DurabilityError::Decode(format!(
                    "unknown symbol kind in sidecar: {other}"
                )));
            }
        };

        let symbol_data = BASE64_STANDARD
            .decode(&record.data_b64)
            .map_err(|err| DurabilityError::Decode(format!("base64 decode failed: {err}")))?;

        let symbol = Symbol::new(
            SymbolId::new(object_id, record.sbn, record.esi),
            symbol_data,
            kind,
        );
        let auth = AuthenticatedSymbol::new_verified(symbol, AuthenticationTag::zero());

        decoder
            .feed(auth)
            .map_err(|err| DurabilityError::Decode(err.to_string()))?;
    }

    decoder
        .into_data()
        .map_err(|err| DurabilityError::Decode(err.to_string()))
}

fn validate_manifest(manifest: &SidecarManifest) -> Result<(), DurabilityError> {
    if manifest.symbol_size == 0 {
        return Err(DurabilityError::InvalidConfig(
            "symbol_size must be non-zero".to_owned(),
        ));
    }
    if manifest.max_block_size == 0 {
        return Err(DurabilityError::InvalidConfig(
            "max_block_size must be non-zero".to_owned(),
        ));
    }
    if manifest.source_blocks == 0 && manifest.artifact_size > 0 {
        return Err(DurabilityError::InvalidConfig(
            "source_blocks must be > 0 for non-empty artifacts".to_owned(),
        ));
    }
    Ok(())
}

fn validate_config(config: &SidecarConfig) -> Result<(), DurabilityError> {
    if config.symbol_size == 0 {
        return Err(DurabilityError::InvalidConfig(
            "symbol_size must be > 0".to_owned(),
        ));
    }
    if config.max_block_size == 0 {
        return Err(DurabilityError::InvalidConfig(
            "max_block_size must be > 0".to_owned(),
        ));
    }
    if config.repair_overhead < 1.0 {
        return Err(DurabilityError::InvalidConfig(
            "repair_overhead must be >= 1.0".to_owned(),
        ));
    }
    Ok(())
}

fn compute_source_blocks(object_size: usize, max_block_size: usize) -> Result<u8, DurabilityError> {
    if object_size == 0 {
        return Ok(0);
    }
    let blocks = object_size.div_ceil(max_block_size);
    u8::try_from(blocks).map_err(|_| {
        DurabilityError::InvalidConfig(format!("source block count {} exceeds u8 range", blocks))
    })
}

fn compute_symbols_per_block(
    object_size: usize,
    max_block_size: usize,
    symbol_size: u16,
) -> Result<u16, DurabilityError> {
    if object_size == 0 {
        return Ok(0);
    }
    let first_block = object_size.min(max_block_size);
    let count = first_block.div_ceil(usize::from(symbol_size));
    u16::try_from(count).map_err(|_| {
        DurabilityError::InvalidConfig(format!("symbols per block {} exceeds u16 range", count))
    })
}

fn object_id_from_sha256(sha256_hex: &str) -> Result<ObjectId, DurabilityError> {
    if sha256_hex.len() < 32 {
        return Err(DurabilityError::InvalidConfig(
            "sha256 hex digest too short".to_owned(),
        ));
    }

    let high = u64::from_str_radix(&sha256_hex[0..16], 16).map_err(|err| {
        DurabilityError::InvalidConfig(format!("invalid high object-id digest: {err}"))
    })?;
    let low = u64::from_str_radix(&sha256_hex[16..32], 16).map_err(|err| {
        DurabilityError::InvalidConfig(format!("invalid low object-id digest: {err}"))
    })?;

    Ok(ObjectId::new(high, low))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{
        SidecarConfig, encode_artifact_to_sidecar, generate_decode_proof, read_sidecar_manifest,
        scrub_sidecar,
    };
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn sidecar_round_trip_and_scrub_pass() {
        let tmp = tempdir().expect("tempdir should build");
        let artifact_path = tmp.path().join("artifact.bin");
        let sidecar_path = tmp.path().join("artifact.sidecar.json");
        let scrub_path = tmp.path().join("artifact.scrub.json");
        let proof_path = tmp.path().join("artifact.proof.json");

        fs::write(&artifact_path, b"frankenjax durability test payload")
            .expect("artifact should write");

        encode_artifact_to_sidecar(&artifact_path, &sidecar_path, &SidecarConfig::default())
            .expect("sidecar generation should succeed");

        let manifest = read_sidecar_manifest(&sidecar_path).expect("manifest should parse");
        assert!(manifest.total_symbols >= manifest.source_symbols);

        let scrub = scrub_sidecar(&sidecar_path, &artifact_path, &scrub_path)
            .expect("scrub should succeed");
        assert!(scrub.decoded_matches_expected);

        let proof = generate_decode_proof(&sidecar_path, &artifact_path, &proof_path, 1)
            .expect("decode proof generation should succeed");
        assert!(proof.recovered);
    }
}
