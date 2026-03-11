use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct TensorMetaRaw {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub data_offsets: (u64, u64),
}

#[derive(Debug, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

pub fn read_header_from_file(path: &Path) -> Result<Vec<TensorMetaRaw>> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open safetensors file: {}", path.display()))?;

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .with_context(|| format!("failed to read header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf);

    if header_len == 0 {
        return Ok(Vec::new());
    }

    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf)
        .with_context(|| format!("failed to read header bytes from {}", path.display()))?;

    let header = String::from_utf8(header_buf)
        .with_context(|| format!("invalid utf8 header in {}", path.display()))?;

    parse_header_json(&header)
}

pub fn parse_header_json(header_json: &str) -> Result<Vec<TensorMetaRaw>> {
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(header_json).context("invalid safetensors header json")?;

    let mut out = Vec::new();
    for (name, value) in raw {
        if name == "__metadata" {
            continue;
        }
        let parsed: HeaderTensor = match serde_json::from_value(value) {
            Ok(v) => v,
            Err(_) => continue,
        };
        out.push(TensorMetaRaw {
            name,
            shape: parsed.shape,
            dtype: parsed.dtype,
            data_offsets: (parsed.data_offsets[0], parsed.data_offsets[1]),
        });
    }

    if out.is_empty() {
        bail!("safetensors header did not contain tensor metadata");
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_header_json() {
        let input = r#"{
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [0, 64]
            }
        }"#;

        let tensors = parse_header_json(input).expect("header should parse");
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].shape, vec![4, 4]);
    }

    #[test]
    fn header_reader_ignores_missing_payload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tiny.safetensors");

        let header = r#"{"a":{"dtype":"F32","shape":[4,4],"data_offsets":[0,64]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(&path, bytes).expect("write file");

        let tensors =
            read_header_from_file(&path).expect("header should parse without payload bytes");
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "a");
    }
}
