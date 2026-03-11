use std::path::Path;

use anyhow::{Context, Result, bail};
use reqwest::Client;

pub async fn get_text_cached(client: &Client, url: &str, cache_path: &Path) -> Result<String> {
    if cache_path.exists() {
        return std::fs::read_to_string(cache_path)
            .with_context(|| format!("failed to read cache file: {}", cache_path.display()));
    }

    let body = get_text(client, url).await?;
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create cache directory: {}", parent.display()))?;
    }
    std::fs::write(cache_path, &body)
        .with_context(|| format!("failed to write cache file: {}", cache_path.display()))?;

    Ok(body)
}

pub async fn get_text(client: &Client, url: &str) -> Result<String> {
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("request failed: {url}"))?;

    if !response.status().is_success() {
        bail!("request failed ({}) for {url}", response.status());
    }

    response
        .text()
        .await
        .with_context(|| format!("failed to decode response body for {url}"))
}

pub async fn get_safetensor_header_cached(
    client: &Client,
    url: &str,
    cache_path: &Path,
) -> Result<String> {
    if cache_path.exists() {
        return std::fs::read_to_string(cache_path)
            .with_context(|| format!("failed to read header cache: {}", cache_path.display()));
    }

    let header = get_safetensor_header(client, url).await?;
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create cache directory: {}", parent.display()))?;
    }
    std::fs::write(cache_path, &header)
        .with_context(|| format!("failed to write header cache: {}", cache_path.display()))?;

    Ok(header)
}

async fn get_safetensor_header(client: &Client, url: &str) -> Result<String> {
    let len_response = client
        .get(url)
        .header("Range", "bytes=0-7")
        .send()
        .await
        .with_context(|| format!("range request failed for {url}"))?;

    if !len_response.status().is_success() {
        bail!("failed to fetch safetensors header length for {url}");
    }

    let len_bytes = len_response
        .bytes()
        .await
        .with_context(|| format!("failed to read header length response for {url}"))?;
    if len_bytes.len() < 8 {
        bail!("short header length response for {url}");
    }

    let mut len_buf = [0u8; 8];
    len_buf.copy_from_slice(&len_bytes[..8]);
    let header_len = u64::from_le_bytes(len_buf);

    if header_len == 0 {
        bail!("empty safetensors header for {url}");
    }

    let end = 8u64 + header_len - 1;
    let range_header = format!("bytes=8-{end}");
    let header_response = client
        .get(url)
        .header("Range", range_header)
        .send()
        .await
        .with_context(|| format!("range request for header bytes failed for {url}"))?;

    if !header_response.status().is_success() {
        bail!("failed to fetch safetensors header bytes for {url}");
    }

    let header_bytes = header_response
        .bytes()
        .await
        .with_context(|| format!("failed to read header bytes for {url}"))?;

    if header_bytes.is_empty() {
        bail!("empty safetensors header payload for {url}");
    }

    String::from_utf8(header_bytes.to_vec())
        .with_context(|| format!("safetensors header is not valid utf8 for {url}"))
}
