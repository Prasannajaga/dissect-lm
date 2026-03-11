use anyhow::{Context, Result, bail};
use serde_json::Value;
use tokio::process::Command;

pub async fn run_deep_inspection(model: &str) -> Result<Value> {
    let uv_bin = std::env::var("DISSECTLM_UV_BIN").unwrap_or_else(|_| "uv".to_string());

    let output = Command::new(&uv_bin)
        .args([
            "run",
            "--project",
            "python",
            "python",
            "-m",
            "dissectlm.inspector",
            "--model",
            model,
        ])
        .output()
        .await;

    let output = match output {
        Ok(v) => v,
        Err(err) => {
            bail!(
                "Failed to run deep inspection ({err}). Set up Python deps with `uv sync --project python`."
            )
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "Deep inspection failed: {}\nInstall deps with `uv sync --project python`.",
            stderr.trim()
        );
    }

    let stdout = String::from_utf8(output.stdout).context("deep inspector output is not utf8")?;
    let json: Value =
        serde_json::from_str(&stdout).context("deep inspector did not return valid json")?;

    Ok(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deep_mode_reports_install_hint_when_uv_missing() {
        let old = std::env::var("DISSECTLM_UV_BIN").ok();
        unsafe {
            std::env::set_var("DISSECTLM_UV_BIN", "definitely_missing_uv_binary");
        }

        let result = run_deep_inspection("gpt2").await;
        let msg = result.expect_err("expected error").to_string();
        assert!(msg.contains("uv sync --project python"));

        if let Some(prev) = old {
            unsafe {
                std::env::set_var("DISSECTLM_UV_BIN", prev);
            }
        } else {
            unsafe {
                std::env::remove_var("DISSECTLM_UV_BIN");
            }
        }
    }
}
