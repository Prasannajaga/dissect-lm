use anyhow::{Context, Result, bail};
use indicatif::ProgressBar;
use serde_json::Value;
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;
use tokio::time::{Duration, sleep};

pub async fn run_deep_inspection(
    model: &str,
    checkpoint: Option<&str>,
    progress: Option<ProgressBar>,
) -> Result<Value> {
    let uv_bin = std::env::var("DISSECTLM_UV_BIN").unwrap_or_else(|_| "uv".to_string());

    let mut cmd = Command::new(&uv_bin);
    cmd.args([
        "run",
        "--project",
        "python",
        "python",
        "-m",
        "dissectlm.inspector",
    ]);
    if let Some(path) = checkpoint {
        cmd.args(["--checkpoint", path]);
    } else {
        cmd.args(["--model", model]);
    }
    let child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn();

    let mut child = match child {
        Ok(v) => v,
        Err(err) => {
            bail!(
                "Failed to run deep inspection ({err}). Set up Python deps with `uv sync --project python`."
            )
        }
    };

    let started = Instant::now();
    announce(
        &progress,
        "Running deep inspection: starting Python bridge...",
    );
    if checkpoint.is_some() {
        announce(
            &progress,
            "Running deep inspection: loading checkpoint in Python...",
        );
    } else {
        announce(
            &progress,
            "Running deep inspection: loading model in Python...",
        );
    }

    loop {
        match child
            .try_wait()
            .context("failed while polling deep inspection process")?
        {
            Some(_) => break,
            None => {
                let elapsed = started.elapsed().as_secs();
                set_status(
                    &progress,
                    &format!("Running deep inspection: analyzing weights... {elapsed}s elapsed"),
                );
                sleep(Duration::from_secs(1)).await;
            }
        }
    }

    let output = child
        .wait_with_output()
        .await
        .context("failed to collect deep inspection output")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "Deep inspection failed: {}\nInstall deps with `uv sync --project python`.",
            stderr.trim()
        );
    }

    announce(&progress, "Running deep inspection: parsing results...");

    let stdout = String::from_utf8(output.stdout).context("deep inspector output is not utf8")?;
    let json: Value =
        serde_json::from_str(&stdout).context("deep inspector did not return valid json")?;

    Ok(json)
}

fn announce(progress: &Option<ProgressBar>, message: &str) {
    if let Some(pb) = progress {
        if pb.is_hidden() {
            eprintln!("{message}");
        } else {
            pb.println(format!("• {message}"));
            pb.set_message(message.to_string());
        }
    }
}

fn set_status(progress: &Option<ProgressBar>, message: &str) {
    if let Some(pb) = progress {
        pb.set_message(message.to_string());
    }
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

        let result = run_deep_inspection("gpt2", None, None).await;
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
