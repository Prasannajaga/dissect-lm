use anyhow::Result;
use clap::Parser;

use dissectlm::cli::{args::Cli, commands};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    commands::run(cli).await
}
