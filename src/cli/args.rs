use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "dissectlm",
    version,
    about = "Inspect ML model architecture from metadata",
    arg_required_else_help = true
)]
pub struct Cli {
    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    #[arg(long, value_name = "PATH", requires = "deep")]
    pub checkpoint: Option<String>,

    #[arg(long, requires = "model", conflicts_with = "checkpoint")]
    pub params: bool,

    #[arg(long, requires = "model", conflicts_with = "checkpoint")]
    pub graph: bool,

    #[arg(
        long = "attention-breakdown",
        requires = "model",
        conflicts_with = "checkpoint"
    )]
    pub attention_breakdown: bool,

    #[arg(long)]
    pub deep: bool,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Compare {
        model1: String,
        model2: String,
        #[arg(long)]
        deep: bool,
    },
}
