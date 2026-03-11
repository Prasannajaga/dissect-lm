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

    #[arg(long, requires = "model")]
    pub params: bool,

    #[arg(long, requires = "model")]
    pub graph: bool,

    #[arg(long = "attention-breakdown", requires = "model")]
    pub attention_breakdown: bool,

    #[arg(long, requires = "model")]
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
