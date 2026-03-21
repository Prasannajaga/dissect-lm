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

    #[arg(long, global = true, conflicts_with = "json")]
    pub tui: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    #[arg(long, value_name = "PATH", requires = "deep", conflicts_with = "model")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn checkpoint_without_model_is_allowed_in_deep_mode() {
        let cli = Cli::parse_from(["dissectlm", "--deep", "--checkpoint", "/tmp/model.ckpt"]);
        assert!(cli.model.is_none());
        assert_eq!(cli.checkpoint.as_deref(), Some("/tmp/model.ckpt"));
        assert!(cli.deep);
    }

    #[test]
    fn checkpoint_conflicts_with_model() {
        let parsed = Cli::try_parse_from([
            "dissectlm",
            "gpt2",
            "--deep",
            "--checkpoint",
            "/tmp/model.ckpt",
        ]);
        assert!(parsed.is_err());
    }
}
