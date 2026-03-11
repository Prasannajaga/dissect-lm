pub mod analyzer;
pub mod cli;
pub mod graph;
pub mod hf;
pub mod python_bridge;
pub mod report;
pub mod types;

pub use cli::commands::{CompareOptions, InspectOptions, compare_models, inspect_model};
