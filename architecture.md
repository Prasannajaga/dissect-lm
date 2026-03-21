# DissectLM Architecture

## Project Purpose

`dissectlm` is a Rust CLI for model introspection with two modes:

- Fast metadata mode (default): reads `config.json` and `.safetensors` headers only.
- Deep inspection mode (`--deep`): delegates to Python for runtime-level introspection.

Primary goals:

- Avoid loading full model weights in normal usage.
- Give architecture + parameter distribution quickly.
- Provide multiple output modes (text, JSON, TUI).

## General Flow 

```mermaid
flowchart TD
    A[main.rs] --> B[Cli::parse]
    B --> C{subcommand?}

    C -->|none| D[inspect_model]
    C -->|compare| E[compare_models]

    D --> D1{--checkpoint?}
    D1 -->|yes| D2[checkpoint fast-path]
    D2 --> Q[run_deep_inspection]
    D1 -->|no| F[resolve_input]
    F --> G{path exists?}
    G -->|yes| H[resolve_local_input]
    G -->|no| I[resolve_hf_input]

    H --> J[raw tensors + config]
    I --> J

    J --> K[architecture_from_config]
    K --> L[summarize_tensors]
    L --> M{--graph?}
    M -->|yes| N[build_architecture_graph]
    M -->|no| O[skip graph]

    N --> P{--deep?}
    O --> P
    P -->|yes| Q[run_deep_inspection]
    P -->|no| R[skip deep]

    Q --> S[ModelReport]
    R --> S

    S --> T{output mode}
    T -->|text| U[render_model]
    T -->|json| V[render_model_json]
    T -->|tui| W[run_model_tui]

    E --> X[inspect_model left]
    E --> Y[inspect_model right]
    X --> Z[build diffs]
    Y --> Z
    Z --> AA{output mode}
    AA -->|text| AB[render_compare]
    AA -->|json| AC[render_compare_json]
    AA -->|tui| AD[run_compare_tui]
```
 

## Deep Flow Diagram (Mermaid)

```mermaid
sequenceDiagram
    participant User
    participant RustCLI as Rust CLI (commands.rs)
    participant Runner as Python Bridge (runner.rs)
    participant UV as uv
    participant Py as python -m dissectlm.inspector

    User->>RustCLI: dissectlm <model> --deep
    RustCLI->>Runner: run_deep_inspection(model, checkpoint?)
    Runner->>UV: spawn uv run --project python ...
    UV->>Py: start inspector

    alt model mode
        Py->>Py: AutoConfig.from_pretrained
        Py->>Py: collect framework versions
    else checkpoint mode
        Py->>Py: torch.load(checkpoint)
        Py->>Py: summarize tensors/params
    end

    Py-->>Runner: JSON on stdout
    Runner-->>RustCLI: serde_json::Value
    RustCLI->>RustCLI: attach to ModelReport.deep
    RustCLI-->>User: render deep section
```

## Feature Review (Detailed)

### 5.1 Base Inspect (`dissectlm <model>`)

What it does:

- Resolves local vs HF source.
- Reads config + tensor metadata.
- Outputs architecture and param distribution.

Key outputs:

- Model/source summary
- Total params
- Category percentage breakdown
- Architecture inferred fields
- Full raw config keys under `cfg.<key>`

### 5.2 `--params`

Adds:

- Top tensors ranked by param count (up to renderer-defined limit).

When useful:

- Identify largest parameter contributors quickly.

### 5.3 `--graph`

Adds:

- High-level architecture graph section.

How graph is formed:

- Uses inferred model type + layer count.
- Applies coarse block pattern from graph layer map.

### 5.4 `--attention-breakdown`

Adds:

- Q/K/V/O projection parameter totals.

How values are computed:

- Tensor name matching in `param_counter.rs`.
- Aggregated from all matched tensors.

### 5.5 `compare <model1> <model2>`

Runs two independent inspect pipelines, then computes differences for:

- Layers
- Hidden size
- Heads / KV heads
- Attention type
- Total params
- Category percentage metrics

### 5.6 `--json`

Switches renderer from human-focused output to machine-readable JSON.

Use cases:

- CI checks
- dashboard ingestion
- scripted comparisons

### 5.7 `--tui`

Switches renderer to interactive terminal UI.

Behavior:

- If stdout is TTY -> opens ratatui app.
- If not TTY -> logs fallback warning and prints text report.

### 5.8 `--deep`

Purpose:

- Inspect beyond metadata-only flow.
- Access framework-level details and raw checkpoint inspection.

Behavior:

- Rust process spawns Python inspector via `uv`.
- Appends Python JSON result into report under deep section.

Failure handling:

- If `uv`/deps are missing, returns explicit install hint.

### 5.9 `--checkpoint <PATH>` (requires `--deep`)

Purpose:

- Inspect a specific local checkpoint directly (`.pt`, `.bin`, `.ckpt`, etc.).

Behavior:

- Checkpoint mode bypasses metadata resolution (`resolve_input`), so it does not call Hugging Face APIs or local safetensors header parsing.
- Rust builds a deep-only `ModelReport` shell and runs Python bridge with `--checkpoint <PATH>`.
- Python path uses `torch.load`.
- Extracts tensor-level summary even if metadata-only path cannot parse the file.