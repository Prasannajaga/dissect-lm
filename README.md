# dissectlm

`dissectlm` is a fast, metadata-first CLI for inspecting model architecture and parameter distribution without loading full model weights.

## Build

```bash
cargo build --release
```

Run the binary:

```bash
./target/release/dissectlm --help
```

## Command Reference

### `dissectlm <model>`
Inspect a model and print organized summary sections:
- total parameters
- layer distribution
- architecture fields (layers, hidden size, heads, attention type)

Example:

```bash
dissectlm gpt2
dissectlm Qwen/Qwen2.5-Coder-0.5B-Instruct
dissectlm /path/to/local/model_dir
```

### `dissectlm <model> --params`
Show detailed parameter stats and top tensors by parameter count.

Example:

```bash
dissectlm gpt2 --params
```

### `dissectlm <model> --graph`
Show simplified architecture graph section.

Example:

```bash
dissectlm gpt2 --graph
```

### `dissectlm <model> --attention-breakdown`
Show Q/K/V/O projection parameter totals.

Example:

```bash
dissectlm gpt2 --attention-breakdown
```

### `dissectlm compare <model1> <model2>`
Compare two models side-by-side:
- layers
- hidden size
- heads / KV heads
- attention type
- params and category percentages

Example:

```bash
dissectlm compare gpt2 gpt2-medium
dissectlm compare meta-llama/Llama-2-7b-hf Qwen/Qwen2.5-Coder-0.5B-Instruct
```

### `--json`
Return machine-readable JSON instead of rich terminal UI.

Examples:

```bash
dissectlm gpt2 --json
dissectlm gpt2 --params --json
dissectlm compare gpt2 gpt2-medium --json
```

### `--tui`
Open an interactive full-screen TUI (tabs + scrollable sections).

Controls:
- `q` / `Esc`: quit
- `←` / `→` (or `h` / `l`): switch sections
- `j` / `k` (or `↑` / `↓`): scroll
- `PgUp` / `PgDn`: fast scroll

Examples:

```bash
dissectlm gpt2 --tui
dissectlm compare gpt2 gpt2-medium --tui
```

### `--deep` (optional Python path)
Run optional deep inspection through Python bridge (`uv` project under `python/`).

Example:

```bash
dissectlm gpt2 --deep
dissectlm compare gpt2 gpt2-medium --deep
```

### `--checkpoint <PATH>` (deep mode for specific files)
Inspect a specific local checkpoint file (`.pt/.bin/.ckpt`) directly via deep mode.

Example:

```bash
dissectlm --deep --checkpoint /path/to/checkpoint.pt
```

## Common Flag Combinations

```bash
# detailed params + architecture graph
dissectlm gpt2 --params --graph

# params + attention details
dissectlm gpt2 --params --attention-breakdown

# deep mode + JSON output
dissectlm gpt2 --deep --json

# interactive TUI
dissectlm gpt2 --tui
```

## Notes

- Default mode is metadata-first (fast, low memory).
- `--json` is best for automation/CI parsing.
- `--deep` is optional and only needed for deeper framework-level inspection.
```
